# -*- coding: utf-8 -*-
from __future__ import annotations

"""
@file   kan/utils/registry.py
@brief  Lightweight, typed, and logger-friendly registry for factories/components.
@date   2025-09-16

@zh
  提供一个**可命名空间**的注册表（Registry/RegistryHub）：
  - `Registry[T]`：泛型注册/获取/构建；支持别名、覆盖策略、冻结、线程安全；
  - `RegistryHub`：跨命名空间管理若干 Registry；
  - `build_from_config()`：从 dict/yaml 配置构建对象（`{"type": name, ...}`）。
  该实现 Windows 友好，仅依赖标准库；日志命名空间为 `kan.utils.registry`。

@en
  A typed, namespaced registry with aliasing, override policy, freeze/unfreeze, and a tiny
  config-driven builder. Thread-safe. No third-party deps. Logger namespace: `kan.utils.registry`.
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
import importlib
import inspect
import logging
import threading
import types
import difflib

LOGGER = logging.getLogger("kan.utils.registry")

T = TypeVar("T")

__all__ = [
    "RegistryError",
    "Registry",
    "RegistryHub",
    "build_from_config",
]


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class RegistryError(RuntimeError):
    """Errors raised by the registry.

    @zh 一般用于重名、未找到、冻结后写入等错误。
    """


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _canonical(key: str, *, case_insensitive: bool) -> str:
    if not isinstance(key, str):
        raise TypeError(f"registry key must be str, got {type(key)!r}")
    return key.lower() if case_insensitive else key


def _iter_aliases(alias: Union[str, Sequence[str], None]) -> Iterable[str]:
    if alias is None:
        return []
    if isinstance(alias, str):
        return [alias]
    return list(alias)


def _callable_name(obj: Any) -> str:
    if hasattr(obj, "__name__"):
        return str(getattr(obj, "__name__"))
    return obj.__class__.__name__


@dataclass
class Entry(Generic[T]):
    name: str
    obj: T
    aliases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        doc = inspect.getdoc(self.obj) or ""
        return (doc.strip().splitlines() or [""])[0]


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------


class Registry(Generic[T]):
    """A typed registry that stores factories/components.

    Features
    --------
    - Aliases & metadata
    - Override policy per-registry and per-add
    - Freeze/unfreeze to protect during training
    - Thread-safe (RLock)
    - Helpful error with fuzzy suggestions

    Example
    -------
    ```python
    MODELS = Registry[Callable[[int], object]]("model")

    @MODELS.register("mlp", alias=["MLP", "MultiLayerPerceptron"])
    class MLP:
        pass

    model = MODELS.build({"type": "mlp"})
    ```
    """

    def __init__(
        self,
        name: str,
        *,
        case_insensitive: bool = True,
        allow_override: bool = False,
    ) -> None:
        self._name = name
        self._case_insensitive = case_insensitive
        self._allow_override_default = allow_override
        self._lock = threading.RLock()
        # primary map
        self._items: Dict[str, Entry[T]] = {}
        # alias -> primary key
        self._alias_to_key: Dict[str, str] = {}
        self._frozen = False

    # ------------------------ basic properties ------------------------
    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return self._name

    @property
    def frozen(self) -> bool:  # pragma: no cover - trivial
        return self._frozen

    def __contains__(self, key: str) -> bool:
        k = _canonical(key, case_insensitive=self._case_insensitive)
        with self._lock:
            return k in self._items or k in self._alias_to_key

    def __len__(self) -> int:  # pragma: no cover - trivial
        with self._lock:
            return len(self._items)

    # ------------------------ mutation control ------------------------
    def freeze(self) -> None:
        with self._lock:
            self._frozen = True

    def unfreeze(self) -> None:
        with self._lock:
            self._frozen = False

    # ------------------------ registration ------------------------
    def register(
        self,
        key: Optional[str] = None,
        *,
        alias: Union[str, Sequence[str], None] = None,
        override: Optional[bool] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Callable[[T], T]:
        """Decorator to register an object.

        Usage:
          @REG.register("name", alias=["n1", "n2"])  def fn(...): ...
          @REG.register()  class Foo: ...  # key defaults to class/function name
        """

        def deco(obj: T) -> T:
            nm = key or _callable_name(obj)
            self.add(
                nm, obj, alias=alias, override=override, metadata=dict(metadata or {})
            )
            return obj

        return deco

    def add(
        self,
        key: str,
        obj: T,
        *,
        alias: Union[str, Sequence[str], None] = None,
        override: Optional[bool] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self._frozen:
            raise RegistryError(
                f"registry '{self._name}' is frozen; cannot add '{key}'"
            )

        override_flag = (
            self._allow_override_default if override is None else bool(override)
        )
        k = _canonical(key, case_insensitive=self._case_insensitive)
        with self._lock:
            if (k in self._items or k in self._alias_to_key) and not override_flag:
                # If k is alias to existing entry, show primary
                primary = self._alias_to_key.get(k, k)
                raise RegistryError(
                    f"key '{key}' already exists in registry '{self._name}' (primary='{primary}')."
                    " Set override=True to replace."
                )

            # Remove stale mappings when overriding
            if override_flag and (k in self._items or k in self._alias_to_key):
                self._remove_no_check(k)

            aliases = [
                _canonical(a, case_insensitive=self._case_insensitive)
                for a in _iter_aliases(alias)
            ]
            # prevent alias collision
            for a in aliases:
                if a in self._items or a in self._alias_to_key:
                    raise RegistryError(
                        f"alias '{a}' conflicts with existing key/alias in registry '{self._name}'"
                    )

            entry = Entry[T](
                name=k, obj=obj, aliases=aliases, metadata=dict(metadata or {})
            )
            self._items[k] = entry
            for a in aliases:
                self._alias_to_key[a] = k

            LOGGER.debug(
                "Registered '%s' into registry '%s' (aliases=%s)",
                key,
                self._name,
                aliases,
            )

    def _remove_no_check(self, key: str) -> None:
        """Remove key or alias without policy checks (internal)."""
        k = _canonical(key, case_insensitive=self._case_insensitive)
        primary = self._alias_to_key.get(k, k)
        entry = self._items.pop(primary, None)
        if entry:
            # purge aliases
            for a in list(entry.aliases):
                self._alias_to_key.pop(a, None)
            self._alias_to_key.pop(primary, None)

    def remove(self, key: str) -> None:
        if self._frozen:
            raise RegistryError(
                f"registry '{self._name}' is frozen; cannot remove '{key}'"
            )
        with self._lock:
            if key not in self:
                raise RegistryError(f"key '{key}' not found in registry '{self._name}'")
            self._remove_no_check(key)

    def clear(self) -> None:
        if self._frozen:
            raise RegistryError(f"registry '{self._name}' is frozen; cannot clear")
        with self._lock:
            self._items.clear()
            self._alias_to_key.clear()

    # ------------------------ retrieval ------------------------
    def keys(self) -> List[str]:
        with self._lock:
            return sorted(self._items.keys())

    def items(self) -> List[Tuple[str, T]]:
        with self._lock:
            return [
                (k, e.obj) for k, e in sorted(self._items.items(), key=lambda kv: kv[0])
            ]

    def entries(self) -> List[Entry[T]]:
        with self._lock:
            return [e for _, e in sorted(self._items.items(), key=lambda kv: kv[0])]

    def get(self, key: str, *, default: Optional[T] = None) -> T:
        k = _canonical(key, case_insensitive=self._case_insensitive)
        with self._lock:
            primary = self._alias_to_key.get(k, k)
            if primary in self._items:
                return self._items[primary].obj

        if default is not None:
            return default

        # Suggest similar keys
        candidates = self.keys() + list(self._alias_to_key.keys())
        sugg = difflib.get_close_matches(k, candidates, n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(sugg)}?" if sugg else ""
        raise RegistryError(f"key '{key}' not found in registry '{self._name}'." + hint)

    def get_entry(self, key: str) -> Entry[T]:
        k = _canonical(key, case_insensitive=self._case_insensitive)
        with self._lock:
            primary = self._alias_to_key.get(k, k)
            if primary in self._items:
                return self._items[primary]
        raise RegistryError(f"key '{key}' not found in registry '{self._name}'")

    def help(self, key: str) -> str:
        """Return the first line of the object's docstring (or empty string)."""
        return self.get_entry(key).describe()

    # ------------------------ building ------------------------
    def build(self, spec: Union[str, Mapping[str, Any]], **kwargs: Any) -> Any:
        """Build an object from name or config.

        - If `spec` is a string → look up and call with `**kwargs` (if callable), else return object.
        - If `spec` is a Mapping → expect a field `type` (or `name`) and treat the rest as params.
        """
        if isinstance(spec, str):
            factory = self.get(spec)
            params = dict(kwargs)
        elif isinstance(spec, Mapping):
            if "type" in spec:
                typ = str(spec["type"])  # primary key or alias
            elif "name" in spec:
                typ = str(spec["name"])  # fallback spelling
            else:
                raise RegistryError("config must contain field 'type' or 'name'")
            factory = self.get(typ)
            params = {k: v for k, v in spec.items() if k not in {"type", "name"}}
            params.update(kwargs)
        else:
            raise TypeError("spec must be str or Mapping")

        if callable(factory):
            return factory(**params)
        # Non-callable stored object: return as is if no params, else error
        if params:
            raise RegistryError(
                f"registered object '{spec}' is not callable but got params {sorted(params.keys())}"
            )
        return factory

    # ------------------------ import helper ------------------------
    def load(
        self,
        dotted: str,
        *,
        register_as: Optional[str] = None,
        alias: Union[str, Sequence[str], None] = None,
        override: Optional[bool] = None,
    ) -> Entry[T]:
        """Import a symbol by dotted path and register it.

        Supports `module:attr` (preferred) or `module.attr`.
        """
        module_name, attr = None, None
        if ":" in dotted:
            module_name, attr = dotted.split(":", 1)
        else:
            parts = dotted.split(".")
            module_name, attr = ".".join(parts[:-1]), parts[-1]
        mod = importlib.import_module(module_name)
        obj = getattr(mod, attr)
        name = register_as or attr
        self.add(name, obj, alias=alias, override=override)
        return self.get_entry(name)

    def __repr__(self) -> str:  # pragma: no cover - human readable
        return f"Registry(name={self._name!r}, size={len(self)})"


# -----------------------------------------------------------------------------
# RegistryHub (namespaces)
# -----------------------------------------------------------------------------


class RegistryHub:
    """Manage multiple registries under namespaces.

    Example
    -------
    ```python
    HUB = RegistryHub()
    encoders = HUB.get_or_create("encoder")
    encoders.add("bag_of_words", BagOfWords)
    ```
    """

    def __init__(
        self, *, case_insensitive: bool = True, allow_override: bool = False
    ) -> None:
        self._registries: Dict[str, Registry[Any]] = {}
        self._case_insensitive = case_insensitive
        self._allow_override = allow_override
        self._lock = threading.RLock()

    def get(self, namespace: str) -> Registry[Any]:
        ns = _canonical(namespace, case_insensitive=True)
        with self._lock:
            if ns not in self._registries:
                raise RegistryError(f"registry namespace '{namespace}' not found")
            return self._registries[ns]

    def get_or_create(self, namespace: str) -> Registry[Any]:
        ns = _canonical(namespace, case_insensitive=True)
        with self._lock:
            reg = self._registries.get(ns)
            if reg is None:
                reg = Registry[Any](
                    ns,
                    case_insensitive=self._case_insensitive,
                    allow_override=self._allow_override,
                )
                self._registries[ns] = reg
                LOGGER.debug("Created registry namespace '%s'", ns)
            return reg

    def namespaces(self) -> List[str]:
        with self._lock:
            return sorted(self._registries.keys())

    def ensure_defaults(self, names: Sequence[str]) -> None:
        for n in names:
            self.get_or_create(n)

    def __repr__(self) -> str:  # pragma: no cover - human readable
        return f"RegistryHub(namespaces={self.namespaces()!r})"


# -----------------------------------------------------------------------------
# Builder helper
# -----------------------------------------------------------------------------


def build_from_config(
    cfg: Mapping[str, Any],
    registry: Registry[Any],
    *,
    type_key: str = "type",
    name_key: str = "name",
    **overrides: Any,
) -> Any:
    """Build with a flexible schema.

    Accepts either `{type: NAME, ...}` or `{name: NAME, ...}`; merges `**overrides`.
    """
    if type_key in cfg:
        spec: Mapping[str, Any] = {**cfg}
    elif name_key in cfg:
        spec = {**cfg}
        spec["type"] = spec.pop(name_key)
    else:
        raise RegistryError(f"config must contain '{type_key}' or '{name_key}'")
    spec = dict(spec)
    spec.update(overrides)
    return registry.build(spec)


# -----------------------------------------------------------------------------
# Global hub with conventional namespaces (optional)
# -----------------------------------------------------------------------------

HUB = RegistryHub()
HUB.ensure_defaults(
    [
        "text_encoder",
        "entity_encoder",
        "context_encoder",
        "head",
        "attention",
        "loss",
        "optimizer",
        "scheduler",
        "dataset",
        "tokenizer",
    ]
)


# -----------------------------------------------------------------------------
# Mini self-test
# -----------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.DEBUG)

    MODELS = Registry("model")

    @MODELS.register("mlp", alias=["MLP"])  # noqa: E302
    class MLP:
        """A tiny demo model."""

        def __init__(self, hidden: int = 128):
            self.hidden = hidden

        def __repr__(self) -> str:  # pragma: no cover
            return f"MLP(hidden={self.hidden})"

    print("Keys:", MODELS.keys())
    print("Help(mlp):", MODELS.help("mlp"))
    print("Build: ", MODELS.build({"type": "MLP", "hidden": 64}))

    # Hub demo
    enc = HUB.get_or_create("encoder")
    enc.add("bag", object)
    print("Hub namespaces:", HUB.namespaces())
