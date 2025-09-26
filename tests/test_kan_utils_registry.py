# -*- coding: utf-8 -*-
"""
PyTest test suite for `kan.utils.registry` (and fallback to local `registry.py`).

- Covers Registry, RegistryHub, build_from_config, HUB defaults
- Tests aliasing, override policies, freeze/unfreeze, metadata, suggestions,
  case sensitivity, build() with string/mapping, non-callables, dotted-path load,
  concurrency safety, and error branches.
"""
from __future__ import annotations

import sys
import types
import threading
import pathlib
import math
import pytest

# -----------------------------------------------------------------------------
# Import target module with robust fallbacks so the suite works in multiple layouts
# -----------------------------------------------------------------------------
try:  # preferred: installed package path
    from kan.utils.registry import (
        Registry,
        RegistryHub,
        RegistryError,
        build_from_config,
        HUB,
    )
except Exception:  # fallback: same-folder registry.py
    ROOT = pathlib.Path(__file__).resolve().parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from registry import (  # type: ignore
            Registry,
            RegistryHub,
            RegistryError,
            build_from_config,
            HUB,
        )
    except Exception as e:  # as a last resort, try parent dir (for monorepo layouts)
        PARENT = ROOT.parent
        if str(PARENT) not in sys.path:
            sys.path.insert(0, str(PARENT))
        from registry import (  # type: ignore
            Registry,
            RegistryHub,
            RegistryError,
            build_from_config,
            HUB,
        )


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture()
def reg() -> Registry[object]:
    return Registry("test")


@pytest.fixture()
def reg_cs() -> Registry[object]:
    # case sensitive registry
    return Registry("test-cs", case_insensitive=False)


# -----------------------------------------------------------------------------
# Utilities for tests
# -----------------------------------------------------------------------------
class _Adder:
    """Dummy callable class for factory testing.

    The docstring first line is used by `help()`.
    """

    def __init__(self, bias: int = 0):
        self.bias = bias

    def __call__(self, x: int) -> int:
        return x + self.bias


class _Greeter:
    """Greets politely.

    Second line should be ignored by help().
    """

    def __init__(self, name: str = "Klee") -> None:
        self.name = name

    def greet(self) -> str:  # pragma: no cover - trivial behavior
        return f"Hello, {self.name}!"


# -----------------------------------------------------------------------------
# Registry: basic add/get/alias/keys/items/entries
# -----------------------------------------------------------------------------


def test_add_and_get_with_alias(reg: Registry[object]):
    reg.add("adder", _Adder, alias=["sum", "plus"])
    # contains checks across primary and aliases (case-insensitive default)
    assert "adder" in reg
    assert "SUM" in reg
    adder = reg.get("PLUS")(bias=2)  # 先构造实例，把 bias 传给 __init__
    assert adder(1) == 3             # 再把 x 传给 __call__

    # keys() are sorted and contain only primary keys
    assert reg.keys() == ["adder"]

    # items() reflect mapping of primary->object
    items = reg.items()
    assert items == [("adder", _Adder)]

    # entries() expose metadata (empty by default) and aliases
    entries = reg.entries()
    assert entries[0].name == "adder"
    assert set(entries[0].aliases) == {"sum", "plus"}


def test_help_uses_first_doc_line(reg: Registry[object]):
    reg.add("greeter", _Greeter)
    assert reg.help("greeter") == "Greets politely."


# -----------------------------------------------------------------------------
# Registry: register() decorator and metadata
# -----------------------------------------------------------------------------


def test_register_decorator_and_metadata(reg: Registry[object]):
    @reg.register(alias=["a1", "A2"], metadata={"family": "math"})
    class Foo:  # noqa: D401 - short demo class
        """Foo doc."""

    # key defaulted to callable/class name
    assert reg.get("foo") is Foo
    entry = reg.get_entry("A1")  # alias caseless
    assert entry.metadata["family"] == "math"


# -----------------------------------------------------------------------------
# Registry: override policy and collisions
# -----------------------------------------------------------------------------


def test_duplicate_key_without_override_raises(reg: Registry[object]):
    reg.add("x", object)
    with pytest.raises(RegistryError) as ei:
        reg.add("X", object)  # same canonical key
    assert "already exists" in str(ei.value)


def test_override_by_flag_and_cleanup_aliases(reg: Registry[object]):
    reg.add("foo", object, alias=["bar", "baz"])
    # overriding via alias removes the original primary entry
    reg.add("BAR", int, override=True)  # using alias as the key
    assert reg.get("bar") is int
    # old primary and aliases should be gone
    with pytest.raises(RegistryError):
        reg.get("foo")
    with pytest.raises(RegistryError):
        reg.get("baz")


def test_allow_override_default_on_registry():
    reg = Registry("ovr", allow_override=True)
    reg.add("k", int)
    reg.add("K", float)  # override allowed by default
    assert reg.get("k") is float


def test_alias_collision_is_detected(reg: Registry[object]):
    reg.add("p", int)
    with pytest.raises(RegistryError) as ei:
        reg.add("q", float, alias=["P"])  # alias collides with existing key
    assert "conflicts" in str(ei.value)


# -----------------------------------------------------------------------------
# Registry: freeze/unfreeze and destructive ops
# -----------------------------------------------------------------------------


def test_freeze_blocks_mutations(reg: Registry[object]):
    reg.freeze()
    with pytest.raises(RegistryError):
        reg.add("z", object)
    with pytest.raises(RegistryError):
        reg.remove("z")
    with pytest.raises(RegistryError):
        reg.clear()
    reg.unfreeze()  # should unblock
    reg.add("z", object)
    assert "z" in reg


# -----------------------------------------------------------------------------
# Registry: get()/get_entry() not found + suggestion text
# -----------------------------------------------------------------------------


def test_get_not_found_suggests_similar(reg: Registry[object]):
    reg.add("mlp", object, alias=["MultiLayerPerceptron"])  # improve similarity surface
    with pytest.raises(RegistryError) as ei:
        reg.get("multilayer")
    msg = str(ei.value)
    # best-effort check: implementation uses difflib, so we just check the hint stub
    assert "Did you mean" in msg


# -----------------------------------------------------------------------------
# Registry: remove/clear semantics including alias removal and miss cases
# -----------------------------------------------------------------------------


def test_remove_by_alias_and_clear(reg: Registry[object]):
    reg.add("a", object, alias=["aa"])
    reg.remove("AA")  # remove via alias (case-insensitive)
    assert "a" not in reg
    reg.add("b", object)
    reg.add("c", object)
    reg.clear()
    assert len(reg.keys()) == 0


# -----------------------------------------------------------------------------
# Registry: case sensitivity variant
# -----------------------------------------------------------------------------


def test_case_sensitive_variant(reg_cs: Registry[object]):
    reg_cs.add("Name", int)
    assert "name" not in reg_cs  # different key when case-sensitive
    reg_cs.add("name", float)
    assert reg_cs.get("Name") is int
    assert reg_cs.get("name") is float


# -----------------------------------------------------------------------------
# Registry: build() behavior with callable and non-callable
# -----------------------------------------------------------------------------


def test_build_from_string_and_mapping(reg: Registry[object]):
    reg.add("adder", _Adder)
    # by string + kwargs
    obj1 = reg.build("adder", bias=3)
    assert isinstance(obj1, _Adder) and obj1.bias == 3

    # by mapping with type, plus extra kwargs
    obj2 = reg.build({"type": "Adder", "bias": 7})
    assert isinstance(obj2, _Adder) and obj2.bias == 7

    # by mapping with name (fallback key spelling)
    obj3 = reg.build({"name": "ADDER"}, bias=9)
    assert isinstance(obj3, _Adder) and obj3.bias == 9


def test_build_non_callable_and_param_error(reg: Registry[object]):
    reg.add("pi", math.pi)  # non-callable
    assert reg.build("pi") == math.pi  # no params
    with pytest.raises(RegistryError) as ei:
        reg.build("pi", extra=True)
    assert "not callable" in str(ei.value)


# -----------------------------------------------------------------------------
# Registry: load() by dotted path (module:attr and module.attr)
# -----------------------------------------------------------------------------


def test_load_dotted_path_with_register_as_and_alias(reg: Registry[object]):
    # build a tiny ephemeral module
    mod = types.ModuleType("tmpmod_hello")

    def foo(x: int = 1):
        """Return x+1."""
        return x + 1

    mod.foo = foo
    sys.modules[mod.__name__] = mod

    # module:attr form
    e1 = reg.load(
        f"{mod.__name__}:foo", register_as="inc", alias=["INC"], override=None
    )
    assert e1.name == "inc" and reg.get("inc") is foo and reg.get("inc")(3) == 4

    # module.attr form
    e2 = reg.load(f"{mod.__name__}.foo", register_as="inc2")
    assert e2.name == "inc2" and reg.get("inc2")(4) == 5


# -----------------------------------------------------------------------------
# Registry: internal canonicalization type guard (TypeError path)
# -----------------------------------------------------------------------------


def test_contains_with_non_str_key_raises_type_error(reg: Registry[object]):
    with pytest.raises(TypeError):
        123 in reg  # type: ignore[operator]


# -----------------------------------------------------------------------------
# RegistryHub: creation, retrieval, namespaces, propagation of flags
# -----------------------------------------------------------------------------


def test_registry_hub_get_or_create_and_flags():
    hub = RegistryHub(case_insensitive=False, allow_override=True)
    r1 = hub.get_or_create("Encoder")
    # flags must propagate to created registries
    assert r1.frozen is False
    r1.add("X", int)
    r1.add(
        "x", float
    )  # allowed override default because hub passed allow_override=True
    assert r1.get("x") is float

    # namespaces()
    assert hub.namespaces() == ["encoder"]  # canonicalized lower

    # get() error path
    with pytest.raises(RegistryError):
        hub.get("missing")


# -----------------------------------------------------------------------------
# build_from_config(): flexible schema + overrides merging
# -----------------------------------------------------------------------------


def test_build_from_config_helpers():
    reg = Registry("bfc")
    reg.add("adder", _Adder)

    obj = build_from_config({"type": "adder", "bias": 5}, reg)
    assert isinstance(obj, _Adder) and obj.bias == 5

    obj2 = build_from_config({"name": "Adder"}, reg, bias=11)
    assert isinstance(obj2, _Adder) and obj2.bias == 11

    with pytest.raises(RegistryError):
        build_from_config({}, reg)


# -----------------------------------------------------------------------------
# Concurrency: thread-safety smoke test under RLock
# -----------------------------------------------------------------------------


def test_thread_safety_registration(reg: Registry[object]):
    N = 32

    def worker(i: int) -> None:
        reg.add(f"k{i}", object)
        # also test alias addition and retrieval concurrently
        reg.add(f"t{i}", int, alias=[f"T{i}"])  # distinct
        assert reg.get(f"t{i}") is int

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    # All keys should be present and no deadlock
    for i in range(N):
        assert f"k{i}" in reg
        assert reg.get(f"T{i}") is int


# -----------------------------------------------------------------------------
# HUB defaults from the target module (non-destructive check)
# -----------------------------------------------------------------------------


def test_global_hub_has_conventional_namespaces():
    # The module under test initializes HUB with several conventional namespaces.
    # We only check that a representative subset exists to avoid brittle coupling.
    existing = set(HUB.namespaces())
    expected_subset = {
        "dataset",
        "optimizer",
        "scheduler",
        "tokenizer",
    }
    assert expected_subset <= existing
