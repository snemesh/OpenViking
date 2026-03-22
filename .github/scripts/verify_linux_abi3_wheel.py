#!/usr/bin/env python3
"""Smoke-test the Linux abi3 vectordb engine wheel contents.

This script intentionally avoids importing ``openviking.__init__`` so the check
only verifies the packaged vectordb runtime loader and native backend.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType

ENGINE_MODULE_NAME = "openviking.storage.vectordb.engine"


def resolve_engine_dir() -> Path:
    package_spec = importlib.util.find_spec("openviking")
    if package_spec is None or not package_spec.submodule_search_locations:
        raise SystemExit("openviking package was not installed")

    package_root = Path(next(iter(package_spec.submodule_search_locations))).resolve()
    engine_dir = package_root / "storage" / "vectordb" / "engine"
    if not engine_dir.is_dir():
        raise SystemExit(f"vectordb engine package was not installed: {engine_dir}")
    return engine_dir


def _install_package_stub(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module

    module.__file__ = str(path / "__init__.py")
    module.__package__ = name
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    module.__spec__.submodule_search_locations = [str(path)]


def _install_parent_package_stubs(engine_dir: Path) -> None:
    package_root = engine_dir.parents[2]
    _install_package_stub("openviking", package_root)
    _install_package_stub("openviking.storage", package_root / "storage")
    _install_package_stub("openviking.storage.vectordb", package_root / "storage" / "vectordb")


def load_engine_module(engine_dir: Path) -> ModuleType:
    _install_parent_package_stubs(engine_dir)

    existing = sys.modules.get(ENGINE_MODULE_NAME)
    if existing is not None:
        return existing

    init_file = engine_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        ENGINE_MODULE_NAME,
        init_file,
        submodule_search_locations=[str(engine_dir)],
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to create import spec for {init_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[ENGINE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    engine_dir = resolve_engine_dir()
    engine = load_engine_module(engine_dir)

    native_spec = importlib.util.find_spec(f"{ENGINE_MODULE_NAME}._native")
    if native_spec is None or native_spec.origin is None:
        raise SystemExit("openviking storage native backend extension was not installed")

    native_module = importlib.import_module(f"{ENGINE_MODULE_NAME}._native")
    if getattr(engine, "ENGINE_VARIANT", None) != "native":
        raise SystemExit(
            f"expected native engine variant on Linux abi3 wheel, got {engine.ENGINE_VARIANT}"
        )

    print(f"Loaded runtime engine variant {engine.ENGINE_VARIANT}")
    print(f"Loaded native extension from {native_spec.origin}")
    print(f"Imported backend module {native_module.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
