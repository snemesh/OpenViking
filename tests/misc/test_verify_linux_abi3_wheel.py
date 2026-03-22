import importlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "verify_linux_abi3_wheel.py"


class _BlockYaml(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "yaml" or fullname.startswith("yaml."):
            raise ModuleNotFoundError("No module named 'yaml'")
        return None


def _clear_openviking_modules():
    for name in list(sys.modules):
        if name == "openviking" or name.startswith("openviking."):
            sys.modules.pop(name)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("verify_linux_abi3_wheel", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_verify_script_loads_engine_without_executing_openviking_package_init(
    monkeypatch, tmp_path
):
    site_packages = tmp_path / "site-packages"
    package_root = site_packages / "openviking"
    engine_dir = package_root / "storage" / "vectordb" / "engine"
    engine_dir.mkdir(parents=True)

    (package_root / "__init__.py").write_text("import yaml\n", encoding="utf-8")
    (package_root / "storage" / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "storage" / "vectordb" / "__init__.py").write_text("", encoding="utf-8")
    (engine_dir / "__init__.py").write_text(
        "ENGINE_VARIANT = 'native'\n",
        encoding="utf-8",
    )
    (engine_dir / "_native.py").write_text("BACKEND_NAME = 'native'\n", encoding="utf-8")

    monkeypatch.syspath_prepend(str(site_packages))
    _clear_openviking_modules()
    module = _load_script_module()

    finder = _BlockYaml()
    sys.meta_path.insert(0, finder)
    try:
        resolved_engine_dir = module.resolve_engine_dir()
        engine_module = module.load_engine_module(resolved_engine_dir)
        native_spec = importlib.util.find_spec("openviking.storage.vectordb.engine._native")
        native_module = importlib.import_module("openviking.storage.vectordb.engine._native")
    finally:
        sys.meta_path.remove(finder)

    assert resolved_engine_dir == engine_dir
    assert engine_module.ENGINE_VARIANT == "native"
    assert native_spec is not None
    assert native_spec.origin is not None
    assert native_spec.origin.endswith("_native.py")
    assert native_module.__name__ == "openviking.storage.vectordb.engine._native"
