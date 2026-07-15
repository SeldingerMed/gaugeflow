import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments_v2"
    / "brats_seq_gauge"
    / "prepare_brats_cases.py"
)


def load_prepare_module(monkeypatch, data_root: Path | None = None):
    if data_root is None:
        monkeypatch.delenv("DATA_ROOT", raising=False)
    else:
        monkeypatch.setenv("DATA_ROOT", str(data_root))

    spec = importlib.util.spec_from_file_location("prepare_brats_cases", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_schema_path_expands_data_root_env(monkeypatch, tmp_path):
    data_root = tmp_path / "data-root"

    module = load_prepare_module(monkeypatch, data_root)

    assert module.DEFAULT_SCHEMA_PATH == (
        data_root
        / "experiments"
        / "main"
        / "run-coin-mpmri-msd-task01-stream-schema-v2"
        / "stream_msd_task01_schema.py"
    )
    assert "$" not in str(module.DEFAULT_SCHEMA_PATH)


def test_parser_allows_schema_path_override(monkeypatch, tmp_path):
    module = load_prepare_module(monkeypatch)
    schema_path = tmp_path / "custom_schema.py"

    args = module.parse_args(["--schema-path", str(schema_path)])

    assert args.schema_path == schema_path
