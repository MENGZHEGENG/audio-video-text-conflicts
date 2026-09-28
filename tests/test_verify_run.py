import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "verify_run.py"
_SPEC = importlib.util.spec_from_file_location("conflictbench_verify_run", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
verify_run = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify_run)


def _method(status="verified"):
    return {
        "status": status,
        "metrics": {"accuracy": 0.5, "utility": 0.0},
        "by_mechanism": {"clean": {"accuracy": 0.5}},
    }


def _record(*, dataset="synthetic", temporal_status="verified", include_temporal=True):
    methods = {name: _method() for name in verify_run.BASE_METHODS}
    if include_temporal:
        methods.update({name: _method(temporal_status) for name in verify_run.TEMPORAL_METHODS})
    return {
        "schema": "conflictbench.run.v1",
        "seed": 11,
        "config": {"dataset": dataset},
        "validation": {
            "ok": True,
            "base_methods_verified": True,
            "learned_methods_verified": True,
            "temporal_methods_verified": dataset != "synthetic_temporal" or temporal_status == "verified",
        },
        "splits": {
            "seen": {"methods": methods},
            "unseen": {"methods": methods},
        },
    }


def test_temporal_result_requires_sequence_methods(tmp_path):
    path = tmp_path / "temporal-missing.json"
    record = _record(include_temporal=False)
    record["config"] = {"dataset": "synthetic_temporal"}
    path.write_text(__import__("json").dumps(record), encoding="utf-8")

    ok, errors = verify_run.validate(path)

    assert not ok
    assert any("temporal" in error for error in errors)


def test_temporal_result_requires_verified_sequence_methods(tmp_path):
    path = tmp_path / "temporal-failed.json"
    record = _record(dataset="synthetic_temporal", temporal_status="failed")
    path.write_text(__import__("json").dumps(record), encoding="utf-8")

    ok, errors = verify_run.validate(path)

    assert not ok
    assert any("temporal" in error and "verified" in error for error in errors)


def test_temporal_result_requires_declared_protocol(tmp_path):
    path = tmp_path / "temporal-protocol.json"
    record = _record(dataset="synthetic_temporal")
    record["dataset"] = {"temporal": {"protocol": "legacy"}}
    path.write_text(__import__("json").dumps(record), encoding="utf-8")

    ok, errors = verify_run.validate(path)

    assert not ok
    assert any("protocol" in error for error in errors)


def test_temporal_result_accepts_zero_mean_protocol(tmp_path):
    path = tmp_path / "temporal-protocol-valid.json"
    record = _record(dataset="synthetic_temporal")
    record["dataset"] = {"temporal": {"protocol": verify_run.TEMPORAL_PROTOCOL}}
    path.write_text(__import__("json").dumps(record), encoding="utf-8")

    ok, errors = verify_run.validate(path)

    assert ok, errors


def test_scalar_result_does_not_require_sequence_methods(tmp_path):
    path = tmp_path / "scalar.json"
    path.write_text(__import__("json").dumps(_record(include_temporal=False)), encoding="utf-8")

    ok, errors = verify_run.validate(path)

    assert ok, errors
