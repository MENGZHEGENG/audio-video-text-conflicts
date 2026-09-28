import importlib.util
import json
import sys
from pathlib import Path


def test_temporal_shortcut_runner_output_omits_machine_identity(
    monkeypatch, tmp_path
):
    repository = Path(__file__).resolve().parents[1]
    script = repository / "scripts" / "run_temporal_shortcut_check.py"
    sys.path.insert(0, str(repository / "src"))
    try:
        spec = importlib.util.spec_from_file_location(
            "temporal_shortcut_runner", script
        )
        assert spec is not None and spec.loader is not None
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
    finally:
        sys.path.remove(str(repository / "src"))

    monkeypatch.setattr(runner, "torch_status", lambda: {"cuda_available": True})
    monkeypatch.setattr(
        runner,
        "_load_job",
        lambda _path, _index: ({"name": "smoke"}, {"id": "cell"}, 3),
    )
    monkeypatch.setattr(runner, "_generated_splits", lambda _cell, _seed: ([], {}))
    monkeypatch.setattr(
        runner,
        "_transform",
        lambda train, splits, **_kwargs: (train, splits),
    )
    monkeypatch.setattr(
        runner,
        "_condition_result",
        lambda *_args, **_kwargs: {"score": 0.0},
    )
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--config",
            str(tmp_path / "config.json"),
            "--index",
            "0",
            "--output-dir",
            str(output_dir),
        ],
    )

    runner.main()

    report_path = output_dir / "smoke-gpu-cell-seed3.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "host" not in report
