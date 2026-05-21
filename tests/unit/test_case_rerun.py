from pathlib import Path

import pytest

from epa.benchmark import case_rerun


def _write_dummy_pair(prepared_dir: Path, case_id: str) -> None:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    (prepared_dir / f"{case_id}__gt.tum").write_text("0 0 0 0 0 0 0 1\n1 0 0 0 0 0 0 1\n", encoding="utf-8")
    (prepared_dir / f"{case_id}__est.tum").write_text("0 0 0 0 0 0 0 1\n1 0 0 0 0 0 0 1\n", encoding="utf-8")


def test_resolve_latest_run_dir_from_parent(tmp_path: Path) -> None:
    root = tmp_path / "outputs" / "benchmark_harness"
    old_run = root / "run_20260417_120000"
    new_run = root / "run_20260417_130000"
    _write_dummy_pair(old_run / "prepared_tum", "old_case")
    _write_dummy_pair(new_run / "prepared_tum", "new_case")

    resolved = case_rerun._resolve_run_dir(root)
    assert resolved == new_run.resolve()


def test_run_builds_and_executes_epa_cmd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "outputs" / "benchmark_harness" / "run_20260417_130000"
    case_id = "euroc_mav_MH_01_easy_rovio"
    _write_dummy_pair(run_dir / "prepared_tum", case_id)

    parser = case_rerun.build_parser()
    args = parser.parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--case",
            case_id,
            "--python-bin",
            "/usr/bin/python3",
        ]
    )

    captured = {"cmd": None}

    class _Proc:
        returncode = 0

    def _fake_run(cmd, check=False):  # noqa: ARG001
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(case_rerun.subprocess, "run", _fake_run)
    rc = case_rerun.run(args, passthrough=["--t-max-diff", "0.02"])
    assert rc == 0
    cmd = captured["cmd"]
    assert cmd is not None
    assert cmd[:3] == ["/usr/bin/python3", "-m", "epa.cli"]
    assert cmd[3:5] == [
        str(run_dir / "prepared_tum" / f"{case_id}__gt.tum"),
        str(run_dir / "prepared_tum" / f"{case_id}__est.tum"),
    ]
    assert "--plot" in cmd
    assert "--rerun" in cmd
    assert cmd[-2:] == ["--t-max-diff", "0.02"]


def test_run_requires_case_when_multiple_pairs(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "benchmark_harness" / "run_20260417_130000"
    _write_dummy_pair(run_dir / "prepared_tum", "case_a")
    _write_dummy_pair(run_dir / "prepared_tum", "case_b")

    parser = case_rerun.build_parser()
    args = parser.parse_args(["--run-dir", str(run_dir)])
    with pytest.raises(ValueError, match="--case"):
        case_rerun.run(args, passthrough=[])
