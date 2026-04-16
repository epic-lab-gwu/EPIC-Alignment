from __future__ import annotations
from pathlib import Path

from epa.benchmark.latex_summary import load_summary_rows, write_latex_tables


def _make_row(index: int, dataset: str) -> dict[str, object]:
    return {
        "case": f"case_{index:03d}",
        "dataset": dataset,
        "method": "rovio" if index % 2 == 0 else "svo_stereo",
        "epa_status": "ok",
        "evo_status": "ok",
        "epa_ate_rmse_step3_m": 0.5 + 0.01 * index,
        "evo_ape_se3_rmse_m": 0.7 + 0.01 * index,
        "epa_improve_pct": 40.0 + 0.1 * index,
        "evo_improve_pct": 30.0 + 0.1 * index,
        "epa_offset_est_s": 0.1,
        "evo_offset_s": 0.2,
    }


def test_write_latex_tables_short_case(tmp_path: Path) -> None:
    rows = [_make_row(0, "euroc_mav"), _make_row(1, "grand_tour")]

    produced = write_latex_tables(rows, tmp_path, max_main_rows=10)

    assert produced["main_table"].exists()
    assert produced["dataset_table"].exists()
    assert produced["appendix_full_table"].exists()

    main_text = produced["main_table"].read_text(encoding="utf-8")
    dataset_text = produced["dataset_table"].read_text(encoding="utf-8")
    appendix_text = produced["appendix_full_table"].read_text(encoding="utf-8")

    assert "\\begin{table}" in main_text
    assert "Main-paper per-case metrics" in main_text
    assert "Showing 2 of" not in main_text
    assert "Dataset-level aggregate metrics" in dataset_text
    assert "\\begin{longtable}" in appendix_text


def test_write_latex_tables_truncates_main_table(tmp_path: Path) -> None:
    rows = [_make_row(i, f"dataset_{i % 3}") for i in range(30)]

    produced = write_latex_tables(rows, tmp_path, max_main_rows=8)
    main_text = produced["main_table"].read_text(encoding="utf-8")

    assert "Showing 8 of 30 comparable cases" in main_text
    assert main_text.count("case\\_") == 8


def test_write_latex_tables_supports_legacy_vicon_columns(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    summary_csv.write_text(
        "\n".join(
            [
                "case,dataset,method,vicon_status,evo_status,vicon_offset_est_s,evo_offset_s,vicon_ate_rmse_step3_m,evo_ape_se3_rmse_m,vicon_improve_pct,evo_improve_pct",
                "legacy_case,euroc_mav,rovio,ok,ok,1.23,0.20,0.456,0.789,42.0,15.0",
            ]
        ),
        encoding="utf-8",
    )

    rows = load_summary_rows(summary_csv)
    produced = write_latex_tables(rows, tmp_path / "paper_tables", max_main_rows=10)

    main_text = produced["main_table"].read_text(encoding="utf-8")
    dataset_text = produced["dataset_table"].read_text(encoding="utf-8")
    appendix_text = produced["appendix_full_table"].read_text(encoding="utf-8")

    assert "legacy\\_case" in main_text
    assert "0.456" in main_text
    assert "(none)" not in dataset_text
    assert "euroc\\_mav" in dataset_text
    assert " & ok & ok & " in appendix_text
