from __future__ import annotations

import numpy as np
import pytest

from epa.ov_io import load_ov_txt


def test_load_ov_txt_regular_whitespace_table(tmp_path) -> None:
    path = tmp_path / "poses.txt"
    path.write_text(
        "# timestamp tx ty tz qx qy qz qw\n"
        "1.0 1 2 3 0 0 0 2 ignored\n"
        "2.0 4 5 6 0 0 1 0 ignored\n",
        encoding="utf-8",
    )

    stamps, positions, quaternions = load_ov_txt(path)

    np.testing.assert_allclose(stamps, [1.0, 2.0])
    np.testing.assert_allclose(positions, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(quaternions, [[0, 0, 0, 1], [0, 0, 1, 0]])


def test_load_ov_txt_falls_back_for_commas_and_ragged_rows(tmp_path) -> None:
    path = tmp_path / "legacy.txt"
    path.write_text(
        "short row\n"
        "1.0,1,2,3,0,0,0,1\n"
        "2.0,4,5,6,0,0,1,0,extra\n",
        encoding="utf-8",
    )

    stamps, positions, quaternions = load_ov_txt(path)

    np.testing.assert_allclose(stamps, [1.0, 2.0])
    np.testing.assert_allclose(positions, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(quaternions, [[0, 0, 0, 1], [0, 0, 1, 0]])


def test_load_ov_txt_rejects_empty_input(tmp_path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("# no samples\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not parse any trajectory samples"):
        load_ov_txt(path)
