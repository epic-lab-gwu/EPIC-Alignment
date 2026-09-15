import pytest

from epa.alignment.modes import public_align_mode, resolve_metric_eval_align_mode


def test_public_align_mode_collapses_compatibility_aliases() -> None:
    assert public_align_mode("epa_step3") == "se3"
    assert public_align_mode("epa_se3_eval") == "se3"
    assert public_align_mode("se3r") == "se3r"
    assert public_align_mode("epa_se3r") == "se3r"
    assert public_align_mode("rotation_first_se3") == "se3r"
    assert public_align_mode("se3-original") == "se3"
    assert public_align_mode("se3-orginal") == "se3"
    assert public_align_mode("umeyama_se3") == "se3"
    assert public_align_mode("epa_posyaw") == "posyaw"
    assert public_align_mode("epica_sim3_stable") == "sim3"
    assert public_align_mode("epa_sim3_v2") == "sim3"


def test_metric_eval_align_mode_preserves_advanced_modes_by_default() -> None:
    assert resolve_metric_eval_align_mode("epa_step3") == "none"
    assert resolve_metric_eval_align_mode("epa_se3") == "se3"
    assert resolve_metric_eval_align_mode("se3") == "se3"
    assert resolve_metric_eval_align_mode("epa_se3r") == "se3r"
    assert resolve_metric_eval_align_mode("rotation_first_se3") == "se3r"
    assert resolve_metric_eval_align_mode("se3-original") == "se3"
    assert resolve_metric_eval_align_mode("se3-orginal") == "se3"
    assert resolve_metric_eval_align_mode("epa_posyaw") == "posyaw"
    assert resolve_metric_eval_align_mode("epa_sim3") == "sim3"
    assert resolve_metric_eval_align_mode("epica_sim3_stable") == "epica_sim3_stable"


def test_metric_eval_align_mode_can_collapse_to_public_sim3() -> None:
    assert (
        resolve_metric_eval_align_mode("epica_sim3_stable", collapse_sim3_aliases=True)
        == "sim3"
    )


def test_metric_eval_align_mode_can_treat_step3_as_public_se3() -> None:
    assert resolve_metric_eval_align_mode("epa_step3", step3_as_se3=True) == "se3"


def test_unknown_mode_can_be_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported alignment mode"):
        public_align_mode("surprise", strict=True)
