from epa.openvins_runner import _map_align_mode_to_eval_align


def test_openvins_runner_maps_epa_modes_without_extra_se3_alignment() -> None:
    assert _map_align_mode_to_eval_align("epa_step3") == "epa_step3"
    assert _map_align_mode_to_eval_align("se3") == "epa_step3"
    assert _map_align_mode_to_eval_align("epa_se3") == "epa_se3"
    assert _map_align_mode_to_eval_align("epa_se3_eval") == "epa_se3"
    assert _map_align_mode_to_eval_align("sim3") == "sim3"
    assert _map_align_mode_to_eval_align("ov_sim3") == "ov_sim3"
    assert _map_align_mode_to_eval_align("posyaw") == "posyaw"
    assert _map_align_mode_to_eval_align("se3single") == "origin"
    assert _map_align_mode_to_eval_align("posyawsingle") == "origin"
