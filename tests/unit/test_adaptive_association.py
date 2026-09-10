import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from epa.core.adaptive_association import associate_adaptive, detect_resets, interpolate_supported, reset_crossing_mask
from epa.core.evaluation import _longest_successful_group, compute_rpe
from epa.compat.ov_eval.evaluate import _compute_valid_segment_summary
from epa.core.solve_eval import _prepare_solve_eval_trajectories
from epa.core.calibration import solve_extrinsic_rotation_multibaseline


def poses(t):
    t = np.asarray(t)
    return np.column_stack((t, t*0, t*0)), Rotation.from_rotvec(np.column_stack((t*0, t*0, t*.05))).as_quat()


@pytest.mark.parametrize('est_dt,gt_dt', [(.05,.5), (.5,.05), (.1,.1)])
def test_samples_slower_trajectory_without_extrapolation(est_dt, gt_dt):
    te, tg = np.arange(0,10,est_dt), np.arange(.013,9.99,gt_dt)
    pe, qe = poses(te); pg, qg = poses(tg)
    t, pe_m, qe_m, _, pg_m, qg_m, _ = associate_adaptive(te,pe,qe,tg,pg,qg)
    np.testing.assert_allclose(pe_m,pg_m,atol=1e-12)
    np.testing.assert_allclose(qe_m,qg_m,atol=1e-12)
    assert t.min() >= max(te.min(),tg.min()) and t.max() <= min(te.max(),tg.max())
    assert len(t) <= min(len(te),len(tg))
    expected = tg if gt_dt > est_dt else te
    assert np.all(np.isin(t,expected))


def test_switches_direction_locally_and_deduplicates():
    te = np.arange(0,20,.1)
    tg = np.r_[np.arange(0,10,.02),np.arange(10,20,.5)]
    pe,qe=poses(te);pg,qg=poses(tg)
    t,*_=associate_adaptive(te,pe,qe,tg,pg,qg)
    assert np.all(np.diff(t)>0)
    np.testing.assert_allclose(np.median(np.diff(t[(t>2)&(t<8)])),.1)
    np.testing.assert_allclose(np.median(np.diff(t[(t>12)&(t<18)])),.5)


@pytest.mark.parametrize('long_gap,jump,expected',[(False,True,False),(True,False,False),(True,True,True)])
def test_reset_requires_both_long_gap_and_motion(long_gap,jump,expected):
    t=np.arange(100)*.05
    if long_gap:t[50:]+=1.
    p,q=poses(t)
    if jump:p[50:,0]+=100.
    resets=detect_resets(t,p,q)
    assert bool(resets)==expected
    if expected:
        assert resets[0]['index']==49
        query=np.linspace(t[49],t[50],31)
        ids,*_=interpolate_supported(t,p,q,query,resets=resets)
        np.testing.assert_array_equal(ids,[0,30])


def test_regular_widely_spaced_keyframes_are_not_resets():
    t=np.arange(50)*10.;p,q=poses(t);p[25:,0]+=1000
    assert detect_resets(t,p,q)==[]


def test_rotation_reset_requires_gap():
    t=np.arange(100)*.05;p,q=poses(t)
    q[50:]=(Rotation.from_euler('x',120,degrees=True)*Rotation.from_quat(q[50:])).as_quat()
    assert not detect_resets(t,p,q)
    t[50:]+=1
    assert detect_resets(t,p,q)[0]['index']==49


def test_short_reset_gap_joins_groups_but_cannot_score_rpe():
    t=np.r_[np.arange(0,3,.1),np.arange(4,6,.1)];p,q=poses(t)
    resets=[dict(start_s=t[29],end_s=t[30])]
    blocked=reset_crossing_mask(t,resets)
    result=_compute_valid_segment_summary(t,p,q,p,q,input_coverage={'reset_intervals':resets})
    mask=result['success']['valid_segment_mask']
    assert np.all(mask[:29]) and not mask[29] and np.all(mask[30:])
    assert result['success']['reset_blocked_interval_count']==1
    block=compute_rpe(p,q,p,q,delta=2,delta_unit='s',timestamps=t,include_raw=True,valid_segment_mask=~blocked)
    assert len(block['_pair_ids'])>0
    for i,j in block['_pair_ids']:assert not np.any(blocked[i:j])


def test_dense_fallback_cannot_bridge_reset_or_extrapolate():
    te=np.r_[np.arange(0,3,.1),np.arange(4,6,.1)];pe,qe=poses(te);pe[30:,0]+=100
    tg=np.arange(-1,7,.05);pg,qg=poses(tg)
    out=_prepare_solve_eval_trajectories(t_gt=tg,pos_gt=pg,quat_gt=qg,t_est=te,pos_est=pe,quat_est=qe,calculated_offset=0.,downsample_hz=0,quat_interp='slerp',safe_association=True)
    t=out['t_gt'];assert not np.any((t>te[29])&(t<te[30]))
    assert t.min()>=te.min() and t.max()<=te.max()


def test_calibration_pairs_respect_explicit_boundary():
    t=np.arange(200)*.05
    q=Rotation.from_rotvec(np.column_stack((.3*np.sin(t),.4*np.cos(t*.7),.2*t))).as_quat()
    _,pairs,mask,info=solve_extrinsic_rotation_multibaseline(q,q,timestamps_s=t,reset_intervals=[dict(start_s=t[99],end_s=t[100])])
    assert len(pairs)>0
    assert not np.any((pairs[:,0]<=99)&(pairs[:,1]>=100))


def test_gt_outage_is_not_interpolated():
    tg=np.r_[np.arange(0,3,.05),np.arange(5,8,.05)];pg,qg=poses(tg)
    ids,*_=interpolate_supported(tg,pg,qg,np.array([2.,3.5,4.,6.]))
    np.testing.assert_array_equal(ids,[0,3])

@pytest.mark.parametrize('gt_dt, expected', [(.01, False), (.3, True)])
def test_reset_checks_only_sparse_gt_interpolation(gt_dt, expected):
    from epa.core.adaptive_association import detect_interpolation_resets
    te = np.arange(100)*.05
    te[50:] += 1.
    p,q = poses(te); p[50:,0] += 100
    tg = np.arange(.013, te[-1], gt_dt)
    assert bool(detect_interpolation_resets(te,p,q,tg)) == expected


def test_exact_matches_do_not_request_reset_checks():
    from epa.core.adaptive_association import detect_interpolation_resets
    te = np.arange(100)*.05; te[50:] += 1.
    p,q = poses(te); p[50:,0] += 100
    assert detect_interpolation_resets(te,p,q,te[::4]) == []


def test_reset_checks_follow_local_rate_switch():
    from epa.core.adaptive_association import detect_interpolation_resets
    te = np.arange(800)*.05; te[200:] += 1.; te[600:] += 1.
    p,q = poses(te); p[200:,0] += 100; p[600:,0] += 100
    tg = np.r_[np.arange(.013,20,.01), np.arange(20.013,te[-1],.3)]
    assert [r['index'] for r in detect_interpolation_resets(te,p,q,tg)] == [599]


def test_chained_sr_covers_every_supported_interval_once():
    from epa.core.evaluation import rpe_pairs_chained
    t = np.array([0.,.31,.92,1.37,2.08,3.6,3.7,4.9,5.1])
    pairs = rpe_pairs_chained(t)
    assert pairs[0][0] == 0 and pairs[-1][1] == len(t)-1
    assert all(a[1] == b[0] for a,b in zip(pairs,pairs[1:]))
    coverage = np.zeros(len(t)-1, int)
    for i,j in pairs:
        coverage[i:j] += 1
        assert j == i+1+np.argmin(abs(t[i+1:]-t[i]-1.))
    assert np.all(coverage == 1)
    valid = np.ones(len(t)-1, bool); valid[3] = False
    coverage[:] = 0
    for i,j in rpe_pairs_chained(t, valid_segment_mask=valid):
        coverage[i:j] += 1
    np.testing.assert_array_equal(coverage, valid.astype(int))
    with pytest.raises(ValueError):
        rpe_pairs_chained([0.,1.,1.])


def test_stable_alignment_step_split_selects_longest_window():
    from epa.core.trajectory_alignment import _select_stable_alignment_window
    p,q = poses(np.arange(200)*.1)
    estimate = p.copy(); estimate[30:,0] += 1000
    mask, info = _select_stable_alignment_window(p, estimate)
    assert not np.any(mask[:30]) and np.all(mask[30:])
    assert info['step3_stable_segment_used'] == 1
    assert info['step3_stable_solve_count'] == 170


@pytest.mark.parametrize("split,expected_start,expected_count", [(70, 0, 70), (50, 0, 50), (30, 30, 70)])
def test_stable_window_longest_and_tie_selection(split, expected_start, expected_count):
    from epa.core.trajectory_alignment import _select_stable_alignment_window
    p, _ = poses(np.arange(100) * .1)
    estimate = p.copy()
    estimate[split:, 0] += 1000
    mask, info = _select_stable_alignment_window(p, estimate)
    assert np.flatnonzero(mask).tolist() == list(range(expected_start, expected_start + expected_count))
    assert info['step3_stable_solve_count'] == expected_count
