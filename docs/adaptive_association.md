# Adaptive EPA association

EPA's interpolating association samples the locally slower trajectory. By default
it interpolates GT at estimate timestamps; where GT is slower, it interpolates
VIO at GT timestamps. Local periods use up to ten neighboring intervals on each
side, excluding the interval being tested. Equal rates prefer estimate times.
Exact matches remain valid. Interpolation uses linear positions and quaternion
SLERP, requires a bracket no longer than three local source periods, and never
extrapolates. This runs even with `--epa-no-fallback`.

Reset checks apply only to VIO brackets needed to interpolate at GT timestamps
where GT is locally slower. Exact matches and dense-GT regions do not trigger
reset checks. A checked VIO reset requires **both**:

- Timestamp gap > max(0.5 s, five local sampling periods).
- Translation > max(5 m, five local speeds × elapsed time), or rotation >
  max(45 degrees, five local angular speeds × elapsed time).

Normal timestamp spacing never triggers this detector. A long gap alone does not
imply a reset; regular sparse keyframes and motion consistent with elapsed time
remain eligible. These conservative thresholds are heuristics: a reset with
normal timestamps or plausible endpoint motion will not be detected. Existing
motion-relative SR tests still apply independently.

Reset intervals use the time-offset-corrected clock and are recorded under
`input_coverage.reset_intervals`. Their interiors are excluded from association;
calibration and RPE pairs cannot cross them. They earn no SR credit. Successful regions on either side can join the same
group when the unsuccessful gap is strictly shorter than 10 seconds; the gap
itself remains excluded from SR and drift-valid metrics.
The full GT SR denominator is unchanged. Safe dense fallback uses the same source
support and reset guards; the benchmark skill continues to disable fallback.
Time-offset estimation itself is unchanged.

SR checks every pose using overlapping pairs with the available future endpoint
closest to one second. Checks shorten near a supported fragment's end; the final
pose uses a backward check closest to one second. No check crosses an explicitly
rejected reset interval. A check validates only its owner pose, not its interior.
An interval succeeds only when both endpoint poses pass their own checks. The
longest-group rule then selects intervals, and only passing poses adjacent to
selected successful intervals enter drift-valid ATE. Drift-valid RPE cannot cross
an unsuccessful interval. Generic RPE reporting keeps its existing pairing policy.

Stable-window selection is enabled: estimate steps exceeding max(5 m, 30 times
GT displacement) split the trajectory into candidate windows. The longest eligible
window by associated pose count is considered, without preferring the initial
segment. Equal-length windows use the earliest window. Alignment selection can prefer this stable candidate over the full
trajectory, including its extrinsic translation fit. Full-trajectory robust
fitting remains available.
