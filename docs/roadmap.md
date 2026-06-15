# EPA Follow-Up Roadmap

## Alignment / Success-Rate Evaluation

- Revisit robust Step3 alignment selection. The 2026-06-12 full run showed no new SR=0 regressions outside the known 44-case union, but the 44-case union still has mixed movement: some cases improve, some Lamaria cases drop slightly, and 18 remain SR=0. Keep the current implementation for now; next pass should inspect `outputs/alignanything2_full_robust/run_20260612_203300/summary.html` and `analysis/robust_vs_p0_summary.md` before changing thresholds.
- Investigate the remaining SR=0 cases after robust Step3. Prioritize EuRoC `V2_02_medium/svo_mono`, Grand Tour `svo_mono` failures, Harbor `harbor_sequence_1/svo_stereo`, and Lamaria hard/add/cp `svo_mono` cases.
- Classify each persistent failure as time-alignment error, Step3 outlier dominance, trajectory jump, late drift, scale/unit issue, or GT mapping issue.
- Add orientation instability reporting before changing SR semantics. Aqualoc/Rovio has cases where translation SR can be high while orientation error is extreme; report this as a warning first rather than folding rotation into SR.
- Resolve LaMARia hard GT mapping for `R_09_hard` / `R_10_hard`; current data discovery still reports unresolved cases.

## Reports / Visualization

- Done: default reports keep the normal figure gallery but replace the raw + Step2 + Step3 triptych with the Step3 alignment map; `--debug` also includes the triptych.
- Add clipped or zoomed histogram views, e.g. p95/p99 range, alongside full histograms.
- Continue improving `summary.html`: add filters for `case_status`, SR range, dataset, method, and Step3 mode; keep Markdown/CSV outputs.
- Consider a Plotly 3D trajectory view for drag/rotate/top-down inspection while keeping static PNGs for quick browsing.
