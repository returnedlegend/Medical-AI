"""Human-readable, explicitly synthetic experiment report."""
from __future__ import annotations


def percent(value: float | None) -> str:
    return 'undefined' if value is None else f'{100 * value:.1f}%'


def metric(value: float | None) -> str:
    return 'undefined' if value is None else f'{value:.3f}'


def render_report(result: dict, run_id: str | None = None) -> str:
    summary, probe, cfg = result['summary'], result['probe'], result['config']
    limits = result['thresholds']
    ci = summary['personalization_gain_ci']
    strategy = summary['best_shared_strategy']
    lines = [
        '# Retraction Lab · Exploratory experiment report', '',
        '**Analytic surrogate only. These synthetic results do not validate tissue mechanics or clinical safety.**', '',
        f'Run: `{run_id or "headless"}`  ', f'Model: `{result["model_version"]}`  ',
        f'Seed: {cfg["seed"]} · Patients: {summary["n_patients"]} · '
        f'Training: {summary["train_size"]} · Held out: {summary["test_size"]} · Strategies: {summary["strategy_count"]}', '',
        '## Primary comparison', '',
        f'On held-out synthetic patients, the training-selected shared strategy succeeded in '
        f'{percent(summary["shared_success_rate"])} and per-patient grid search in '
        f'{percent(summary["oracle_success_rate"])}. The paired difference was '
        f'{100 * summary["personalization_gain"]:.1f} percentage points '
        f'(95% patient-bootstrap interval {100 * ci[0]:.1f}–{100 * ci[1]:.1f} pp).', '',
        f'No tested strategy succeeded in {percent(summary["bailout_rate"])} of held-out cases. '
        'This describes this finite strategy grid, not all possible interventions.', '',
        '| Comparison | Held-out success |', '|---|---:|',
        f'| Fixed reference | {percent(summary["standard_success_rate"])} |',
        f'| Training-selected shared | {percent(summary["shared_success_rate"])} |',
        f'| Per-patient grid optimum | {percent(summary["oracle_success_rate"])} |', '',
        f'Shared strategy: grasp {strategy["grasp_position"]:.2f}, direction '
        f'{strategy["direction_deg"]}°, distance {strategy["distance_mm"]} mm.', '',
        '## Early-warning model', '',
        'A logistic regression uses only the fixed probe force-response features. '
        'Its target is shared-strategy failure with per-patient grid success.', '',
        f'- Held-out personalization AUROC: {metric(probe["auroc"])}',
        f'- Brier score: {metric(probe["brier_score"])}',
        f'- Ten-bin expected calibration error: {metric(probe["ece"])}',
        f'- Separate grid-bail-out AUROC: {metric(probe["bailout"]["auroc"])}',
        f'- Held-out probe-limit violation rate: {percent(summary.get("probe_violation_rate"))}', '',
        'Undefined metrics indicate insufficient outcome classes; they are not zero scores. '
        'Probe observations are noise-free and do not implement a force-triggered stop controller.', '',
        '## Patient strata', '',
        'Fixed illustrative cutoffs; held-out patients only. Dimensions overlap. '
        'These descriptive comparisons are exploratory and not adjusted for multiple comparisons.', '',
        '| Dimension | Stratum | n | Shared | Grid optimum | Gain (pp) | Bail-out |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for group in result.get('subgroups', []):
        gain = 'undefined' if group['personalization_gain'] is None else f'{100 * group["personalization_gain"]:.1f}'
        lines.append(f'| {group["dimension"]} | {group["label"]} | {group["n"]} | '
                     f'{percent(group["shared_success_rate"])} | {percent(group["oracle_success_rate"])} | '
                     f'{gain} | {percent(group["bailout_rate"])} |')
    lines += ['', '## Active assumptions', '',
              f'Profile: **{cfg["threshold_profile"]}**; force ≤ {limits["force_limit_n"]} N; '
              f'strain ≤ {limits["strain_limit"]}; exposure ≥ {limits["exposure_threshold"]}.', '',
              'The surrogate explicitly derives preferred pull direction from synthetic patient parameters, '
              'so heterogeneous strategy benefit is partly imposed. Bootstrap intervals quantify patient '
              'sampling conditional on the chosen model and shared strategy, not physical-model uncertainty.', '',
              '## Limitations', '']
    for limitation in result['methodology'].get('limitations', []):
        lines.append(f'- {limitation}')
    lines += ['', '## Reproducible configuration', '', '```json']
    import json
    lines += [json.dumps(cfg, indent=2, sort_keys=True, allow_nan=False), '```', '',
              'Export the full JSON alongside this report to preserve patient-level data, force curves, '
              'calibration bins, and methodology. This run is exploratory, not externally preregistered.', '']
    return '\n'.join(lines)
