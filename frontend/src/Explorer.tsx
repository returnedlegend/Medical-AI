import { useEffect, useId, useState } from "react";
import {
  ArrowUpRight,
  CheckCircle2,
  ChevronDown,
  LoaderCircle,
  MousePointer2,
  RotateCcw,
  TriangleAlert,
} from "lucide-react";
import { request } from "./api";
import type { Outcome, Patient, Strategy } from "./types";
import { Anatomy, fixed, pct } from "./visuals";
interface Thresholds {
  force_limit_n: number;
  strain_limit: number;
  exposure_threshold: number;
}
interface Preview {
  outcome: Outcome;
  thresholds: Thresholds;
}
interface Grid {
  patient_id: string;
  thresholds: Thresholds;
  strategies: Outcome[];
}
export default function Explorer({
  patient,
  reference,
  config,
  runId,
}: {
  patient: Patient;
  reference: Outcome;
  config: Record<string, unknown>;
  runId: string;
}) {
  const [open, setOpen] = useState(false);
  const [strategy, setStrategy] = useState<Strategy>(reference.strategy),
    [preview, setPreview] = useState<Preview | null>(null),
    [loading, setLoading] = useState(false),
    [error, setError] = useState(""),
    [grid, setGrid] = useState<Grid | null>(null);
  const id = useId().replace(/:/g, "");
  useEffect(() => {
    setStrategy(reference.strategy);
    setPreview(null);
    setError("");
  }, [reference, patient.id]);
  useEffect(() => {
    if (!open) return;
    const abort = new AbortController();
    setGrid(null);
    request<Grid>(
      `/runs/${encodeURIComponent(runId)}/patients/${encodeURIComponent(patient.id)}/strategies`,
      { signal: abort.signal },
    )
      .then(setGrid)
      .catch((e) => {
        if (!abort.signal.aborted) setError(e.message);
      });
    return () => abort.abort();
  }, [open, runId, patient.id]);
  useEffect(() => {
    if (!open) return;
    const abort = new AbortController();
    setLoading(true);
    setError("");
    const timer = window.setTimeout(() => {
      const limits = Object.fromEntries(
        ["force_limit_n", "strain_limit", "exposure_threshold"]
          .filter((k) => config[k] != null)
          .map((k) => [k, config[k]]),
      );
      request<Preview>("/preview", {
        method: "POST",
        signal: abort.signal,
        body: JSON.stringify({
          patient: {
            attachment_length_mm: patient.attachment_length_mm,
            adhesion_count: patient.adhesion_count,
            adhesion_stiffness_n_per_mm: patient.adhesion_stiffness_n_per_mm,
            distension: patient.distension,
          },
          strategy: {
            grasp_position: strategy.grasp_position,
            direction_deg: strategy.direction_deg,
            distance_mm: strategy.distance_mm,
          },
          threshold_profile: config.threshold_profile,
          ...limits,
        }),
      })
        .then((data) => {
          if (!abort.signal.aborted) {
            setPreview(data);
            setLoading(false);
          }
        })
        .catch((e) => {
          if (!abort.signal.aborted) {
            setError(e.message);
            setLoading(false);
          }
        });
    }, 180);
    return () => {
      window.clearTimeout(timer);
      abort.abort();
    };
  }, [open, patient, strategy, config]);
  const set = (key: keyof Strategy, value: number) =>
    setStrategy((old) => ({ ...old, id: "custom", [key]: value }));
  return (
    <div className="strategy-explorer">
      <button
        className="explorer-toggle"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
      >
        <span className="explorer-icon">
          <MousePointer2 size={15} />
        </span>
        <span>
          <strong>Try a custom retraction</strong>
          <small>
            Adjust the pull or select a strategy from the measured search grid.
          </small>
        </span>
        <span className="interactive-tag">INTERACTIVE</span>
        <ChevronDown size={16} className={open ? "rotate" : ""} />
      </button>
      {open && (
        <div className="explorer-body">
          <div className="explorer-note">
            <ArrowUpRight size={13} />
            Live analytic preview. Adjustments do not change the saved
            experiment.
          </div>
          <div className="explorer-layout">
            <div className="explorer-controls">
              <h3>Design a pull</h3>
              <Slider
                id={`${id}grasp`}
                label="Custom grasp position"
                value={strategy.grasp_position}
                min={0}
                max={1}
                step={0.01}
                display={`${fixed(strategy.grasp_position, 2)} along neck`}
                onChange={(v) => set("grasp_position", v)}
              />
              <Slider
                id={`${id}angle`}
                label="Custom pull direction"
                value={strategy.direction_deg}
                min={0}
                max={180}
                step={1}
                display={`${fixed(strategy.direction_deg, 0)}°`}
                onChange={(v) => set("direction_deg", v)}
              />
              <Slider
                id={`${id}distance`}
                label="Custom pull distance"
                value={strategy.distance_mm}
                min={0}
                max={30}
                step={0.5}
                display={`${fixed(strategy.distance_mm, 1)} mm`}
                onChange={(v) => set("distance_mm", v)}
              />
              <button
                className="reset-strategy"
                onClick={() => setStrategy(reference.strategy)}
              >
                <RotateCcw size={12} />
                Reset to selected strategy
              </button>
              <div className="preview-status" role="status">
                {loading ? (
                  <>
                    <LoaderCircle size={13} className="spin" />
                    Updating mechanics…
                  </>
                ) : preview ? (
                  <span
                    className={
                      preview.outcome.success ? "preview-pass" : "preview-fail"
                    }
                  >
                    {preview.outcome.success ? (
                      <CheckCircle2 size={13} />
                    ) : (
                      <TriangleAlert size={13} />
                    )}{" "}
                    {preview.outcome.success
                      ? "All proxy constraints pass"
                      : "Proxy constraints do not all pass"}
                  </span>
                ) : null}
              </div>
              {error && (
                <p className="explorer-error" role="alert">
                  {error}
                </p>
              )}
            </div>
            <div className={`preview-visual ${loading ? "updating" : ""}`}>
              <Anatomy
                compact
                patient={patient}
                outcome={{ ...(preview?.outcome ?? reference), strategy }}
              />
              <span>ILLUSTRATIVE SCHEMATIC · ANALYTIC METRICS BELOW</span>
            </div>
          </div>
          {preview && (
            <div className={`constraint-meters ${loading ? "updating" : ""}`}>
              <Constraint
                label="Peak force"
                value={preview.outcome.force_n}
                threshold={preview.thresholds.force_limit_n}
                display={`${fixed(preview.outcome.force_n)} N`}
                limit={`≤ ${fixed(preview.thresholds.force_limit_n, 1)} N`}
                pass={preview.outcome.force_ok ?? false}
              />
              <Constraint
                label="Tissue strain"
                value={preview.outcome.strain}
                threshold={preview.thresholds.strain_limit}
                display={pct(preview.outcome.strain, 1)}
                limit={`≤ ${pct(preview.thresholds.strain_limit)}`}
                pass={preview.outcome.strain_ok ?? false}
              />
              <Constraint
                label="Exposure proxy"
                value={preview.outcome.exposure}
                threshold={preview.thresholds.exposure_threshold}
                display={pct(preview.outcome.exposure, 1)}
                limit={`≥ ${pct(preview.thresholds.exposure_threshold)}`}
                pass={preview.outcome.exposure_ok ?? false}
              />
            </div>
          )}
          {grid && (
            <div className="grid-search-view">
              <div>
                <h3>The complete strategy grid</h3>
                <p>
                  {grid.strategies.length} evaluated strategies for this
                  patient. Select a point to preview its pull.
                </p>
              </div>
              <StrategyScatter
                outcomes={grid.strategies}
                thresholds={grid.thresholds}
                selected={strategy}
                onSelect={setStrategy}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
function Slider({
  id,
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="explorer-slider">
      <label htmlFor={id}>
        {label}
        <span>{display}</span>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div>
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
function Constraint({
  label,
  value,
  threshold,
  display,
  limit,
  pass,
}: {
  label: string;
  value: number;
  threshold: number;
  display: string;
  limit: string;
  pass: boolean;
}) {
  const max = Math.max(value, threshold) * 1.15;
  return (
    <div className={`constraint ${pass ? "pass" : "fail"}`}>
      <span>
        {label}
        <strong>
          {pass ? <CheckCircle2 size={12} /> : <TriangleAlert size={12} />}{" "}
          {pass ? "Pass" : "Fail"}
        </strong>
      </span>
      <div>
        <b>{display}</b>
        <small>{limit}</small>
      </div>
      <div className="constraint-track">
        <span style={{ width: `${(value / max) * 100}%` }} />
        <i style={{ left: `${(threshold / max) * 100}%` }} />
      </div>
    </div>
  );
}
function StrategyScatter({
  outcomes,
  thresholds,
  selected,
  onSelect,
}: {
  outcomes: Outcome[];
  thresholds: Thresholds;
  selected: Strategy;
  onSelect: (s: Strategy) => void;
}) {
  const [hover, setHover] = useState<Outcome | null>(null);
  const maxForce =
    Math.max(...outcomes.map((o) => o.force_n), thresholds.force_limit_n) *
    1.07;
  const x = (f: number) => 53 + (Math.log1p(f) / Math.log1p(maxForce)) * 530;
  const y = (e: number) => 228 - e * 181;
  const ticks = [0, 0.5, 1, 2, 5, 10, 20, 50, 100].filter((t) => t <= maxForce);
  return (
    <div className="scatter-container">
      <svg
        className="strategy-scatter chart"
        viewBox="0 0 620 284"
        role="group"
        aria-label="All strategy outcomes: force versus exposure. Each point selects a strategy."
      >
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1="53" x2="583" y1={y(t)} y2={y(t)} stroke="#e3e8df" />
            <text x="42" y={y(t) + 3} textAnchor="end">
              {pct(t)}
            </text>
          </g>
        ))}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1="47" y2="228" stroke="#edf0e8" />
            <text x={x(t)} y="247" textAnchor="middle">
              {t}
            </text>
          </g>
        ))}
        <rect
          x="53"
          y="47"
          width={x(thresholds.force_limit_n) - 53}
          height={y(thresholds.exposure_threshold) - 47}
          fill="#dcecd5"
          fillOpacity=".55"
        />
        <line
          x1={x(thresholds.force_limit_n)}
          x2={x(thresholds.force_limit_n)}
          y1="39"
          y2="228"
          stroke="#b69d77"
          strokeDasharray="4 4"
        />
        <line
          x1="53"
          x2="583"
          y1={y(thresholds.exposure_threshold)}
          y2={y(thresholds.exposure_threshold)}
          stroke="#9fb38f"
          strokeDasharray="4 4"
        />
        <text x="54" y="23">
          Exposure proxy
        </text>
        <text x="318" y="275" textAnchor="middle">
          Force (N, log(1 + force) scale)
        </text>
        {outcomes.map((o, i) => {
          const active =
            o.strategy.grasp_position === selected.grasp_position &&
            o.strategy.direction_deg === selected.direction_deg &&
            o.strategy.distance_mm === selected.distance_mm;
          return (
            <circle
              key={i}
              cx={x(o.force_n)}
              cy={y(o.exposure)}
              r={active ? 6 : 4}
              fill={o.success ? "#347b5b" : "#c0b399"}
              fillOpacity={active ? 1 : 0.7}
              stroke={active ? "#244f3c" : "#fffefa"}
              strokeWidth={active ? 2 : 1}
              role="button"
              tabIndex={0}
              aria-label={`${o.strategy.id}: force ${fixed(o.force_n)} N, exposure ${pct(o.exposure)}, ${o.success ? "success" : "failure"}`}
              onClick={() => onSelect(o.strategy)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelect(o.strategy);
                }
              }}
              onMouseEnter={() => setHover(o)}
              onMouseLeave={() => setHover(null)}
              onFocus={() => setHover(o)}
              onBlur={() => setHover(null)}
            >
              <title>{`${o.strategy.id} · ${fixed(o.force_n)} N · ${pct(o.exposure)} exposure · ${pct(o.strain)} strain`}</title>
            </circle>
          );
        })}
      </svg>
      <div className="scatter-key">
        <span>
          <i />
          All three constraints pass
        </span>
        <span>
          <i className="scatter-fail" />
          One or more constraints fail
        </span>
        <span>
          <i className="scatter-threshold" />
          Thresholds
        </span>
      </div>
      <p className="scatter-hover" aria-live="polite">
        {hover
          ? `${hover.strategy.id} · grasp ${fixed(hover.strategy.grasp_position, 2)} · ${hover.strategy.direction_deg}° · ${hover.strategy.distance_mm} mm · strain ${pct(hover.strain, 1)}`
          : "Hover or focus a point for its strategy details. The shaded region marks the force and exposure limits; strain is checked separately."}
      </p>
    </div>
  );
}
