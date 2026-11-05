import { useEffect, useState } from "react";
import {
  ArrowDownToLine,
  ArrowRight,
  Box,
  ChevronDown,
  Clock3,
  ExternalLink,
  FlaskConical,
  LoaderCircle,
  Play,
  RefreshCw,
  TriangleAlert,
} from "lucide-react";
import { request, type RunStatus } from "./api";
import { fixed } from "./visuals";
interface PhysicsConfig {
  seed: number;
  steps: number;
  start_grasped: boolean;
  pan: number;
  tilt: number;
  spin: number;
  insertion: number;
  jaw: number;
  damage_model: boolean;
  cautery_power: number;
}
interface PhysicsRun {
  id: string;
  created_at: string;
  status: RunStatus;
  progress: number;
  config: PhysicsConfig;
  error: string | null;
  summary: Record<string, unknown> | null;
}
interface PhysicsStatus {
  available: boolean;
  model_ready: false;
  purpose: string;
  reason?: string;
  upstream_commit?: string;
}
interface Step {
  step: number;
  simulation_time_s: number;
  grasp_established: boolean;
  target_visible: boolean;
  internal_force_sum_raw: number;
  mean_vertex_displacement_scene_units: number;
  max_vertex_displacement_scene_units: number;
  gripper_pose: number[];
  gripper_ptsda_state: number[];
  element_injury?: {
    max_damage: number;
    damaged_element_count: number;
    max_peak_tensile_strain: number;
    minimum_young_modulus_scene_units: number;
    max_temperature_c: number;
    mean_volume_weighted_damage: number;
  } | null;
}
interface PhysicsResult {
  model_ready: false;
  topology_mode?: string;
  topology?: Record<
    string,
    {
      component_state: string;
      vertex_count: number;
      tetrahedra: number[][];
      triangles: number[][];
      edges: number[][];
    }
  >;
  diagnostic_state: string;
  diagnostics: {
    error_count: number;
    warning_count: number;
    topology_error_count: number;
    messages: { type: string; sender: string; message: string }[];
  };
  steps: Step[];
  gallbladder_vertex_count: number;
  sofa_version: string;
  sofa_env_commit: string;
  elapsed_seconds: number;
  requested_steps: number;
  limitations: string[];
  start_grasped: boolean;
  damage_model?: string | null;
}
const defaultConfig: PhysicsConfig = {
  seed: 42,
  steps: 10,
  start_grasped: false,
  pan: 0,
  tilt: 0,
  spin: 0,
  insertion: 0,
  jaw: 0,
  damage_model: false,
  cautery_power: 0,
};
const when = (value: string) =>
  new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
const message = (e: unknown) =>
  e instanceof Error ? e.message : "The diagnostic request failed.";
const number = (value: number | undefined) =>
  value == null
    ? "—"
    : Math.abs(value) > 10000 ||
        (Math.abs(value) > 0 && Math.abs(value) < 0.001)
      ? value.toExponential(3)
      : fixed(value, 3);
export default function Physics() {
  const [status, setStatus] = useState<PhysicsStatus | null>(null),
    [runs, setRuns] = useState<PhysicsRun[]>([]),
    [selected, setSelected] = useState<PhysicsRun | null>(null),
    [result, setResult] = useState<PhysicsResult | null>(null);
  const [config, setConfig] = useState<PhysicsConfig>(defaultConfig),
    [error, setError] = useState(""),
    [historyError, setHistoryError] = useState(""),
    [starting, setStarting] = useState(false),
    [loading, setLoading] = useState(false),
    [refresh, setRefresh] = useState(0),
    [retry, setRetry] = useState(0),
    [frame, setFrame] = useState<"initial" | "final">("final");
  const selectedId = selected?.id;
  const hasActiveHistory = runs.some(
    (run) => run.status === "queued" || run.status === "running",
  );
  const busy =
    starting ||
    hasActiveHistory ||
    selected?.status === "queued" ||
    selected?.status === "running";
  useEffect(() => {
    const abort = new AbortController();
    setError("");
    Promise.all([
      request<PhysicsStatus>("/physics/status", { signal: abort.signal }),
      request<{ runs: PhysicsRun[] }>("/physics/runs", {
        signal: abort.signal,
      }),
    ])
      .then(([state, history]) => {
        if (abort.signal.aborted) return;
        setStatus(state);
        setRuns(history.runs);
        setHistoryError("");
        if (!selectedId && history.runs[0]) setSelected(history.runs[0]);
      })
      .catch((e) => {
        if (!abort.signal.aborted) setError(`Physics service: ${message(e)}`);
      });
    return () => abort.abort();
  }, [refresh]);
  useEffect(() => {
    if (!hasActiveHistory) {
      setHistoryError("");
      return;
    }
    const abort = new AbortController();
    let timer: number | undefined;
    let retryDelay = 1000;
    const pollHistory = async () => {
      try {
        const history = await request<{ runs: PhysicsRun[] }>("/physics/runs", {
          signal: abort.signal,
        });
        if (abort.signal.aborted) return;
        setRuns(history.runs);
        setHistoryError("");
        retryDelay = 1000;
        if (
          history.runs.some(
            (run) => run.status === "queued" || run.status === "running",
          )
        ) {
          timer = window.setTimeout(() => void pollHistory(), 1000);
        }
      } catch (e) {
        if (!abort.signal.aborted) {
          setHistoryError(`Physics history: ${message(e)}`);
          timer = window.setTimeout(() => void pollHistory(), retryDelay);
          retryDelay = Math.min(retryDelay * 2, 10000);
        }
      }
    };
    timer = window.setTimeout(() => void pollHistory(), 1000);
    return () => {
      abort.abort();
      window.clearTimeout(timer);
    };
  }, [hasActiveHistory]);
  useEffect(() => {
    if (!selectedId) return;
    const abort = new AbortController();
    let timer: number | undefined;
    let retryDelay = 1000;
    setResult(null);
    setLoading(true);
    setError("");
    const poll = async () => {
      setLoading(true);
      try {
        const run = await request<PhysicsRun>(`/physics/runs/${selectedId}`, {
          signal: abort.signal,
        });
        if (abort.signal.aborted) return;
        setSelected(run);
        setRuns((old) =>
          old.some((r) => r.id === run.id)
            ? old.map((r) => (r.id === run.id ? run : r))
            : [run, ...old],
        );
        if (run.status === "completed") {
          const data = await request<PhysicsResult>(
            `/physics/runs/${selectedId}/results`,
            { signal: abort.signal },
          );
          if (!abort.signal.aborted) {
            setResult(data);
            setLoading(false);
            setError("");
          }
        } else if (run.status === "failed") {
          setLoading(false);
          setError(
            run.error ||
              "The physics subprocess failed. Inspect the server log before retrying.",
          );
        } else {
          setError("");
          retryDelay = 1000;
          timer = window.setTimeout(() => void poll(), 1000);
        }
      } catch (e) {
        if (!abort.signal.aborted) {
          setLoading(false);
          setError(message(e));
          timer = window.setTimeout(() => void poll(), retryDelay);
          retryDelay = Math.min(retryDelay * 2, 10000);
        }
      }
    };
    void poll();
    return () => {
      abort.abort();
      window.clearTimeout(timer);
    };
  }, [selectedId, retry]);
  const launch = async () => {
    setStarting(true);
    setError("");
    try {
      const run = await request<PhysicsRun>("/physics/runs", {
        method: "POST",
        body: JSON.stringify(config),
      });
      setRuns((old) => [run, ...old]);
      setSelected(run);
    } catch (e) {
      setError(message(e));
    } finally {
      setStarting(false);
    }
  };
  const change = (key: keyof PhysicsConfig, value: number | boolean) =>
    setConfig((old) => ({ ...old, [key]: value }));
  const final = result?.steps[result.steps.length - 1];
  return (
    <div className="physics-page">
      <div className="physics-intro panel">
        <span className="physics-intro-icon">
          <Box size={28} />
        </span>
        <div>
          <span className="eyebrow">OPTIONAL UPSTREAM PHYSICS DIAGNOSTIC</span>
          <h2>SOFA gallbladder scene</h2>
          <p>
            Run the upstream LapGym gallbladder scene in its own process.
            Inspect rendered frames, gripper motion, and raw mesh telemetry.
            This diagnostic is separate from the analytic cohort study.
          </p>
        </div>
        <span
          className={`physics-runtime ${status?.available ? "available" : ""}`}
        >
          <span className="status-dot" />
          {status === null
            ? "Checking runtime"
            : status.available
              ? "Runtime installed"
              : "Runtime unavailable"}
        </span>
      </div>
      <div className="physics-warning">
        <TriangleAlert size={19} />
        <div>
          <strong>Tissue-model validation remains incomplete.</strong>
          <p>
            No patient-specific parameterization, CVS exposure metric, threshold
            enforcement, or strategy search is implemented in this scene.
            Rendered frames and finite telemetry do not establish physical
            fidelity.
          </p>
        </div>
      </div>
      {(error || historyError) && (
        <div className="error-banner" role="alert">
          <TriangleAlert size={17} />
          <span>{error || historyError}</span>
          <button
            className="physics-retry"
            onClick={() => {
              setRefresh((v) => v + 1);
              setRetry((v) => v + 1);
            }}
          >
            <RefreshCw size={13} />
            Retry
          </button>
        </div>
      )}
      {status && !status.available ? (
        <div className="physics-unavailable panel">
          <Box size={38} />
          <h3>The optional runtime is not installed.</h3>
          <p>
            {status.reason ||
              "The analytic study remains available. Install the pinned runtime to enable this independent physics diagnostic."}
          </p>
          <code>bash scripts/setup-sofa.sh</code>
          <span>
            The standalone Windows application includes the portable procedure
            and cohort engines. This optional scene needs a separate runtime;
            run this setup command from a repository checkout on a supported
            host, then check again.
          </span>
          <button
            className="secondary-button"
            onClick={() => setRefresh((v) => v + 1)}
          >
            <RefreshCw size={14} />
            Check runtime
          </button>
        </div>
      ) : (
        <div className="physics-layout">
          <aside>
            <div className="panel physics-config">
              <div className="panel-heading">
                <h2>Diagnostic controls</h2>
                <span className="step-badge">SOFA</span>
              </div>
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  void launch();
                }}
              >
                <div className="physics-preset">
                  <span className="eyebrow">COMMAND PRESET</span>
                  <div className="segmented">
                    <button
                      type="button"
                      onClick={() =>
                        setConfig((old) => ({
                          ...old,
                          pan: 0,
                          tilt: 0,
                          spin: 0,
                          insertion: 0,
                          jaw: 0,
                        }))
                      }
                    >
                      Neutral
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        setConfig((old) => ({
                          ...old,
                          pan: 0,
                          tilt: 0.1,
                          spin: 0,
                          insertion: 0.1,
                          jaw: 0,
                        }))
                      }
                    >
                      Gentle command
                    </button>
                  </div>
                  <p>
                    Presets set actuator commands. They do not imply successful
                    grasping or exposure.
                  </p>
                </div>
                <div className="physics-number-fields">
                  <label htmlFor="physics-seed">
                    Physics seed
                    <input
                      id="physics-seed"
                      type="number"
                      min="0"
                      max="4294967295"
                      value={config.seed}
                      required
                      onChange={(e) => change("seed", Number(e.target.value))}
                    />
                  </label>
                  <label htmlFor="physics-steps">
                    Simulation steps
                    <input
                      id="physics-steps"
                      type="number"
                      min="1"
                      max="50"
                      value={config.steps}
                      required
                      onChange={(e) => change("steps", Number(e.target.value))}
                    />
                  </label>
                </div>
                <label className="physics-checkbox">
                  <input
                    type="checkbox"
                    checked={config.start_grasped}
                    onChange={(e) => change("start_grasped", e.target.checked)}
                  />
                  <span>
                    Request a scripted initial grasp
                    <small>
                      Uses upstream’s bounded grasp routine. Check the recorded
                      grasp state.
                    </small>
                  </span>
                </label>
                <div className="physics-actuators">
                  <span className="eyebrow">NORMALIZED GRIPPER VELOCITIES</span>
                  {(["pan", "tilt", "insertion"] as const).map((key) => (
                    <div className="explorer-slider" key={key}>
                      <label htmlFor={`physics-${key}`}>
                        Physics {key}
                        <span>{fixed(config[key], 2)}</span>
                      </label>
                      <input
                        id={`physics-${key}`}
                        type="range"
                        min="-.5"
                        max=".5"
                        step=".01"
                        value={config[key]}
                        onChange={(e) => change(key, Number(e.target.value))}
                      />
                      <div>
                        <span>−0.5</span>
                        <span>+0.5</span>
                      </div>
                    </div>
                  ))}
                  <p>
                    Spin and jaw commands remain neutral. These are upstream
                    normalized velocities, not position or force targets.
                  </p>
                </div>
                <label className="physics-checkbox">
                  <input
                    type="checkbox"
                    checked={config.damage_model}
                    onChange={(e) =>
                      setConfig((old) => ({
                        ...old,
                        damage_model: e.target.checked,
                        cautery_power: e.target.checked ? old.cautery_power : 0,
                      }))
                    }
                  />
                  <span>
                    Experimental material softening
                    <small>
                      Distinct FEM variant with persistent per-element injury
                      and stiffness updates. Separate from the portable RL
                      environment.
                    </small>
                  </span>
                </label>
                {config.damage_model && (
                  <div className="explorer-slider physics-thermal-control">
                    <label htmlFor="physics-cautery-power">
                      Experimental contact heat{" "}
                      <span>{fixed(config.cautery_power, 2)}</span>
                    </label>
                    <input
                      id="physics-cautery-power"
                      type="range"
                      min="0"
                      max="1"
                      step=".05"
                      value={config.cautery_power}
                      onChange={(e) =>
                        change("cautery_power", Number(e.target.value))
                      }
                    />
                    <p>
                      Heat is applied near the actual cauter tip. No contact
                      means no burn; this model does not cut tissue.
                    </p>
                  </div>
                )}
                <button
                  className="primary-button launch"
                  disabled={busy || !status?.available}
                  type="submit"
                >
                  {busy ? (
                    <LoaderCircle size={15} className="spin" />
                  ) : (
                    <Play size={14} fill="currentColor" />
                  )}
                  {busy ? "Diagnostic in progress…" : "Run physics diagnostic"}
                  {!busy && <ArrowRight size={15} />}
                </button>
                <p className="launch-note">
                  One isolated scene per subprocess. Bounded run length.
                </p>
              </form>
            </div>
            <div className="panel physics-history">
              <div className="panel-heading">
                <h2>
                  <Clock3 size={15} />
                  Physics history
                </h2>
                <div className="physics-history-tools">
                  <span className="step-badge">{runs.length}</span>
                  <button
                    className="icon-button"
                    type="button"
                    aria-label="Refresh physics history"
                    onClick={() => setRefresh((v) => v + 1)}
                  >
                    <RefreshCw size={13} />
                  </button>
                </div>
              </div>
              {runs.length ? (
                runs.map((run) => (
                  <button
                    key={run.id}
                    className={selected?.id === run.id ? "selected" : ""}
                    onClick={() => {
                      if (selected?.id === run.id) setRetry((v) => v + 1);
                      else setSelected(run);
                    }}
                  >
                    <span>
                      <strong>{run.id.slice(0, 8)}</strong>
                      <small>
                        {when(run.created_at)} · {run.config.steps} steps
                      </small>
                    </span>
                    <span className={`run-status ${run.status}`}>
                      {run.status}
                    </span>
                  </button>
                ))
              ) : (
                <p>No physics diagnostics saved yet.</p>
              )}
            </div>
          </aside>
          <section className="physics-results" aria-busy={loading || starting}>
            <div className="results-header">
              <div>
                <span className="eyebrow">UPSTREAM SCENE OBSERVATION</span>
                <h2>
                  {selected
                    ? `Diagnostic ${selected.id.slice(0, 8)}`
                    : "Physics diagnostic"}
                </h2>
                {selected && (
                  <p>
                    {when(selected.created_at)}
                    <span>·</span>Seed {selected.config.seed}
                    <span>·</span>
                    {selected.status}
                  </p>
                )}
              </div>
              {result && selected && (
                <a
                  className="small-button"
                  href={`/api/physics/runs/${selected.id}/results`}
                  download={`physics-${selected.id}.json`}
                >
                  <ArrowDownToLine size={13} />
                  JSON
                </a>
              )}
            </div>
            {loading || starting ? (
              <div className="panel running-state" role="status">
                <div className="loading-orbit">
                  <Box size={29} />
                  <span />
                </div>
                <h3>
                  {selected?.status === "completed"
                    ? "Loading scene diagnostics"
                    : "Running the upstream scene"}
                </h3>
                <p>
                  Starting SOFA, initializing the mesh, and stepping a bounded
                  gripper command. Initialization can take longer than the
                  requested simulation interval.
                </p>
                <div className="progress-track">
                  <span
                    style={{
                      width: `${Math.max(5, (selected?.progress ?? 0) * 100)}%`,
                    }}
                  />
                </div>
                <span className="eyebrow">
                  PHYSICS DIAGNOSTIC · NO STUDY OUTCOME
                </span>
              </div>
            ) : !result ? (
              <div className="panel empty-state">
                <span className="empty-icon">
                  <Box size={31} />
                </span>
                <span className="eyebrow">ACTUAL UPSTREAM RENDERING</span>
                <h3>No diagnostic selected</h3>
                <p>
                  Run a short diagnostic to inspect the gallbladder mesh and
                  instrument state. This view generates no cohort-study success
                  rate.
                </p>
                <div className="physics-empty-note">
                  <FlaskConical size={14} />
                  The analytic study is available in Experiments.
                </div>
              </div>
            ) : (
              <>
                <div className="panel physics-frame-panel">
                  <div className="panel-heading">
                    <div>
                      <h2>Rendered scene</h2>
                      <p>
                        {frame === "initial"
                          ? "Actual upstream frame after reset, before commanded steps."
                          : `Actual upstream frame after ${result.steps.length} recorded steps.`}
                      </p>
                    </div>
                    <div className="physics-frame-controls">
                      <div
                        role="group"
                        className="segmented"
                        aria-label="Physics image frame"
                      >
                        <button
                          className={frame === "initial" ? "selected" : ""}
                          aria-pressed={frame === "initial"}
                          onClick={() => setFrame("initial")}
                        >
                          Initial
                        </button>
                        <button
                          className={frame === "final" ? "selected" : ""}
                          aria-pressed={frame === "final"}
                          onClick={() => setFrame("final")}
                        >
                          Final
                        </button>
                      </div>
                      <span className="physics-engine-tag">
                        SOFA {result.sofa_version}
                      </span>
                    </div>
                  </div>
                  <div className="physics-frame">
                    <img
                      src={`/api/physics/runs/${selectedId}/image?frame=${frame}`}
                      alt="Actual SOFA-rendered upstream gallbladder and surgical instruments at the selected diagnostic frame"
                    />
                    <div className="physics-frame-overlay">
                      <span>UPSTREAM LAPGYM SCENE</span>
                      <span>NO CVS MEASUREMENT</span>
                    </div>
                  </div>
                  <div className="physics-topology-mode">
                    <span>TOPOLOGY MODE</span>
                    <strong>
                      {result.topology_mode === "derived"
                        ? "Tet-derived connectivity"
                        : result.topology_mode === "upstream"
                          ? "Upstream baseline"
                          : "Not recorded"}
                    </strong>
                    <p>
                      {result.topology_mode === "derived"
                        ? "A runtime wrapper derives connectivity from the upstream tetrahedra. Source assets are unchanged; physical fidelity is not validated."
                        : "Inspect runtime diagnostics before interpreting forces or deformation."}
                    </p>
                  </div>
                  <div className="physics-frame-footer">
                    <span>
                      {result.gallbladder_vertex_count.toLocaleString()} mesh
                      vertices
                    </span>
                    <span>{fixed(result.elapsed_seconds, 1)}s wall time</span>
                    <a
                      href={`/api/physics/runs/${selectedId}/image?frame=${frame}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Open frame
                      <ExternalLink size={11} />
                    </a>
                  </div>
                </div>
                <div className="physics-metrics">
                  <PhysicsMetric
                    label="Actual grasp at final step"
                    value={
                      final?.grasp_established
                        ? "Established"
                        : "Not established"
                    }
                    note={
                      result.start_grasped
                        ? "Initial grasp requested"
                        : "Initial grasp not requested"
                    }
                  />
                  <PhysicsMetric
                    label="Max mesh displacement"
                    value={number(final?.max_vertex_displacement_scene_units)}
                    note="Raw scene coordinates · units unverified"
                  />
                  <PhysicsMetric
                    label="Internal-force sum"
                    value={number(final?.internal_force_sum_raw)}
                    note="Raw diagnostic · NOT newtons"
                  />
                </div>
                <div className="panel physics-diagnostics">
                  {result.damage_model && (
                    <div className="physics-material-result">
                      <h2>Experimental FEM material response</h2>
                      <p>
                        {result.damage_model} · damage changes element modulus
                        for subsequent SOFA steps. Maximum injury is local;
                        volume-weighted injury describes the whole mesh.
                      </p>
                      {final?.element_injury ? (
                        <dl>
                          <div>
                            <dt>Maximum element damage</dt>
                            <dd>{number(final.element_injury.max_damage)}</dd>
                          </div>
                          <div>
                            <dt>Volume-weighted mean damage</dt>
                            <dd>
                              {number(
                                final.element_injury
                                  .mean_volume_weighted_damage,
                              )}
                            </dd>
                          </div>
                          <div>
                            <dt>Damaged elements</dt>
                            <dd>
                              {final.element_injury.damaged_element_count}
                            </dd>
                          </div>
                          <div>
                            <dt>Peak tensile strain</dt>
                            <dd>
                              {number(
                                final.element_injury.max_peak_tensile_strain,
                              )}
                            </dd>
                          </div>
                          <div>
                            <dt>Minimum requested modulus</dt>
                            <dd>
                              {number(
                                final.element_injury
                                  .minimum_young_modulus_scene_units,
                              )}{" "}
                              raw scene units
                            </dd>
                          </div>
                          <div>
                            <dt>Maximum temperature</dt>
                            <dd>
                              {number(final.element_injury.max_temperature_c)}{" "}
                              °C*
                            </dd>
                          </div>
                        </dl>
                      ) : (
                        <p>No per-element injury telemetry was recorded.</p>
                      )}
                      <p>
                        * Illustrative heat and damage laws. Material softening
                        does not represent cutting, rupture, vascular flow, or
                        validated tissue response.
                      </p>
                    </div>
                  )}
                  <div className="panel-heading">
                    <div>
                      <h2>
                        <TriangleAlert size={16} />
                        Runtime diagnostics
                      </h2>
                      <p>
                        Messages are evidence to inspect, not a validated
                        physical model.
                      </p>
                    </div>
                    <span
                      className={`physics-diagnostic-state ${result.diagnostic_state === "needs_review" ? "needs-review" : ""}`}
                    >
                      {result.diagnostic_state === "needs_review"
                        ? "Needs review"
                        : "Infrastructure checked"}
                    </span>
                  </div>
                  <div className="physics-diagnostic-counts">
                    <span>
                      <strong>{result.diagnostics.error_count}</strong>SOFA
                      errors
                    </span>
                    <span>
                      <strong>{result.diagnostics.warning_count}</strong>
                      Warnings
                    </span>
                    <span>
                      <strong>{result.diagnostics.topology_error_count}</strong>
                      Topology errors
                    </span>
                  </div>
                  {result.diagnostics.topology_error_count > 0 && (
                    <p className="physics-topology-alert">
                      Topology errors were reported by the upstream scene. Do
                      not interpret forces or deformations as validated tissue
                      mechanics.
                    </p>
                  )}
                  {result.topology && (
                    <div
                      className="physics-topology-table table-scroll"
                      tabIndex={0}
                      role="region"
                      aria-label="Physics topology table"
                    >
                      <table>
                        <thead>
                          <tr>
                            <th>Tissue</th>
                            <th>Component state</th>
                            <th>Vertices</th>
                            <th>Tetrahedra</th>
                            <th>Triangles</th>
                            <th>Edges</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(result.topology).map(
                            ([name, mesh]) => (
                              <tr key={name}>
                                <td>{name}</td>
                                <td>{mesh.component_state}</td>
                                <td>{mesh.vertex_count.toLocaleString()}</td>
                                <td>
                                  {mesh.tetrahedra.length.toLocaleString()}
                                </td>
                                <td>
                                  {mesh.triangles.length.toLocaleString()}
                                </td>
                                <td>{mesh.edges.length.toLocaleString()}</td>
                              </tr>
                            ),
                          )}
                        </tbody>
                      </table>
                      <p>
                        Component states are reported verbatim. “Undefined” or
                        absence of an “Invalid” state does not establish
                        topology or physics validation.
                      </p>
                    </div>
                  )}
                  <details className="physics-messages">
                    <summary>
                      Inspect captured diagnostic messages
                      <ChevronDown size={13} />
                    </summary>
                    {result.diagnostics.messages.length ? (
                      result.diagnostics.messages.map((item, i) => (
                        <div key={i}>
                          <span>
                            {item.type} · {item.sender}
                          </span>
                          <p>{item.message}</p>
                        </div>
                      ))
                    ) : (
                      <p>No captured error or warning messages.</p>
                    )}
                  </details>
                </div>
                <div className="panel physics-telemetry">
                  <div className="panel-heading">
                    <div>
                      <h2>Step telemetry</h2>
                      <p>Uncalibrated diagnostics from this upstream scene.</p>
                    </div>
                    <span className="heldout-tag">
                      {result.steps.length} RECORDED STEPS
                    </span>
                  </div>
                  <div className="physics-telemetry-charts">
                    <Telemetry
                      steps={result.steps}
                      field="internal_force_sum_raw"
                      title="Internal-force sum"
                      unit="Raw internal diagnostic · not N"
                    />
                    <Telemetry
                      steps={result.steps}
                      field="max_vertex_displacement_scene_units"
                      title="Max vertex displacement"
                      unit="Raw scene coordinates · units unverified"
                    />
                  </div>
                  <div
                    className="table-scroll physics-step-table"
                    tabIndex={0}
                    role="region"
                    aria-label="Physics step telemetry table"
                  >
                    <table>
                      <thead>
                        <tr>
                          <th>Step</th>
                          <th>Sim. time (s)</th>
                          <th>Raw force sum</th>
                          <th>Max displacement</th>
                          <th>Grasp</th>
                          <th>Upstream target</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.steps.map((step) => (
                          <tr key={step.step}>
                            <td>{step.step}</td>
                            <td>{fixed(step.simulation_time_s, 3)}</td>
                            <td>{number(step.internal_force_sum_raw)}</td>
                            <td>
                              {number(step.max_vertex_displacement_scene_units)}
                            </td>
                            <td>
                              {step.grasp_established ? "Established" : "None"}
                            </td>
                            <td>
                              {step.target_visible ? "Visible" : "Hidden"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="panel-footnote">
                    <TriangleAlert size={13} />
                    “Target” is an upstream target sphere. Its visibility does
                    not measure the hepatocystic triangle or CVS.
                  </div>
                </div>
                {final && (
                  <div className="panel physics-pose">
                    <div className="panel-heading">
                      <div>
                        <h2>Final gripper state</h2>
                        <p>
                          Pose position is in raw scene coordinates; rotation
                          uses quaternion components.
                        </p>
                      </div>
                    </div>
                    <div className="physics-pose-values">
                      {final.gripper_pose.map((value, index) => (
                        <span key={index}>
                          {["x", "y", "z", "qx", "qy", "qz", "qw"][index] ??
                            `pose ${index}`}
                          <strong>{number(value)}</strong>
                        </span>
                      ))}
                    </div>
                    <details>
                      <summary>
                        Pan / tilt / spin / depth / jaw state
                        <ChevronDown size={13} />
                      </summary>
                      <div className="physics-pose-values">
                        {final.gripper_ptsda_state.map((value, index) => (
                          <span key={index}>
                            {["pan", "tilt", "spin", "depth", "jaw"][index] ??
                              `state ${index}`}
                            <strong>{number(value)}</strong>
                          </span>
                        ))}
                      </div>
                      <p>
                        Upstream state values. Angular and length conventions
                        have not been independently validated.
                      </p>
                    </details>
                  </div>
                )}
                <div className="panel physics-limitations">
                  <h3>Interpretation boundaries</h3>
                  <ul>
                    {result.limitations.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                  <span>
                    Upstream commit <code>{result.sofa_env_commit}</code>
                  </span>
                  <strong>
                    Model ready: false · Computational diagnostic only
                  </strong>
                </div>
              </>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
function PhysicsMetric({
  label,
  value,
  note,
}: {
  label: string;
  value: string;
  note: string;
}) {
  return (
    <div className="metric-card">
      <span className="metric-label">{label}</span>
      <strong>{value}</strong>
      <span className="metric-note">{note}</span>
    </div>
  );
}
function Telemetry({
  steps,
  field,
  title,
  unit,
}: {
  steps: Step[];
  field: "internal_force_sum_raw" | "max_vertex_displacement_scene_units";
  title: string;
  unit: string;
}) {
  const values = steps.map((step) => step[field]),
    min = Math.min(0, ...values),
    max = Math.max(...values, min + 0.000001),
    width = 280,
    height = 100;
  const points = steps
    .map(
      (step, index) =>
        `${45 + (steps.length === 1 ? 0.5 : index / (steps.length - 1)) * width},${135 - ((step[field] - min) / (max - min)) * height}`,
    )
    .join(" ");
  return (
    <div className="physics-telemetry-chart">
      <h3>{title}</h3>
      <span>{unit}</span>
      <svg
        viewBox="0 0 350 170"
        className="chart"
        role="img"
        aria-label={`${title} across recorded simulation steps`}
      >
        <line x1="45" x2="325" y1="135" y2="135" stroke="#dfe7d7" />
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line
              x1="45"
              x2="325"
              y1={135 - t * height}
              y2={135 - t * height}
              stroke="#e7eddf"
            />
            <text x="37" y={139 - t * height} textAnchor="end">
              {number(min + t * (max - min))}
            </text>
          </g>
        ))}
        <polyline
          points={points}
          stroke="#6d9762"
          strokeWidth="2.3"
          fill="none"
        />
        {steps.length === 1 && (
          <circle
            cx="185"
            cy={135 - ((values[0] - min) / (max - min)) * height}
            r="3"
            fill="#6d9762"
          />
        )}
        <text x="45" y="154">
          {fixed(steps[0]?.simulation_time_s, 2)}s
        </text>
        <text x="325" y="154" textAnchor="end">
          {fixed(steps.at(-1)?.simulation_time_s, 2)}s
        </text>
      </svg>
    </div>
  );
}
