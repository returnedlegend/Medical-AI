import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Clock3,
  Database,
  FlaskConical,
  GitBranch,
  Layers3,
  LoaderCircle,
  Play,
  RotateCcw,
  Settings2,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
  X,
} from "lucide-react";
import { api, type Run, type RunConfig } from "./api";
import type { Results, Patient, Outcome } from "./types";
import Explorer from "./Explorer";
import { Anatomy, ForcePlot, CalibrationChart, pct, fixed } from "./visuals";

type Tab = "overview" | "patients" | "methods";
const defaults: RunConfig = {
  n_patients: 240,
  seed: 42,
  threshold_profile: "nominal",
};
const categoryNames = {
  standard: "Shared strategy succeeds",
  personalize: "Personalization needed",
  bail_out: "No grid strategy succeeds",
};
const errText = (error: unknown) =>
  error instanceof Error
    ? error.message
    : "Something went wrong. Please try again.";
const time = (date: string) =>
  new Date(date).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

export default function App() {
  const [tab, setTab] = useState<Tab>("overview"),
    [config, setConfig] = useState<RunConfig>(defaults);
  const [runs, setRuns] = useState<Run[]>([]),
    [selected, setSelected] = useState<Run | null>(null),
    [results, setResults] = useState<Results | null>(null);
  const [health, setHealth] = useState<"checking" | "online" | "offline">(
      "checking",
    ),
    [error, setError] = useState(""),
    [creating, setCreating] = useState(false),
    [loading, setLoading] = useState(false);
  const [patientId, setPatientId] = useState<string>(""),
    [strategy, setStrategy] = useState<"standard" | "shared" | "oracle">(
      "oracle",
    ),
    [filter, setFilter] = useState("all"),
    [showHistory, setShowHistory] = useState(false);
  const [advanced, setAdvanced] = useState(false),
    [configCopied, setConfigCopied] = useState(false);
  const requestId = useRef(0);
  const selectedId = selected?.id;
  const busy =
    creating || selected?.status === "running" || selected?.status === "queued";
  const patient =
    results?.patients.find((p) => String(p.id) === patientId) ||
    results?.patients.find((p) => p.split === "test") ||
    results?.patients[0];
  const selectRun = async (run: Run) => {
    const token = ++requestId.current;
    setSelected(run);
    setResults(null);
    setLoading(false);
    setError(
      run.status === "failed"
        ? run.error || "This experiment failed. Please launch a new run."
        : "",
    );
    setShowHistory(false);
    if (run.status === "completed") {
      setLoading(true);
      try {
        const data = await api.results(run.id);
        if (token === requestId.current) {
          setResults(data as unknown as Results);
          setPatientId("");
        }
      } catch (e) {
        if (token === requestId.current) setError(errText(e));
      } finally {
        if (token === requestId.current) setLoading(false);
      }
    }
  };
  const refresh = async () => {
    try {
      await api.health();
      setHealth("online");
      const data = await api.runs();
      setError("");
      setRuns(data.runs);
      if (!selectedId && data.runs[0]) void selectRun(data.runs[0]);
    } catch (e) {
      setHealth("offline");
      setError(`The research service is unavailable. ${errText(e)}`);
    }
  };
  useEffect(() => {
    void refresh();
  }, []); // connect once; active runs have separate polling
  useEffect(() => {
    if (
      !selectedId ||
      creating ||
      !(selected?.status === "running" || selected?.status === "queued")
    )
      return;
    let stopped = false;
    const timer = window.setInterval(async () => {
      try {
        const updated = await api.run(selectedId);
        if (stopped) return;
        setError("");
        setSelected(updated);
        setRuns((old) => old.map((r) => (r.id === updated.id ? updated : r)));
        if (updated.status === "completed") {
          window.clearInterval(timer);
          setLoading(true);
          const data = await api.results(selectedId);
          if (!stopped) {
            setResults(data as unknown as Results);
            setPatientId("");
            setLoading(false);
          }
        }
        if (updated.status === "failed") {
          window.clearInterval(timer);
          setError(
            updated.error ||
              "Experiment failed. Check the server log for details.",
          );
        }
      } catch (e) {
        if (!stopped) {
          setError(errText(e));
          setLoading(false);
        }
      }
    }, 1000);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [selectedId, creating]);
  useEffect(() => {
    if (!showHistory) return;
    const previous = document.activeElement as HTMLElement | null;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setShowHistory(false);
        return;
      }
      if (event.key !== "Tab") return;
      const elements = Array.from(
        document.querySelectorAll<HTMLElement>(
          '.history-modal button, .history-modal a[href], .history-modal [tabindex="0"]',
        ),
      );
      const first = elements[0],
        last = elements[elements.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      previous?.focus();
    };
  }, [showHistory]);
  const launch = async () => {
    setError("");
    setCreating(true);
    try {
      const run = await api.create(config);
      ++requestId.current;
      setLoading(false);
      setSelected(run);
      setResults(null);
      setRuns((old) => [run, ...old]);
      setTab("overview");
      setHealth("online");
    } catch (e) {
      setError(errText(e));
    } finally {
      setCreating(false);
    }
  };
  const updateConfig = (key: string, value: number | string) => {
    setConfigCopied(false);
    setConfig((old) => ({ ...old, [key]: value }));
  };
  const total = runs.length;
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          href="#"
          className="brand"
          aria-label="Retraction Lab home"
          onClick={(e) => {
            e.preventDefault();
            setTab("overview");
          }}
        >
          <span className="brand-mark">
            <GitBranch size={24} />
          </span>
          <span>
            retraction<span className="brand-light">lab</span>
            <small>RESEARCH WORKBENCH</small>
          </span>
        </a>
        <div className="workspace-label">
          WORKSPACE <span>01</span>
        </div>
        <nav aria-label="Primary navigation">
          <button
            className={tab === "overview" ? "nav-item active" : "nav-item"}
            onClick={() => setTab("overview")}
          >
            <FlaskConical size={18} />
            Experiments<span className="nav-count">{total}</span>
          </button>
          <button
            className={tab === "patients" ? "nav-item active" : "nav-item"}
            onClick={() => setTab("patients")}
          >
            <Layers3 size={18} />
            Virtual patients
          </button>
          <button
            className={tab === "methods" ? "nav-item active" : "nav-item"}
            onClick={() => setTab("methods")}
          >
            <ShieldCheck size={18} />
            Methods & limits
          </button>
        </nav>
        <div className="sidebar-study">
          <span className="eyebrow">THE RESEARCH QUESTION</span>
          <p>
            Does one retraction
            <br />
            fit every patient?
          </p>
          <span>Explore when adaptation matters.</span>
          <div className="study-line">
            <span />
            <span />
            <span />
          </div>
        </div>
        <div className="sidebar-bottom">
          <span className="small-model-tag">
            <span className="status-dot" />
            Analytic surrogate · v0.1
          </span>
          <p>
            Computational research only.
            <br />
            No patient data or clinical use.
          </p>
          <button
            className="text-button sidebar-help"
            onClick={() => setTab("methods")}
          >
            <CircleHelp size={15} />
            About this model
            <ArrowUpRight size={13} />
          </button>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumbs">
            Research workspace
            <ChevronRight size={13} />
            <strong>
              {tab === "overview"
                ? "Experiments"
                : tab === "patients"
                  ? "Virtual patients"
                  : "Methods & limits"}
            </strong>
          </div>
          <div className="topbar-right">
            <button
              className={`service-status ${health}`}
              onClick={() => void refresh()}
              title="Check service connection"
            >
              <span className="status-dot" />
              {health === "online"
                ? "Service connected"
                : health === "checking"
                  ? "Connecting…"
                  : "Service offline"}
            </button>
            <span className="avatar">RL</span>
          </div>
        </header>
        <main>
          <div className="page-heading">
            <div>
              <div className="eyebrow">PATIENT-SPECIFIC STRATEGY SEARCH</div>
              <h1>
                {tab === "overview"
                  ? "Experiment workbench"
                  : tab === "patients"
                    ? "The virtual cohort"
                    : "Methods & limitations"}
              </h1>
              <p>
                {tab === "overview"
                  ? "Explore how patient variation changes safe exposure in a simplified model."
                  : tab === "patients"
                    ? "Inspect individual mechanics, retraction choices, and probe responses."
                    : "What the model measures, how the study is evaluated, and where it stops."}
              </p>
            </div>
            <button
              className="secondary-button"
              onClick={() => setShowHistory(true)}
            >
              <Clock3 size={16} />
              Run history<span className="button-count">{total}</span>
            </button>
          </div>
          {error && (
            <div role="alert" className="error-banner">
              <TriangleAlert size={18} />
              <span>{error}</span>
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          {tab === "methods" ? (
            <Methods />
          ) : (
            <>
              {tab === "overview" && (
                <div className="hero-strip">
                  <div>
                    <span className="hero-kicker">
                      <span />
                      EXPLORATORY SIMULATION
                    </span>
                    <h2>
                      A different patient.
                      <br />A better strategy?
                    </h2>
                    <p>
                      Search a finite set of grasp and pull choices.
                      <br />
                      Find the value—and limits—of personalization.
                    </p>
                    <div className="hero-badges">
                      <span>
                        <Check size={13} />
                        Reproducible by seed
                      </span>
                      <span>
                        <Check size={13} />
                        Held-out evaluation
                      </span>
                    </div>
                  </div>
                  <div className="hero-visual">
                    <Anatomy compact />
                    <div className="hero-visual-label">
                      <span className="status-dot" />
                      Fixed fundus · variable neck pull
                    </div>
                  </div>
                  <div className="hero-caption">
                    <span>01 / SEARCH</span>
                    <p>
                      Grasp point
                      <br />
                      Pull direction
                      <br />
                      Pull distance
                    </p>
                    <ArrowUpRight size={25} />
                  </div>
                </div>
              )}
              <div
                className={
                  tab === "overview" ? "experiment-layout" : "patient-layout"
                }
              >
                {tab === "overview" && (
                  <aside className="config-panel panel">
                    <div className="panel-heading">
                      <h2>
                        <Settings2 size={17} />
                        Experiment setup
                      </h2>
                      <span className="step-badge">01</span>
                    </div>
                    <form
                      onSubmit={(e) => {
                        e.preventDefault();
                        void launch();
                      }}
                    >
                      <div className="form-section">
                        <label htmlFor="cohort">
                          Virtual patients{" "}
                          <span className="field-value">
                            {config.n_patients}
                          </span>
                        </label>
                        <input
                          id="cohort"
                          type="range"
                          min="40"
                          max="2000"
                          step="20"
                          value={config.n_patients}
                          onChange={(e) =>
                            updateConfig("n_patients", Number(e.target.value))
                          }
                        />
                        <div className="range-labels">
                          <span>40</span>
                          <span>2,000</span>
                        </div>
                        <p className="field-note">
                          Randomized attachment, adhesions, and distension. 70%
                          training · 30% held out.
                        </p>
                      </div>
                      <div className="form-section">
                        <label htmlFor="threshold">Threshold profile</label>
                        <div className="select-wrap">
                          <select
                            id="threshold"
                            value={config.threshold_profile}
                            onChange={(e) =>
                              updateConfig("threshold_profile", e.target.value)
                            }
                          >
                            <option value="conservative">Conservative</option>
                            <option value="nominal">Nominal</option>
                            <option value="permissive">Permissive</option>
                          </select>
                          <ChevronDown size={15} />
                        </div>
                        <p className="field-note">
                          Model force, strain, and exposure limits. These are
                          research assumptions.
                        </p>
                      </div>
                      <div className="form-section">
                        <label htmlFor="seed">
                          Random seed{" "}
                          <span className="optional-label">
                            REPRODUCIBILITY
                          </span>
                        </label>
                        <div className="seed-input">
                          <input
                            id="seed"
                            type="number"
                            min="0"
                            max="4294967295"
                            required
                            value={config.seed}
                            onChange={(e) =>
                              updateConfig("seed", Number(e.target.value))
                            }
                          />
                          <button
                            type="button"
                            aria-label="Choose a new random seed"
                            onClick={() =>
                              updateConfig(
                                "seed",
                                Math.floor(Math.random() * 100000),
                              )
                            }
                          >
                            <RotateCcw size={15} />
                          </button>
                        </div>
                      </div>
                      <button
                        type="button"
                        className="advanced-toggle"
                        aria-expanded={advanced}
                        onClick={() => setAdvanced(!advanced)}
                      >
                        Patient parameter ranges
                        <ChevronDown
                          size={14}
                          className={advanced ? "rotate" : ""}
                        />
                      </button>
                      {advanced && (
                        <div className="advanced-fields">
                          <NumberField
                            id="attachment-min"
                            label="Attachment minimum (mm)"
                            value={Number(config.attachment_min_mm ?? 8)}
                            min={4}
                            max={60}
                            onChange={(v) =>
                              updateConfig("attachment_min_mm", v)
                            }
                          />
                          <NumberField
                            id="attachment-max"
                            label="Attachment maximum (mm)"
                            value={Number(config.attachment_max_mm ?? 35)}
                            min={4}
                            max={60}
                            onChange={(v) =>
                              updateConfig("attachment_max_mm", v)
                            }
                          />
                          <NumberField
                            id="adhesions"
                            label="Maximum adhesion count"
                            value={Number(config.adhesion_max_count ?? 6)}
                            min={0}
                            max={12}
                            onChange={(v) =>
                              updateConfig("adhesion_max_count", v)
                            }
                          />
                          <NumberField
                            id="distension-max"
                            label="Maximum distension factor"
                            value={Number(config.distension_max ?? 1.8)}
                            min={0.8}
                            max={2.5}
                            step={0.1}
                            onChange={(v) => updateConfig("distension_max", v)}
                          />
                        </div>
                      )}
                      <div className="config-summary">
                        <Database size={16} />
                        <span>
                          Finite strategy grid
                          <br />
                          <strong>No external data or GPU required</strong>
                        </span>
                      </div>
                      <button
                        className="primary-button launch"
                        type="submit"
                        disabled={busy || health !== "online"}
                      >
                        {busy ? (
                          <LoaderCircle size={17} className="spin" />
                        ) : (
                          <Play size={15} fill="currentColor" />
                        )}
                        {busy ? "Running experiment…" : "Run experiment"}
                        {!busy && <ArrowRight size={16} />}
                      </button>
                      <p className="launch-note">
                        Runs are saved locally. Results stay exploratory.
                      </p>
                    </form>
                  </aside>
                )}
                <section className="results-area" aria-busy={busy || loading}>
                  <div className="results-header">
                    <div>
                      <span className="eyebrow">
                        {results ? "LATEST RESULTS" : "EXPERIMENT RESULTS"}
                      </span>
                      <h2>
                        {selected
                          ? `Run ${selected.id.slice(0, 8)}`
                          : "Ready when you are"}
                        {selected?.status === "completed" && (
                          <span className="completed-chip">
                            <CheckCircle2 size={12} />
                            Complete
                          </span>
                        )}
                      </h2>
                      {selected && (
                        <p>
                          {time(selected.created_at)}
                          <span>·</span>Seed {selected.config.seed}
                          <span>·</span>
                          {selected.config.n_patients} patients
                        </p>
                      )}
                    </div>
                    {results && selected && (
                      <div className="export-buttons">
                        <button
                          className="small-button reuse-button"
                          onClick={() => {
                            setConfig(selected.config);
                            setConfigCopied(true);
                            setTab("overview");
                            window.requestAnimationFrame(() =>
                              document.getElementById("cohort")?.focus(),
                            );
                          }}
                        >
                          {configCopied ? (
                            <Check size={13} />
                          ) : (
                            <RotateCcw size={13} />
                          )}
                          <span>{configCopied ? "Copied" : "Reuse setup"}</span>
                        </button>
                        <a
                          className="small-button"
                          href={`/api/runs/${encodeURIComponent(selected.id)}/export?format=csv`}
                          download
                        >
                          <ArrowDownToLine size={14} />
                          CSV
                        </a>
                        <a
                          className="small-button"
                          href={`/api/runs/${encodeURIComponent(selected.id)}/export?format=json`}
                          download
                        >
                          JSON
                        </a>
                        <a
                          className="small-button"
                          href={`/api/runs/${encodeURIComponent(selected.id)}/report`}
                          download
                          title="Download the research report as Markdown"
                        >
                          Report
                        </a>
                      </div>
                    )}
                  </div>
                  {busy || loading ? (
                    <div className="panel running-state" role="status">
                      <div className="loading-orbit">
                        <FlaskConical size={30} />
                        <span />
                      </div>
                      <h3>
                        {loading
                          ? "Loading experiment results"
                          : "Searching the strategy grid"}
                      </h3>
                      <p>
                        Simulating patients, comparing strategies, and
                        evaluating the probe model.
                      </p>
                      <div className="progress-track">
                        <span
                          style={{
                            width: `${Math.max(5, (selected?.progress ?? 0) * 100)}%`,
                          }}
                        />
                      </div>
                      <span className="eyebrow">
                        {selected?.status === "queued"
                          ? "QUEUED"
                          : loading
                            ? "LOADING"
                            : "EXPERIMENT IN PROGRESS"}
                      </span>
                    </div>
                  ) : !results ? (
                    <div className="panel empty-state">
                      <span className="empty-icon">
                        <FlaskConical size={31} />
                      </span>
                      <span className="eyebrow">
                        YOUR FIRST EXPERIMENT STARTS HERE
                      </span>
                      <h3>When does personalization matter?</h3>
                      <p>
                        Build a virtual cohort and compare a shared strategy
                        with the best strategy found for each patient.
                      </p>
                      <div className="empty-steps">
                        <span>
                          <b>1</b>Generate patients
                        </span>
                        <ChevronRight size={13} />
                        <span>
                          <b>2</b>Search strategies
                        </span>
                        <ChevronRight size={13} />
                        <span>
                          <b>3</b>Compare outcomes
                        </span>
                      </div>
                      {tab === "patients" && (
                        <button
                          className="secondary-button"
                          onClick={() => setTab("overview")}
                        >
                          <ArrowLeft size={15} />
                          Set up an experiment
                        </button>
                      )}
                    </div>
                  ) : (
                    <>
                      {tab === "overview" && (
                        <ResultsOverview results={results} />
                      )}
                      <div className="panel patient-panel">
                        <div className="panel-heading">
                          <div>
                            <h2>Patient explorer</h2>
                            <p>
                              Compare retraction choices in an individual
                              virtual patient.
                            </p>
                          </div>
                          <span className="step-badge">
                            {tab === "overview" ? "03" : "01"}
                          </span>
                        </div>
                        <div className="patient-toolbar">
                          <div className="select-wrap patient-select">
                            <label className="sr-only" htmlFor="patient">
                              Select virtual patient
                            </label>
                            <select
                              id="patient"
                              value={patient ? String(patient.id) : ""}
                              onChange={(e) => setPatientId(e.target.value)}
                            >
                              {results.patients.map((p) => (
                                <option key={p.id} value={String(p.id)}>
                                  {p.id} · {p.split} ·{" "}
                                  {categoryNames[p.category]}
                                </option>
                              ))}
                            </select>
                            <ChevronDown size={14} />
                          </div>
                          <div
                            className="segmented"
                            aria-label="Retraction strategy"
                          >
                            {(["standard", "shared", "oracle"] as const).map(
                              (s) => (
                                <button
                                  key={s}
                                  className={strategy === s ? "selected" : ""}
                                  onClick={() => setStrategy(s)}
                                  aria-pressed={strategy === s}
                                >
                                  {s === "standard"
                                    ? "Reference"
                                    : s === "shared"
                                      ? "Shared"
                                      : "Personalized"}
                                </button>
                              ),
                            )}
                          </div>
                        </div>
                        {patient && (
                          <PatientDetail
                            patient={patient}
                            outcome={patient[strategy]}
                            strategy={strategy}
                            config={results.config}
                            runId={selected!.id}
                          />
                        )}
                      </div>
                      {tab === "overview" && <ProbePanel results={results} />}
                      {tab === "patients" && (
                        <div className="panel cohort-panel">
                          <div className="panel-heading">
                            <div>
                              <h2>Cohort ledger</h2>
                              <p>
                                All generated patients. Reported study rates use
                                the held-out split.
                              </p>
                            </div>
                            <div className="select-wrap">
                              <label className="sr-only" htmlFor="filter">
                                Filter cohort
                              </label>
                              <select
                                id="filter"
                                value={filter}
                                onChange={(e) => setFilter(e.target.value)}
                              >
                                <option value="all">All patients</option>
                                <option value="test">Held-out only</option>
                                <option value="personalize">
                                  Personalization needed
                                </option>
                                <option value="bail_out">Bail-out zone</option>
                              </select>
                              <ChevronDown size={14} />
                            </div>
                          </div>
                          <div className="table-scroll">
                            <table>
                              <thead>
                                <tr>
                                  <th>Patient</th>
                                  <th>Split</th>
                                  <th>Attachment</th>
                                  <th>Adhesions</th>
                                  <th>Distension</th>
                                  <th>Outcome</th>
                                </tr>
                              </thead>
                              <tbody>
                                {results.patients
                                  .filter(
                                    (p) =>
                                      filter === "all" ||
                                      p.split === filter ||
                                      p.category === filter,
                                  )
                                  .map((p) => (
                                    <tr
                                      key={p.id}
                                      className={
                                        String(p.id) === String(patient?.id)
                                          ? "selected-row"
                                          : ""
                                      }
                                    >
                                      <td>
                                        <button
                                          className="table-patient"
                                          onClick={() => {
                                            setPatientId(String(p.id));
                                            document
                                              .querySelector(".patient-panel")
                                              ?.scrollIntoView({
                                                behavior: "smooth",
                                                block: "start",
                                              });
                                          }}
                                        >
                                          {p.id}
                                          <ArrowUpRight size={12} />
                                        </button>
                                      </td>
                                      <td>
                                        {p.split === "test"
                                          ? "Held out"
                                          : "Training"}
                                      </td>
                                      <td>
                                        {fixed(p.attachment_length_mm, 1)} mm
                                      </td>
                                      <td>{p.adhesion_count}</td>
                                      <td>{fixed(p.distension, 2)}×</td>
                                      <td>
                                        <span
                                          className={`category-tag ${p.category}`}
                                        >
                                          {p.category === "standard"
                                            ? "Shared succeeds"
                                            : p.category === "personalize"
                                              ? "Personalize"
                                              : "Bail out"}
                                        </span>
                                      </td>
                                    </tr>
                                  ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </section>
              </div>
            </>
          )}
          <footer className="page-footer">
            <span>
              <FlaskConical size={13} />
              Built for exploration, designed for scrutiny.
            </span>
            <span>
              Analytic surrogate · No clinical safety claims{" "}
              <button onClick={() => setTab("methods")}>
                Read limitations
                <ArrowUpRight size={12} />
              </button>
            </span>
          </footer>
        </main>
      </div>
      {showHistory && (
        <div className="modal-backdrop" onClick={() => setShowHistory(false)}>
          <section
            className="history-modal panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="history-title"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="panel-heading">
              <div>
                <h2 id="history-title">Experiment history</h2>
                <p>Revisit saved runs and their reproducible configurations.</p>
              </div>
              <button
                className="icon-button"
                aria-label="Close history"
                onClick={() => setShowHistory(false)}
                autoFocus
              >
                <X size={20} />
              </button>
            </div>
            {runs.length === 0 ? (
              <p className="history-empty">
                No runs yet. Your first experiment will appear here.
              </p>
            ) : (
              <div className="history-list">
                {runs.map((r) => (
                  <button key={r.id} onClick={() => void selectRun(r)}>
                    <span className="history-icon">
                      <FlaskConical size={18} />
                    </span>
                    <span>
                      <strong>Run {r.id.slice(0, 8)}</strong>
                      <small>
                        {time(r.created_at)} · {r.config.n_patients} patients ·
                        seed {r.config.seed}
                      </small>
                    </span>
                    <span className={`run-status ${r.status}`}>{r.status}</span>
                    <ChevronRight size={16} />
                  </button>
                ))}
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}

function NumberField({
  id,
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        required
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}
function ResultsOverview({ results }: { results: Results }) {
  const s = results.summary;
  return (
    <>
      <div className="metrics-grid">
        <Metric
          label="Best shared strategy"
          value={pct(s.shared_success_rate)}
          note="Held-out safe-exposure proxy"
        />
        <Metric
          label="Personalized strategy"
          value={pct(s.oracle_success_rate)}
          note="Best result in the finite grid"
          accent
        />
        <Metric
          label="Value of personalization"
          value={`+${(s.personalization_gain * 100).toFixed(1)}`}
          unit="pp"
          note={`95% bootstrap CI: ${(s.personalization_gain_ci?.[0] * 100).toFixed(1)}–${(s.personalization_gain_ci?.[1] * 100).toFixed(1)} pp`}
          accent
        />
        <Metric
          label="Bail-out zone"
          value={pct(s.bailout_rate)}
          note="No tested strategy succeeds"
          warning
        />
      </div>
      <div className="panel comparison-panel">
        <div className="panel-heading">
          <div>
            <h2>One strategy, or one per patient?</h2>
            <p>Success requires exposure and both model limits to pass.</p>
          </div>
          <span className="heldout-tag">HELD OUT · n = {s.test_size}</span>
        </div>
        <div className="comparison-body">
          <div className="comparison-bars">
            {[
              {
                label: "Reference strategy",
                value: s.standard_success_rate,
                style: "reference",
              },
              {
                label: "Best shared strategy",
                value: s.shared_success_rate,
                style: "shared",
              },
              {
                label: "Personalized grid search",
                value: s.oracle_success_rate,
                style: "personalized",
              },
            ].map((row) => (
              <div className="comparison-row" key={row.style}>
                <span>{row.label}</span>
                <div className="bar-track">
                  <div
                    className={`bar ${row.style}`}
                    style={{ width: `${row.value * 100}%` }}
                  />
                </div>
                <strong>{pct(row.value, 1)}</strong>
              </div>
            ))}
          </div>
          <div className="comparison-aside">
            <span className="eyebrow">SEARCH SPACE</span>
            <strong>{s.strategy_count.toLocaleString()}</strong>
            <p>strategies per patient</p>
            <span>
              {s.train_size} training · {s.test_size} held out
            </span>
          </div>
        </div>
        <div className="panel-footnote">
          <GitBranch size={14} />
          The shared strategy is selected on training patients. The oracle
          selects per patient within the same finite grid.
        </div>
      </div>
      {results.subgroups?.length ? <SubgroupPanel results={results} /> : null}
    </>
  );
}

function SubgroupPanel({ results }: { results: Results }) {
  const [dimension, setDimension] = useState("attachment_length");
  const rows =
    results.subgroups?.filter((g) => g.dimension === dimension) ?? [];
  return (
    <div className="panel subgroup-panel">
      <div className="panel-heading">
        <div>
          <h2>Where is adaptation concentrated?</h2>
          <p>
            Exploratory held-out comparisons across illustrative patient ranges.
          </p>
        </div>
      </div>
      <div className="subgroup-body">
        <div className="segmented" aria-label="Patient characteristic">
          {[
            { key: "attachment_length", label: "Attachment" },
            { key: "adhesions", label: "Adhesions" },
            { key: "distension", label: "Distension" },
          ].map((d) => (
            <button
              key={d.key}
              className={dimension === d.key ? "selected" : ""}
              aria-pressed={dimension === d.key}
              onClick={() => setDimension(d.key)}
            >
              {d.label}
            </button>
          ))}
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Patient group</th>
                <th>Held out</th>
                <th>Shared success</th>
                <th>Personalized</th>
                <th>Gain (95% CI)</th>
                <th>Bail-out zone</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.group}>
                  <td>{row.label}</td>
                  <td>{row.n}</td>
                  <td>{row.n ? pct(row.shared_success_rate, 1) : "—"}</td>
                  <td>{row.n ? pct(row.oracle_success_rate, 1) : "—"}</td>
                  <td className="gain-cell">
                    {row.n
                      ? `+${(row.personalization_gain * 100).toFixed(1)} pp`
                      : "—"}
                    {row.n > 0 && row.gain_ci && (
                      <small>
                        ({(row.gain_ci[0] * 100).toFixed(1)}–
                        {(row.gain_ci[1] * 100).toFixed(1)})
                      </small>
                    )}
                  </td>
                  <td>{row.n ? pct(row.bailout_rate, 1) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="panel-footnote">
        <TriangleAlert size={14} />
        Patient-specific preferred directions are imposed by the model. These
        comparisons test the pipeline, not a biological hypothesis.
      </div>
    </div>
  );
}
function Metric({
  label,
  value,
  unit,
  note,
  accent,
  warning,
}: {
  label: string;
  value: string;
  unit?: string;
  note: string;
  accent?: boolean;
  warning?: boolean;
}) {
  return (
    <div
      className={`metric-card ${accent ? "accent" : ""} ${warning ? "warning" : ""}`}
    >
      <span className="metric-label">{label}</span>
      <strong>
        {value}
        {unit && <small>{unit}</small>}
      </strong>
      <span className="metric-note">{note}</span>
    </div>
  );
}
function PatientDetail({
  patient: p,
  outcome: o,
  strategy,
  config,
  runId,
}: {
  patient: Patient;
  outcome: Outcome;
  strategy: string;
  config: Record<string, unknown>;
  runId: string;
}) {
  return (
    <>
      <div className="patient-detail">
        <div className="patient-anatomy">
          <Anatomy patient={p} outcome={o} />
          <div className="anatomy-key">
            <span>
              <i className="key-tissue" />
              Modeled tissue
            </span>
            <span>
              <i className="key-pull" />
              Pull vector
            </span>
            <span>
              <i className="key-exposure" />
              Exposure region
            </span>
          </div>
        </div>
        <div className="patient-readout">
          <div className="readout-heading">
            <span className="eyebrow">
              {p.split === "test" ? "HELD-OUT PATIENT" : "TRAINING PATIENT"}
            </span>
            <span className={`outcome-chip ${o.success ? "pass" : "fail"}`}>
              {o.success ? (
                <CheckCircle2 size={13} />
              ) : (
                <TriangleAlert size={13} />
              )}{" "}
              {o.success ? "Proxy succeeds" : "Proxy fails"}
            </span>
          </div>
          <h3>{p.id}</h3>
          <div className="patient-params">
            <span>
              Attachment
              <strong>
                {fixed(p.attachment_length_mm, 1)}
                <small>mm</small>
              </strong>
            </span>
            <span>
              Adhesions
              <strong>
                {p.adhesion_count}
                <small>tethers</small>
              </strong>
            </span>
            <span>
              Distension
              <strong>
                {fixed(p.distension, 2)}
                <small>×</small>
              </strong>
            </span>
          </div>
          <div className="selected-strategy">
            <span className="eyebrow">
              {strategy === "oracle"
                ? "BEST IN GRID"
                : strategy === "shared"
                  ? "TRAINING-SELECTED SHARED"
                  : "FIXED REFERENCE"}{" "}
              STRATEGY
            </span>
            <p>
              Grasp {fixed(o.strategy.grasp_position, 2)}
              <span>·</span>
              {fixed(o.strategy.direction_deg, 0)}° pull<span>·</span>
              {fixed(o.strategy.distance_mm, 0)} mm
            </p>
          </div>
          <div className="outcome-grid">
            <div>
              <span>Peak force</span>
              <strong>
                {fixed(o.force_n)}
                <small>N</small>
              </strong>
            </div>
            <div>
              <span>Tissue strain</span>
              <strong>{pct(o.strain, 1)}</strong>
            </div>
            <div>
              <span>Exposure proxy</span>
              <strong>{pct(o.exposure, 1)}</strong>
            </div>
          </div>
          <div className="threshold-readout">
            <span>Force ≤ {fixed(Number(config.force_limit_n), 1)} N</span>
            <span>Strain ≤ {pct(Number(config.strain_limit))}</span>
            <span>Exposure ≥ {pct(Number(config.exposure_threshold))}</span>
          </div>
          <p className="readout-note">
            {categoryNames[p.category]}. Category is relative to the best shared
            strategy and the current thresholds.
          </p>
        </div>
      </div>
      <div className="patient-probe">
        <div>
          <span className="eyebrow">ONE-SECOND FORCE PROBE</span>
          <h3>A small tug. An early signal.</h3>
          <p>
            A fixed probe uses the same motion for every patient. The prediction
            model sees only its force response.
          </p>
          {p.probe.within_limits === false && (
            <p className="probe-warning">
              <TriangleAlert size={12} />
              Probe exceeded a model force or strain limit. Its response is
              retained.
            </p>
          )}
          <div className="probe-patient-values">
            <span>
              Peak force <strong>{fixed(p.probe.peak_force_n)} N</strong>
            </span>
            <span>
              Personalization probability{" "}
              <strong>{pct(p.personalization_probability, 0)}</strong>
            </span>
          </div>
        </div>
        <ForcePlot patient={p} />
      </div>
      <Explorer patient={p} reference={o} config={config} runId={runId} />
    </>
  );
}
function ProbePanel({ results }: { results: Results }) {
  const p = results.probe;
  const ready = p.auroc != null;
  return (
    <div className="panel probe-panel">
      <div className="panel-heading">
        <div>
          <h2>
            <Activity size={17} />
            Can the probe tell us who needs help?
          </h2>
          <p>Early-warning model evaluated only on held-out patients.</p>
        </div>
        <span className="step-badge">04</span>
      </div>
      <div className="probe-analysis">
        <div className="probe-analysis-copy">
          <span className="eyebrow">PERSONALIZATION EARLY WARNING</span>
          <div className="auroc-value">
            {fixed(p.auroc, 3)}
            <span>AUROC</span>
          </div>
          <p>
            {ready
              ? "A logistic model predicts whether the shared strategy fails but a grid strategy succeeds."
              : "AUROC needs both outcome classes in the held-out split and a fitted training model. This run does not meet those conditions."}
          </p>
          <div className="probe-stat-row">
            <div>
              <span>Brier score</span>
              <strong>{fixed(p.brier_score, 3)}</strong>
            </div>
            <div>
              <span>Calibration error</span>
              <strong>{fixed(p.ece, 3)}</strong>
            </div>
            <div>
              <span>Bail-out AUROC</span>
              <strong>{fixed(p.bailout?.auroc, 3)}</strong>
            </div>
          </div>
          <p className="subtle-note">
            Brier score and calibration error: lower is better. Scores
            characterize this synthetic model, not clinical prediction.
          </p>
        </div>
        <div className="calibration-plot">
          <div>
            <h3>Does confidence match reality?</h3>
            <span>Reliability diagram · dots sized by count</span>
          </div>
          {p.calibration?.some((b) => b.count > 0) ? (
            <CalibrationChart bins={p.calibration} />
          ) : (
            <p className="chart-empty">
              No held-out calibration bins available.
            </p>
          )}
          <div className="plot-key">
            <span>
              <i />
              Observed outcomes
            </span>
            <span>
              <i className="dashed" />
              Perfect calibration
            </span>
          </div>
        </div>
      </div>
      <div className="sensitivity-section">
        <h3>Threshold sensitivity</h3>
        <p>
          The same cohort and search are re-evaluated at three assumed limit
          profiles.
        </p>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Profile</th>
                <th>Force limit</th>
                <th>Strain limit</th>
                <th>Shared success</th>
                <th>Personalized success</th>
                <th>Gain</th>
                <th>Bail-out zone</th>
              </tr>
            </thead>
            <tbody>
              {results.sensitivity?.map((row) => (
                <tr key={row.profile}>
                  <td className="profile-name">{row.profile}</td>
                  <td>{fixed(row.force_limit_n, 1)} N</td>
                  <td>{pct(row.strain_limit)}</td>
                  <td>{pct(row.shared_success_rate, 1)}</td>
                  <td>{pct(row.oracle_success_rate, 1)}</td>
                  <td className="gain-cell">
                    +{(row.personalization_gain * 100).toFixed(1)} pp
                  </td>
                  <td>{pct(row.bailout_rate, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="panel-footnote">
        <TriangleAlert size={14} />
        No trained RL agent or clinically validated confidence estimate is
        included in this MVP.
      </div>
    </div>
  );
}
function Methods() {
  return (
    <div className="methods-page">
      <div className="method-intro panel">
        <span className="method-icon">
          <ShieldCheck size={28} />
        </span>
        <div>
          <span className="eyebrow">A TRANSPARENT STARTING POINT</span>
          <h2>A workbench for testing the research pipeline.</h2>
          <p>
            This MVP implements the strategy-search and early-warning workflow
            from <em>Does One Retraction Fit All?</em> using a deterministic
            analytic surrogate. It is a computational prototype for developing
            experiments, not a validated surgical simulator.
          </p>
        </div>
      </div>
      <div className="method-grid">
        {[
          {
            icon: <Layers3 size={21} />,
            title: "01 · Virtual patients",
            body: "Seeded cohorts vary attachment length, adhesion count and stiffness, and distension. The preferred pull direction is imposed by a rule tied to these parameters, so adaptation is partly built into the model. These are synthetic parameters sampled from configurable ranges. They are not inferred from patient imaging or clinical datasets.",
          },
          {
            icon: <GitBranch size={21} />,
            title: "02 · Finite strategy search",
            body: "A fixed reference is compared with a shared strategy selected on training patients and a per-patient grid oracle. The oracle is the best choice within the enumerated grid, not a global optimum over all possible motions.",
          },
          {
            icon: <ShieldCheck size={21} />,
            title: "03 · Exposure and constraints",
            body: "Success requires the model exposure proxy to cross its threshold while modeled peak force and strain remain within limits. None of these assumed thresholds establishes clinical safety or achievement of the critical view of safety.",
          },
          {
            icon: <Activity size={21} />,
            title: "04 · Early-warning evaluation",
            body: "One fixed, one-second probe provides force-response features. Logistic regression is fit only on the training split. AUROC, Brier score, reliability bins, and calibration error are reported on held-out patients; undefined metrics remain unavailable.",
          },
          {
            icon: <Database size={21} />,
            title: "05 · Reproducibility",
            body: "Each run stores its configuration, seed, patient records, search outcomes, and results. Export JSON for the full record, CSV for patient-level analysis, or a Markdown report for a readable summary. The shared strategy is chosen before held-out outcome evaluation.",
          },
          {
            icon: <Sparkles size={21} />,
            title: "06 · What comes next",
            body: "LapGym/SOFA soft-tissue integration, mesh convergence studies, expert review of mechanics, independently seeded reinforcement-learning agents, and external validation are extensions. No reinforcement-learning or SOFA result is represented by this interface.",
          },
        ].map((item) => (
          <article className="panel method-card" key={item.title}>
            <span>{item.icon}</span>
            <h3>{item.title}</h3>
            <p>{item.body}</p>
          </article>
        ))}
      </div>
      <div className="limitations-callout">
        <TriangleAlert size={24} />
        <div>
          <h3>Research use only</h3>
          <p>
            The model omits real anatomy, inflammation, bleeding, instrument
            contact, and patient-specific tissue mechanics. Its findings are
            hypotheses for technical and expert review. Do not use it to plan
            treatment, guide surgery, or establish clinical safety.
          </p>
        </div>
      </div>
    </div>
  );
}
