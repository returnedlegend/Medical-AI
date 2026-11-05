import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  Activity,
  ArrowDown,
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  ChevronDown,
  Flame,
  Grip,
  Hand,
  LoaderCircle,
  Pause,
  Play,
  Plus,
  RotateCcw,
  SkipForward,
  Terminal,
  TriangleAlert,
} from "lucide-react";
import { request } from "./api";
import "./procedure.css";
const ThreeViewport = lazy(() => import("./ThreeViewport"));
type Tool = "grasper" | "cautery";
type Controller = "manual" | "baseline" | "trained" | "protocol";
interface ResponseScenario {
  id: string;
  title: string;
  description: string;
  scope: string;
  protocol_version: string;
  seed: number;
  config: Record<string, unknown>;
  parameters: Record<string, unknown>;
}
interface ResponseProtocol {
  scenario_id: string;
  version: string;
  status: string;
  phase: string;
  protocol_step: number;
  completed: boolean;
  completion_reason: string | null;
  target_node_id: string | null;
  metadata?: ResponseScenario;
}
interface TissueNode {
  id: string;
  kind: string;
  rest_position_mm: number[];
  position_mm: number[];
  damage: number;
  mechanical_damage: number;
  thermal_damage: number;
  temperature_c: number;
  thermal_dose: number;
  strain: number;
  stiffness_n_per_mm: number;
}
export interface TissueState {
  time_s: number;
  step_count: number;
  tools: {
    grasper: {
      position_mm: number[];
      grip: number;
      contact_node_id: string | null;
      force_n: number;
    };
    cautery: { position_mm: number[]; power: number };
  };
  tissue: { nodes: TissueNode[]; edges: [string, string][] };
  physiology: {
    blood_loss_ml: number;
    bleeding_rate_ml_per_s: number;
    bile_leak_ml: number;
    bile_leak_rate_ml_per_s: number;
    heart_rate_bpm: number;
    mean_arterial_pressure_mmhg: number;
  };
  patient?: Record<string, number>;
  limits?: Record<string, number>;
  exposure: number;
  peak_exposure: number;
  stable_exposure_steps: number;
  termination_reason: string | null;
}
interface Episode {
  id: string;
  revision: number;
  status: string;
  seed: number;
  config: Record<string, unknown>;
  state: TissueState;
  observation: number[];
  info: Record<string, unknown>;
  reward: number;
  reward_components: Record<string, number>;
  events: unknown[];
  episode_reward: number;
  terminated: boolean;
  truncated: boolean;
  created_at: string;
  engine_version: string;
  physics_fingerprint?: string;
  engine_compatible?: boolean;
  read_only_reason?: string | null;
  policy_context?: PolicyContext;
  protocol?: ResponseProtocol | null;
  source?: { type?: string; phase?: string; scenario_id?: string } | null;
}
interface PolicyContext {
  training_run_id: string;
  out_of_distribution: boolean;
  comparison_available: boolean;
  patient_override: boolean;
  reasons: string[];
  differing_parameters?: {
    parameter: string;
    training: unknown;
    session: unknown;
  }[];
  interpretation?: string;
}
interface EngineSpec {
  engine_version: string;
  physics_fingerprint?: string;
}
interface TrainingRun {
  id: string;
  status: string;
  progress: number;
  error: string | null;
  config: Record<string, unknown>;
  summary: Record<string, unknown> | null;
}
interface EvaluationSummary {
  n: number;
  mean_return: number;
  mean_return_ci?: number[];
  success_rate: number;
  success_rate_ci?: number[];
  severe_harm_rate: number;
  severe_harm_rate_ci?: number[];
  handoff_rate: number;
  mean_max_damage: number;
}
interface LearningResult {
  algorithm: string;
  model_version?: string;
  checkpoint?: { model_version?: string; physics_fingerprint?: string };
  learning_curve: {
    episode: number;
    return: number;
    mean_td_loss?: number | null;
  }[];
  evaluation: Record<string, EvaluationSummary>;
  metadata?: { training_method?: string };
}
interface Trace {
  time: number;
  heart: number;
  map: number;
  reward: number;
  total: number;
  force: number;
  damage: number;
  blood: number;
}
const fmt = (v: number | undefined, d = 2) =>
  v == null || !Number.isFinite(v) ? "—" : v.toFixed(d);
const errorMessage = (e: unknown) =>
  e instanceof Error ? e.message : "The local engine request failed.";
export default function Procedure() {
  const [session, setSession] = useState<Episode | null>(null),
    [sessions, setSessions] = useState<Episode[]>([]),
    [seed, setSeed] = useState(2),
    [tool, setTool] = useState<Tool>("grasper"),
    [distance, setDistance] = useState(0.5),
    [power, setPower] = useState(0.35),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [running, setRunning] = useState(false),
    [trace, setTrace] = useState<Trace[]>([]),
    [log, setLog] = useState<{ step: number; text: string }[]>([]),
    [inspector, setInspector] = useState("tissue"),
    [training, setTraining] = useState<TrainingRun | null>(null),
    [trainings, setTrainings] = useState<TrainingRun[]>([]),
    [episodes, setEpisodes] = useState(150),
    [trainingBusy, setTrainingBusy] = useState(false);
  const current = useRef<Episode | null>(null),
    inFlight = useRef(false),
    runningRef = useRef(false),
    generation = useRef(0);
  const [gripTarget, setGripTarget] = useState(0.55);
  const [controller, setController] = useState<Controller>("manual");
  const controllerRef = useRef<Controller>("manual");
  controllerRef.current = controller;
  const selectedTrainingId = useRef<string | null>(null);
  selectedTrainingId.current = training?.id ?? null;
  const [scenarios, setScenarios] = useState<ResponseScenario[]>([]);
  const [scenarioId, setScenarioId] = useState("");
  const [protocolSeed, setProtocolSeed] = useState(2);
  const [scenarioError, setScenarioError] = useState("");
  const [scenarioRefresh, setScenarioRefresh] = useState(0);
  const selectedScenario = scenarios.find(
    (scenario) => scenario.id === scenarioId,
  );
  const [consoleTab, setConsoleTab] = useState<
    "events" | "rewards" | "learning"
  >("events");
  const [viewMode, setViewMode] = useState<"3d" | "xy">("3d");
  const [policyContext, setPolicyContext] = useState<PolicyContext | null>(
    null,
  );
  const [contextError, setContextError] = useState("");
  const [contextSessionId, setContextSessionId] = useState("");
  const [contextRetry, setContextRetry] = useState(0);
  const [learningRecord, setLearningRecord] = useState<{
    runId: string;
    result: LearningResult;
  } | null>(null);
  const learningResult =
    learningRecord && learningRecord.runId === training?.id
      ? learningRecord.result
      : null;
  const [learningFailure, setLearningFailure] = useState<{
    runId: string;
    message: string;
  } | null>(null);
  const learningError =
    learningFailure?.runId === training?.id
      ? (learningFailure?.message ?? "")
      : "";
  const [learningRetry, setLearningRetry] = useState(0);
  const [engineSpec, setEngineSpec] = useState<EngineSpec | null>(null);
  const [specError, setSpecError] = useState("");
  const [specRetry, setSpecRetry] = useState(0);
  const [algorithm, setAlgorithm] = useState<"dqn" | "tabular_q">("tabular_q");
  const [recordedFrames, setRecordedFrames] = useState<Episode[]>([]);
  const [replayIndex, setReplayIndex] = useState<number | null>(null);
  const [replayRunning, setReplayRunning] = useState(false);
  const inspectedEpisode =
    replayIndex === null ? session : recordedFrames[replayIndex];
  const state = inspectedEpisode?.state;
  const episodeFingerprint =
    session?.physics_fingerprint ?? session?.info.physics_fingerprint;
  const legacy =
    session?.engine_compatible === false ||
    !!(
      session &&
      engineSpec &&
      (session.engine_version !== engineSpec.engine_version ||
        (engineSpec.physics_fingerprint &&
          episodeFingerprint !== engineSpec.physics_fingerprint))
    );
  const checkpointCompatible = !!(
    learningResult &&
    engineSpec &&
    (learningResult.checkpoint?.model_version ??
      learningResult.model_version) === engineSpec.engine_version &&
    (!engineSpec.physics_fingerprint ||
      learningResult.checkpoint?.physics_fingerprint ===
        engineSpec.physics_fingerprint)
  );
  const checkpointArchived = !!(
    learningResult &&
    engineSpec &&
    !checkpointCompatible
  );
  const trainedReady = !!(
    session &&
    training?.status === "completed" &&
    checkpointCompatible &&
    policyContext &&
    contextSessionId === session.id &&
    policyContext.training_run_id === training.id
  );
  const referenceOrigin = training?.summary?.origin;
  const referenceProvenance =
    referenceOrigin &&
    typeof referenceOrigin === "object" &&
    "provenance" in referenceOrigin &&
    referenceOrigin.provenance &&
    typeof referenceOrigin.provenance === "object"
      ? (referenceOrigin.provenance as Record<string, unknown>)
      : null;
  const accept = useCallback((data: Episode, fresh = false) => {
    current.current = data;
    setSession(data);
    if (fresh) {
      setSeed(data.seed);
      if (data.protocol) {
        setController("protocol");
        setScenarioId(data.protocol.scenario_id);
        setProtocolSeed(data.seed);
        setConsoleTab("rewards");
      } else if (controllerRef.current === "protocol") setController("manual");
    }
    setReplayIndex(null);
    setReplayRunning(false);
    setRecordedFrames((old) =>
      fresh
        ? [data]
        : [
            ...old.filter((frame) => frame.revision !== data.revision),
            data,
          ].slice(-501),
    );
    const s = data.state;
    const point = {
      time: s.time_s,
      heart: s.physiology.heart_rate_bpm,
      map: s.physiology.mean_arterial_pressure_mmhg,
      reward: data.reward,
      total: data.episode_reward,
      force: s.tools.grasper.force_n,
      damage: Math.max(0, ...s.tissue.nodes.map((n) => n.damage)),
      blood: s.physiology.blood_loss_ml,
    };
    setTrace((old) => (fresh ? [point] : [...old, point].slice(-240)));
    const events = (data.events || []).map((e) => ({
      step: s.step_count,
      text:
        typeof e === "string"
          ? e
          : typeof e === "object" && e !== null && "message" in e
            ? String(e.message)
            : JSON.stringify(e),
    }));
    setLog((old) =>
      fresh
        ? [
            {
              step: 0,
              text: `Episode ${data.id.slice(0, 8)} initialized · seed ${data.seed}`,
            },
            ...events,
          ]
        : [...old, ...events].slice(-120),
    );
    setSessions((old) => [data, ...old.filter((e) => e.id !== data.id)]);
    if (
      data.terminated ||
      data.truncated ||
      data.status !== "active" ||
      (controllerRef.current === "protocol" && data.protocol?.completed)
    ) {
      setRunning(false);
      runningRef.current = false;
    }
  }, []);
  useEffect(() => {
    const abort = new AbortController();
    setScenarioError("");
    request<{ scenarios: ResponseScenario[] }>("/procedure/scenarios", {
      signal: abort.signal,
    })
      .then((data) => {
        if (!abort.signal.aborted) {
          setScenarios(data.scenarios);
          setScenarioId((old) => old || data.scenarios[0]?.id || "");
          setProtocolSeed(data.scenarios[0]?.seed ?? 2);
        }
      })
      .catch((e) => {
        if (!abort.signal.aborted) setScenarioError(errorMessage(e));
      });
    return () => abort.abort();
  }, [scenarioRefresh]);
  useEffect(() => {
    if (!replayRunning) return;
    if ((replayIndex ?? 0) >= recordedFrames.length - 1) {
      setReplayRunning(false);
      return;
    }
    const timer = window.setTimeout(
      () => setReplayIndex((replayIndex ?? 0) + 1),
      180,
    );
    return () => window.clearTimeout(timer);
  }, [replayRunning, recordedFrames.length, replayIndex]);
  useEffect(() => {
    let active = true;
    Promise.allSettled([
      request<{ sessions: Episode[] }>("/procedure/sessions"),
      request<{ runs: TrainingRun[] }>("/learning/runs"),
    ]).then(([history, policies]) => {
      if (!active) return;
      const failures: string[] = [];
      if (history.status === "fulfilled") setSessions(history.value.sessions);
      else failures.push(`Episode history: ${errorMessage(history.reason)}`);
      if (policies.status === "fulfilled") {
        setTrainings(policies.value.runs);
        const complete = policies.value.runs.find(
          (r) => r.status === "completed",
        );
        if (complete) setTraining(complete);
      } else failures.push(`Policy history: ${errorMessage(policies.reason)}`);
      if (failures.length) setError(failures.join(" · "));
    });
    return () => {
      active = false;
      generation.current++;
      runningRef.current = false;
    };
  }, []);
  useEffect(() => {
    const abort = new AbortController();
    setEngineSpec(null);
    setSpecError("");
    request<EngineSpec>("/procedure/spec", { signal: abort.signal })
      .then((data) => {
        if (!abort.signal.aborted) setEngineSpec(data);
      })
      .catch((e) => {
        if (!abort.signal.aborted) setSpecError(errorMessage(e));
      });
    return () => abort.abort();
  }, [specRetry]);
  useEffect(() => {
    setPolicyContext(null);
    setContextSessionId("");
    setContextError("");
    if (!session?.id || training?.status !== "completed") return;
    const abort = new AbortController();
    request<PolicyContext>(
      `/procedure/sessions/${session.id}/policy-context?training_run_id=${training.id}`,
      { signal: abort.signal },
    )
      .then((data) => {
        if (!abort.signal.aborted) {
          setPolicyContext(data);
          setContextSessionId(session.id);
        }
      })
      .catch((e) => {
        if (!abort.signal.aborted) setContextError(errorMessage(e));
      });
    return () => abort.abort();
  }, [session?.id, training?.id, training?.status, contextRetry]);
  useEffect(() => {
    setLearningRecord(null);
    setLearningFailure(null);
    if (training?.status !== "completed") return;
    const abort = new AbortController();
    request<LearningResult>(`/learning/runs/${training.id}/results`, {
      signal: abort.signal,
    })
      .then((data) => {
        if (!abort.signal.aborted)
          setLearningRecord({ runId: training.id, result: data });
      })
      .catch((e) => {
        if (!abort.signal.aborted)
          setLearningFailure({ runId: training.id, message: errorMessage(e) });
      });
    return () => abort.abort();
  }, [training?.id, training?.status, learningRetry]);
  const load = async (id: string) => {
    const token = ++generation.current;
    setRunning(false);
    setReplayRunning(false);
    runningRef.current = false;
    setBusy(true);
    setError("");
    try {
      const data = await request<Episode>(`/procedure/sessions/${id}`);
      if (token !== generation.current) return;
      accept(data, true);
      const records: Episode[] = [];
      let cursor = 0;
      for (let batch = 0; batch < 5; batch++) {
        const history = await request<{
          transitions: Episode[];
          next_revision: number;
          has_more: boolean;
        }>(
          `/procedure/sessions/${id}/transitions?after_revision=${cursor}&limit=200`,
        );
        if (token !== generation.current) return;
        records.push(...history.transitions);
        if (!history.has_more || history.next_revision <= cursor) break;
        cursor = history.next_revision;
      }
      if (records.length) {
        setRecordedFrames(records);
        setTrace(
          records.slice(-240).map((t) => ({
            time: t.state.time_s,
            heart: t.state.physiology.heart_rate_bpm,
            map: t.state.physiology.mean_arterial_pressure_mmhg,
            reward: t.reward,
            total: t.episode_reward,
            force: t.state.tools.grasper.force_n,
            damage: Math.max(0, ...t.state.tissue.nodes.map((n) => n.damage)),
            blood: t.state.physiology.blood_loss_ml,
          })),
        );
        setLog(
          [
            {
              step: 0,
              text: `Restored ${records.length} recorded transitions from episode ${id.slice(0, 8)}`,
            },
            ...records.flatMap((t) =>
              (t.events ?? []).map((e) => ({
                step: t.state.step_count,
                text:
                  typeof e === "string"
                    ? e
                    : typeof e === "object" && e !== null && "message" in e
                      ? String(e.message)
                      : JSON.stringify(e),
              })),
            ),
          ].slice(-120),
        );
      }
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      if (token === generation.current) setBusy(false);
    }
  };
  const initialize = async (
    reset = false,
    matched = false,
    scenario?: string,
  ) => {
    if (inFlight.current) return;
    const plannedSeed = scenario ? protocolSeed : seed;
    if (
      !Number.isInteger(plannedSeed) ||
      plannedSeed < 0 ||
      plannedSeed > 4294967295
    ) {
      setError("Seed must be an integer from 0 to 4294967295.");
      return;
    }
    if (matched && !checkpointCompatible) {
      setError(
        "Select a checkpoint from the current model before creating a matched episode.",
      );
      return;
    }
    const token = ++generation.current;
    setRunning(false);
    setReplayRunning(false);
    runningRef.current = false;
    setBusy(true);
    setError("");
    inFlight.current = true;
    try {
      if (!reset && current.current?.status === "active")
        await request(
          `/procedure/sessions/${current.current.id}?expected_revision=${current.current.revision}`,
          { method: "DELETE" },
        );
      const data = await request<Episode>(
        matched && training
          ? `/learning/runs/${training.id}/session`
          : scenario
            ? `/procedure/scenarios/${scenario}/session`
            : reset && current.current
              ? `/procedure/sessions/${current.current.id}/reset`
              : "/procedure/sessions",
        {
          method: "POST",
          body: JSON.stringify(
            matched
              ? { seed }
              : scenario
                ? { seed: protocolSeed, config: {} }
                : reset
                  ? { seed, expected_revision: current.current?.revision }
                  : { seed, config: {} },
          ),
        },
      );
      if (token === generation.current) {
        const previous = current.current?.id;
        if (previous)
          setSessions((old) =>
            old.map((s) =>
              s.id === previous ? { ...s, status: "closed" } : s,
            ),
          );
        accept(data, true);
        if (matched) setController("trained");
      }
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  };
  const step = useCallback(
    async (action: number[], policy = false) => {
      if (policy && controller === "trained" && !trainedReady) {
        setRunning(false);
        runningRef.current = false;
        return;
      }
      if (
        legacy ||
        replayIndex !== null ||
        inFlight.current ||
        !current.current ||
        current.current.terminated ||
        current.current.truncated ||
        current.current.status !== "active"
      )
        return;
      const before = current.current,
        token = generation.current;
      inFlight.current = true;
      setBusy(true);
      setError("");
      try {
        const body = policy
          ? controller === "trained"
            ? {
                training_run_id: training?.id,
                expected_revision: before.revision,
              }
            : { expected_revision: before.revision }
          : { action, expected_revision: before.revision };
        const data = await request<Episode>(
          `/procedure/sessions/${before.id}/${policy ? (controller === "trained" ? "policy-step" : controller === "protocol" ? "protocol-step" : "scripted-step") : "step"}`,
          { method: "POST", body: JSON.stringify(body) },
        );
        if (token === generation.current) {
          accept(data);
          if (
            data.policy_context &&
            data.policy_context.training_run_id === selectedTrainingId.current
          )
            setPolicyContext(data.policy_context);
        }
      } catch (e) {
        setRunning(false);
        runningRef.current = false;
        setError(errorMessage(e));
        if (token === generation.current) {
          try {
            const latest = await request<Episode>(
              `/procedure/sessions/${before.id}`,
            );
            if (token === generation.current) accept(latest);
          } catch {
            /* original error retained */
          }
        }
      } finally {
        inFlight.current = false;
        setBusy(false);
      }
    },
    [accept, training?.id, controller, legacy, replayIndex, trainedReady],
  );
  const manual = useCallback(
    (dx = 0, dy = 0, grip = 0, cautery = 0) => {
      if (runningRef.current) return;
      const action = [0, 0, grip, 0, 0, cautery, 0];
      if (tool === "grasper") {
        action[0] = dx;
        action[1] = dy;
      } else {
        action[3] = dx;
        action[4] = dy;
      }
      void step(action);
    },
    [step, tool],
  );
  useEffect(() => {
    if (!running) return;
    runningRef.current = true;
    let cancelled = false;
    const tick = async () => {
      if (cancelled || !runningRef.current) return;
      await step([0, 0, 0, 0, 0, 0, 0], true);
      if (!cancelled && runningRef.current)
        timer = window.setTimeout(tick, 160);
    };
    let timer = window.setTimeout(tick, 50);
    return () => {
      cancelled = true;
      runningRef.current = false;
      window.clearTimeout(timer);
    };
  }, [running, step]);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setRunning(false);
        setReplayRunning(false);
        runningRef.current = false;
        return;
      }
      const target = e.target instanceof Element ? e.target : null;
      if (
        !target ||
        target.closest("input,select,textarea,[contenteditable=true]") ||
        e.ctrlKey ||
        e.altKey ||
        e.metaKey
      )
        return;
      if (e.key === "g" || e.key === "G") {
        setTool("grasper");
        (
          document.querySelector(
            ".three-tissue-viewport canvas, .tissue-scene",
          ) as HTMLElement | null
        )?.focus();
        e.preventDefault();
        return;
      }
      if (e.key === "c" || e.key === "C") {
        setTool("cautery");
        (
          document.querySelector(
            ".three-tissue-viewport canvas, .tissue-scene",
          ) as HTMLElement | null
        )?.focus();
        e.preventDefault();
        return;
      }
      if (
        !target.matches("canvas,.tissue-scene") &&
        !target.closest(".tool-dock")
      )
        return;
      const moves: Record<string, number[]> = {
        ArrowLeft: [-distance, 0],
        ArrowRight: [distance, 0],
        ArrowUp: [0, distance],
        ArrowDown: [0, -distance],
      };
      if (moves[e.key]) {
        e.preventDefault();
        manual(...moves[e.key]);
      } else if (e.code === "Space") {
        if (target.closest("button")) return;
        e.preventDefault();
        manual(
          0,
          0,
          tool === "grasper"
            ? current.current?.state.tools.grasper.grip
              ? -0.8
              : gripTarget
            : 0,
          tool === "cautery" ? power : 0,
        );
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [manual, distance, tool, power, gripTarget]);
  useEffect(() => {
    if (!training || !["queued", "running"].includes(training.status)) return;
    let live = true;
    const timer = window.setInterval(async () => {
      try {
        const data = await request<TrainingRun>(
          `/learning/runs/${training.id}`,
        );
        if (live) {
          setTraining(data);
          setTrainings((old) => [data, ...old.filter((r) => r.id !== data.id)]);
          if (data.status === "failed")
            setError(data.error || "Policy training failed.");
        }
      } catch (e) {
        if (live) setError(errorMessage(e));
      }
    }, 1000);
    return () => {
      live = false;
      window.clearInterval(timer);
    };
  }, [training?.id, training?.status]);
  const train = async () => {
    setTrainingBusy(true);
    setError("");
    try {
      const data = await request<TrainingRun>("/learning/runs", {
        method: "POST",
        body: JSON.stringify({
          training_seed: seed,
          algorithm,
          episodes,
          eval_episodes: 20,
          max_steps: 160,
          learning_rate: 0.0007,
          gamma: 0.98,
        }),
      });
      setTraining(data);
      setTrainings((old) => [data, ...old]);
      setConsoleTab("learning");
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setTrainingBusy(false);
    }
  };
  const disabled =
    busy ||
    legacy ||
    replayIndex !== null ||
    !session ||
    session.terminated ||
    session.truncated ||
    session.status !== "active" ||
    running;
  const nodes = state?.tissue.nodes ?? [];
  return (
    <div className="procedure-lab">
      <div className="procedure-toolbar">
        <button
          className="lab-primary"
          disabled={busy}
          onClick={() => void initialize()}
        >
          <Plus size={14} />
          New episode
        </button>
        <button
          disabled={busy || !session}
          onClick={() => void initialize(true)}
        >
          <RotateCcw size={13} />
          Reset
        </button>
        <span className="toolbar-separator" />
        <button
          disabled={
            disabled ||
            (controller === "trained" && !trainedReady) ||
            (controller === "protocol" &&
              (!session?.protocol || session.protocol.completed))
          }
          onClick={() =>
            controller === "manual"
              ? manual()
              : void step([0, 0, 0, 0, 0, 0, 0], true)
          }
        >
          <SkipForward size={13} />
          Step
        </button>
        <button
          className={running ? "is-running" : ""}
          disabled={
            !session ||
            legacy ||
            replayIndex !== null ||
            session.status !== "active" ||
            session.terminated ||
            session.truncated ||
            controller === "manual" ||
            (controller === "protocol" &&
              (!session.protocol || session.protocol.completed)) ||
            (controller === "trained" && !trainedReady) ||
            (busy && !running)
          }
          onClick={() => {
            setRunning(!running);
            runningRef.current = !running;
          }}
        >
          {running ? <Pause size={13} /> : <Play size={13} />}{" "}
          {running
            ? "Pause"
            : controller === "protocol"
              ? "Run protocol"
              : "Run policy"}
        </button>
        <span className="toolbar-separator" />
        <label>
          Seed
          <input
            aria-label="Procedure random seed"
            title="Demonstration default: seed 2. Change it to explore other synthetic anatomy."
            type="number"
            min="0"
            max="4294967295"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
          />
        </label>
        <label className="toolbar-controller">
          <span className="sr-only">Playback controller</span>
          <select
            aria-label="Playback controller"
            value={controller}
            onChange={(e) => {
              setRunning(false);
              runningRef.current = false;
              setController(e.target.value as Controller);
              if (e.target.value === "protocol") setInspector("protocol");
            }}
          >
            <option value="manual">Manual commands</option>
            <option value="baseline">Scripted baseline</option>
            <option value="trained">Trained policy</option>
            <option value="protocol">Response protocol</option>
          </select>
        </label>
        <span className="procedure-clock">
          t = {fmt(state?.time_s, 2)} s · step {state?.step_count ?? 0}
        </span>
        {busy && <LoaderCircle size={13} className="spin" />}
        {session && (
          <a href={`/api/procedure/sessions/${session.id}/export`} download>
            <ArrowDownToLine size={13} />
            Episode JSON
          </a>
        )}
      </div>
      {error && (
        <div className="lab-error" role="alert">
          <TriangleAlert size={14} />
          {error}
          <button onClick={() => setError("")}>Dismiss</button>
        </div>
      )}
      {legacy && (
        <div className="archived-episode-notice" role="status">
          <span>
            Archived model revision · {session?.engine_version}. This episode is
            read-only because its model version or mechanics fingerprint differs
            from the running engine. Reset creates a new episode with current
            mechanics; the original record and export remain available.
          </span>
        </div>
      )}
      {controller === "trained" &&
        session &&
        policyContext?.out_of_distribution && (
          <div className="policy-context-notice" role="status">
            <TriangleAlert size={13} />
            <span>
              Policy evaluation outside training conditions:{" "}
              {policyContext.reasons.join("; ")}
            </span>
            <button
              disabled={busy || !checkpointCompatible}
              onClick={() => void initialize(false, true)}
            >
              Match checkpoint conditions
            </button>
          </div>
        )}
      <div className="procedure-layout">
        <section className="procedure-viewport">
          <div className="viewport-header">
            <span>
              Spatial tissue state{" "}
              <button
                className={viewMode === "3d" ? "selected" : ""}
                onClick={() => setViewMode("3d")}
              >
                3D surface
              </button>
              <button
                className={viewMode === "xy" ? "selected" : ""}
                onClick={() => setViewMode("xy")}
              >
                XY nodes
              </button>
            </span>
            <span>ILLUSTRATIVE GEOMETRY · STATEFUL SURROGATE</span>
          </div>
          <div className="viewport-body">
            {state ? (
              viewMode === "3d" ? (
                <Suspense
                  fallback={
                    <div className="procedure-initializing">
                      Loading local 3D renderer…
                    </div>
                  }
                >
                  <ThreeViewport
                    state={state}
                    onUnavailable={() => setViewMode("xy")}
                    onTarget={(x, y) => {
                      if (disabled) return;
                      const p = state.tools[tool].position_mm;
                      manual(
                        Math.max(-1, Math.min(1, (x - p[0]) / 5)),
                        Math.max(-1, Math.min(1, (y - p[1]) / 5)),
                      );
                    }}
                  />
                </Suspense>
              ) : (
                <TissueViewport
                  state={state}
                  tool={tool}
                  onTarget={(x, y) => {
                    if (disabled) return;
                    const p = state.tools[tool].position_mm;
                    manual(
                      Math.max(-1, Math.min(1, (x - p[0]) / 5)),
                      Math.max(-1, Math.min(1, (y - p[1]) / 5)),
                    );
                  }}
                />
              )
            ) : (
              <div className="procedure-initializing">
                <Activity size={33} />
                <h1>No active episode</h1>
                <p>
                  Initialize a seeded environment to manipulate tissue and
                  record persistent mechanical, thermal, and physiological
                  responses.
                </p>
                <p className="demonstration-seed-note">
                  Seed 2 is an illustrative demonstration case. Change the seed
                  to explore other synthetic patients.
                </p>
                <button
                  className="lab-start-button"
                  onClick={() => void initialize()}
                  disabled={busy}
                >
                  <Plus size={14} />
                  Initialize episode
                </button>
              </div>
            )}
            <div className="viewport-caption">
              <span>Millimetre coordinates · geometry illustrative</span>
              <span>
                {session
                  ? `${replayIndex === null ? session.status : "Recording inspection"} · revision ${inspectedEpisode?.revision}`
                  : "Session not initialized"}
              </span>
            </div>
            {state && (
              <div
                className="viewport-live-readings"
                aria-label="Synthetic telemetry for displayed state"
                role="group"
              >
                <span>
                  Force <strong>{fmt(state.tools.grasper.force_n)} N*</strong>
                </span>
                <span>
                  Peak injury{" "}
                  <strong>
                    {fmt(
                      Math.max(
                        ...state.tissue.nodes.map((node) => node.damage),
                      ),
                      3,
                    )}
                  </strong>
                </span>
                <span>
                  HR{" "}
                  <strong>{fmt(state.physiology.heart_rate_bpm, 1)} bpm</strong>
                </span>
                <span>
                  MAP{" "}
                  <strong>
                    {fmt(state.physiology.mean_arterial_pressure_mmhg, 1)} mmHg*
                  </strong>
                </span>
              </div>
            )}
          </div>
          <div className="tool-dock">
            <div
              role="group"
              className="tool-select"
              aria-label="Active instrument"
            >
              <button
                className={tool === "grasper" ? "selected" : ""}
                aria-pressed={tool === "grasper"}
                onClick={() => setTool("grasper")}
              >
                <Grip size={15} />
                Grasper <kbd>G</kbd>
              </button>
              <button
                className={tool === "cautery" ? "selected" : ""}
                aria-pressed={tool === "cautery"}
                onClick={() => setTool("cautery")}
              >
                <Flame size={15} />
                Cautery <kbd>C</kbd>
              </button>
            </div>
            <div className="move-pad">
              <button
                aria-label="Move instrument left"
                disabled={disabled}
                onClick={() => manual(-distance, 0)}
              >
                <ArrowLeft size={15} />
              </button>
              <button
                aria-label="Move instrument up"
                disabled={disabled}
                onClick={() => manual(0, distance)}
              >
                <ArrowUp size={15} />
              </button>
              <button
                aria-label="Move instrument down"
                disabled={disabled}
                onClick={() => manual(0, -distance)}
              >
                <ArrowDown size={15} />
              </button>
              <button
                aria-label="Move instrument right"
                disabled={disabled}
                onClick={() => manual(distance, 0)}
              >
                <ArrowRight size={15} />
              </button>
            </div>
            <button
              disabled={disabled}
              onClick={() =>
                tool === "grasper"
                  ? manual(0, 0, state?.tools.grasper.grip ? -0.8 : gripTarget)
                  : manual(0, 0, 0, power)
              }
            >
              {tool === "grasper" ? <Hand size={14} /> : <Flame size={14} />}{" "}
              {tool === "grasper"
                ? state?.tools.grasper.grip
                  ? "Release grasp"
                  : "Close grasper"
                : "Cautery pulse"}
              <kbd>Space</kbd>
            </button>
            <button
              className="handoff-button"
              disabled={disabled}
              onClick={() => void step([0, 0, 0, 0, 0, 0, 1])}
            >
              Handoff
            </button>
          </div>
        </section>
        <aside className="procedure-properties">
          <div className="inspector-tabs">
            <button
              className={inspector === "tissue" ? "selected" : ""}
              onClick={() => setInspector("tissue")}
            >
              Tissue / tools
            </button>
            <button
              className={inspector === "protocol" ? "selected" : ""}
              onClick={() => setInspector("protocol")}
              aria-label="Response protocols"
            >
              Protocols
            </button>
            <button
              className={inspector === "policy" ? "selected" : ""}
              onClick={() => setInspector("policy")}
            >
              Policy learning
            </button>
          </div>
          {inspector === "tissue" ? (
            <>
              <h2>Manual tool settings</h2>
              <div className="inspector-group">
                <label>
                  Motion command <strong>{fmt(distance)} normalized</strong>
                  <input
                    aria-label="Instrument motion magnitude"
                    type="range"
                    min=".1"
                    max="1"
                    step=".05"
                    value={distance}
                    onChange={(e) => setDistance(Number(e.target.value))}
                  />
                </label>
                <label>
                  Grasper closure target{" "}
                  <strong>{fmt(gripTarget)} normalized</strong>
                  <input
                    aria-label="Grasper closure target"
                    type="range"
                    min=".1"
                    max="1"
                    step=".05"
                    value={gripTarget}
                    onChange={(e) => setGripTarget(Number(e.target.value))}
                  />
                </label>
                <label>
                  Cautery power <strong>{fmt(power)} normalized</strong>
                  <input
                    aria-label="Cautery power"
                    type="range"
                    min="0"
                    max="1"
                    step=".05"
                    value={power}
                    onChange={(e) => setPower(Number(e.target.value))}
                  />
                </label>
                <p>
                  Press G or C to select a tool and focus the viewport. Arrow
                  keys then move it; click the viewport for one bounded step
                  toward a point. Escape pauses controller or recording
                  playback.
                </p>
                <p>
                  These sliders apply to manual inputs. Automated controllers
                  use their recorded tool commands.
                </p>
              </div>
              <details className="tissue-node-details">
                <summary>
                  Episode parameters and limits
                  <ChevronDown size={12} />
                </summary>
                <dl>
                  <div>
                    <dt>Model version</dt>
                    <dd>{session?.engine_version ?? "—"}</dd>
                  </div>
                  <div>
                    <dt>Mechanics fingerprint</dt>
                    <dd
                      title={
                        typeof episodeFingerprint === "string"
                          ? episodeFingerprint
                          : undefined
                      }
                    >
                      {typeof episodeFingerprint === "string"
                        ? episodeFingerprint.slice(0, 16)
                        : "not recorded"}
                    </dd>
                  </div>
                  {Object.entries(state?.patient ?? {}).map(([k, v]) => (
                    <div key={k}>
                      <dt>{k.replaceAll("_", " ")}</dt>
                      <dd>{fmt(v, 3)}</dd>
                    </div>
                  ))}
                  {Object.entries(state?.limits ?? {}).map(([k, v]) => (
                    <div key={k}>
                      <dt>{k.replaceAll("_", " ")}</dt>
                      <dd>{typeof v === "number" ? fmt(v, 3) : String(v)}</dd>
                    </div>
                  ))}
                </dl>
              </details>
              <h2>Mechanical and tissue state</h2>
              <dl>
                <dt>Grasper force</dt>
                <dd>{fmt(state?.tools.grasper.force_n)} N*</dd>
                <dt>Grasp closure</dt>
                <dd>{fmt(state?.tools.grasper.grip)}</dd>
                <dt>Contact node</dt>
                <dd>{state?.tools.grasper.contact_node_id ?? "none"}</dd>
                <dt>Exposure proxy</dt>
                <dd>{fmt(state ? state.exposure * 100 : undefined, 1)} %</dd>
                <dt>Peak damage</dt>
                <dd>
                  {fmt(
                    state ? Math.max(...nodes.map((n) => n.damage)) : undefined,
                    3,
                  )}
                </dd>
                <dt>Peak temperature</dt>
                <dd>
                  {fmt(
                    state
                      ? Math.max(...nodes.map((n) => n.temperature_c))
                      : undefined,
                    1,
                  )}{" "}
                  °C
                </dd>
                <dt>Bleeding rate</dt>
                <dd>
                  {fmt(state?.physiology.bleeding_rate_ml_per_s, 3)} mL/s*
                </dd>
                <dt>Bile leak rate</dt>
                <dd>
                  {fmt(state?.physiology.bile_leak_rate_ml_per_s, 3)} mL/s*
                </dd>
              </dl>
              <p className="model-unit-note">
                * Synthetic model units and parameters. Force and physiology are
                not clinically calibrated.
              </p>
              <h2>Synthetic physiological response</h2>
              <div className="vital-readings">
                <span>
                  Heart rate
                  <strong>
                    {fmt(state?.physiology.heart_rate_bpm, 1)}
                    <small>bpm</small>
                  </strong>
                </span>
                <span>
                  Mean arterial pressure
                  <strong>
                    {fmt(state?.physiology.mean_arterial_pressure_mmhg, 1)}
                    <small>mmHg*</small>
                  </strong>
                </span>
              </div>
              <MiniTrace
                data={trace}
                field="heart"
                color="#a54e55"
                label="Synthetic heart rate"
              />
              <MiniTrace
                data={trace}
                field="map"
                color="#457caa"
                label="Synthetic mean arterial pressure"
              />
              <dl>
                <dt>Blood loss</dt>
                <dd>{fmt(state?.physiology.blood_loss_ml, 3)} mL*</dd>
                <dt>Bile leak</dt>
                <dd>{fmt(state?.physiology.bile_leak_ml, 3)} mL*</dd>
              </dl>
              <details className="tissue-node-details">
                <summary>
                  Inspect tissue nodes
                  <ChevronDown size={12} />
                </summary>
                <div
                  className="node-table"
                  tabIndex={0}
                  role="region"
                  aria-label="Tissue node state"
                >
                  <table>
                    <thead>
                      <tr>
                        <th>Node</th>
                        <th>Damage</th>
                        <th>°C</th>
                        <th>Strain</th>
                      </tr>
                    </thead>
                    <tbody>
                      {nodes.map((n) => (
                        <tr key={n.id}>
                          <td>{n.id}</td>
                          <td>{fmt(n.damage, 3)}</td>
                          <td>{fmt(n.temperature_c, 1)}</td>
                          <td>{fmt(n.strain, 3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            </>
          ) : inspector === "protocol" ? (
            <>
              <h2>Controlled response protocols</h2>
              <div className="inspector-group response-protocol-controls">
                <p>
                  Scripted interventions issue real bounded tool actions through
                  the tissue engine. These are controlled synthetic experiments,
                  separate from learned policies.
                </p>
                {scenarioError && (
                  <p className="protocol-error" role="status">
                    Protocol catalog unavailable: {scenarioError}{" "}
                    <button
                      onClick={() => setScenarioRefresh((value) => value + 1)}
                    >
                      Reload protocols
                    </button>
                  </p>
                )}
                <label>
                  Response protocol
                  <select
                    aria-label="Response protocol"
                    value={scenarioId}
                    onChange={(e) => {
                      setScenarioId(e.target.value);
                      const chosen = scenarios.find(
                        (item) => item.id === e.target.value,
                      );
                      if (chosen) setProtocolSeed(chosen.seed);
                    }}
                  >
                    {!scenarios.length && (
                      <option value="">Loading protocol catalog…</option>
                    )}
                    {scenarios.map((scenario) => (
                      <option value={scenario.id} key={scenario.id}>
                        {scenario.title}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Protocol seed
                  <input
                    aria-label="Protocol seed"
                    type="number"
                    min="0"
                    max="4294967295"
                    value={protocolSeed}
                    onChange={(e) => setProtocolSeed(Number(e.target.value))}
                  />
                </label>
                {selectedScenario && (
                  <>
                    <p className="protocol-description">
                      {selectedScenario.description}
                    </p>
                    <button
                      className="lab-train-button"
                      disabled={busy || running}
                      onClick={() =>
                        void initialize(false, false, selectedScenario.id)
                      }
                    >
                      <Plus size={13} />
                      New protocol episode
                    </button>
                    <details className="protocol-parameters">
                      <summary>
                        Protocol parameters <ChevronDown size={12} />
                      </summary>
                      <p>{selectedScenario.scope}</p>
                      <dl>
                        {Object.entries(selectedScenario.parameters).map(
                          ([key, value]) => (
                            <div key={key}>
                              <dt>{key.replaceAll("_", " ")}</dt>
                              <dd>
                                {typeof value === "number"
                                  ? fmt(value, 3)
                                  : typeof value === "object"
                                    ? JSON.stringify(value)
                                    : String(value)}
                              </dd>
                            </div>
                          ),
                        )}
                      </dl>
                      <p>
                        Protocol version {selectedScenario.protocol_version}.
                        Episode settings and action provenance are retained in
                        its JSON export.
                      </p>
                    </details>
                  </>
                )}
              </div>
              {session?.protocol && (
                <>
                  <h2>Recorded intervention state</h2>
                  <dl className="protocol-state">
                    <dt>Scenario</dt>
                    <dd>
                      {session.protocol.metadata?.title ??
                        session.protocol.scenario_id}
                    </dd>
                    <dt>Latest protocol status</dt>
                    <dd>{session.protocol.status}</dd>
                    <dt>
                      {inspectedEpisode?.source
                        ? "Displayed action"
                        : "Controller phase"}
                    </dt>
                    <dd>
                      {inspectedEpisode?.source?.phase ??
                        inspectedEpisode?.source?.type ??
                        (replayIndex === null
                          ? session.protocol.phase
                          : "not recorded")}
                    </dd>
                    <dt>Displayed revision</dt>
                    <dd>{inspectedEpisode?.revision ?? session.revision}</dd>
                    <dt>Completion reason</dt>
                    <dd>
                      {session.protocol.completion_reason ?? "in progress"}
                    </dd>
                    <dt>Episode outcome</dt>
                    <dd>
                      {session.state.termination_reason ?? session.status}
                    </dd>
                  </dl>
                  <p className="model-unit-note">
                    Use Step or Run protocol to issue recorded actions. A
                    completed intervention can leave the episode active.
                    Completion does not imply modeled success. Manual
                    intervention interrupts the protocol; Reset starts a fresh
                    copy.
                  </p>
                  <h2>Observed synthetic response</h2>
                  <dl>
                    <dt>Peak injury</dt>
                    <dd>
                      {fmt(
                        state
                          ? Math.max(...nodes.map((node) => node.damage))
                          : undefined,
                        3,
                      )}
                    </dd>
                    <dt>Blood loss</dt>
                    <dd>{fmt(state?.physiology.blood_loss_ml, 3)} mL*</dd>
                    <dt>Bile leak</dt>
                    <dd>{fmt(state?.physiology.bile_leak_ml, 3)} mL*</dd>
                    <dt>Displayed return</dt>
                    <dd>{fmt(inspectedEpisode?.episode_reward, 3)}</dd>
                  </dl>
                  <MiniTrace
                    data={trace}
                    field="damage"
                    color="#9b594a"
                    label="Observed peak tissue injury"
                  />
                </>
              )}
            </>
          ) : (
            <>
              <h2>Train a discrete policy</h2>
              <div className="inspector-group">
                <p>
                  The learning service fits a policy from environment
                  transitions. Training and held-out evaluation remain
                  synthetic. Playback uses fixed checkpoint weights; manual
                  episodes are not automatically used for training.
                </p>
                <label>
                  Learning algorithm
                  <select
                    aria-label="Learning algorithm"
                    value={algorithm}
                    onChange={(e) =>
                      setAlgorithm(e.target.value as "dqn" | "tabular_q")
                    }
                  >
                    <option value="dqn">Double DQN · neural network</option>
                    <option value="tabular_q">Double Q · binned state</option>
                  </select>
                </label>
                <label>
                  Training episodes
                  <input
                    aria-label="Training episodes"
                    type="number"
                    min="20"
                    max="3000"
                    value={episodes}
                    onChange={(e) => setEpisodes(Number(e.target.value))}
                  />
                </label>
                {!engineSpec && (
                  <div
                    className="policy-load-state"
                    role="status"
                    aria-label="Engine specification status"
                  >
                    <strong>
                      {specError
                        ? "Engine specification unavailable"
                        : "Loading engine specification…"}
                    </strong>
                    {specError && (
                      <>
                        <p>{specError}</p>
                        <button
                          onClick={() => {
                            setRunning(false);
                            runningRef.current = false;
                            setSpecError("");
                            setSpecRetry((value) => value + 1);
                          }}
                        >
                          Retry engine specification
                        </button>
                      </>
                    )}
                  </div>
                )}
                {training?.status === "completed" && !learningResult && (
                  <div
                    className="policy-load-state"
                    role="status"
                    aria-label="Checkpoint results status"
                  >
                    <strong>
                      {learningError
                        ? "Checkpoint results unavailable"
                        : "Loading checkpoint results…"}
                    </strong>
                    {learningError && (
                      <>
                        <p>{learningError}</p>
                        <button
                          onClick={() => {
                            setRunning(false);
                            runningRef.current = false;
                            setLearningFailure(null);
                            setLearningRetry((value) => value + 1);
                          }}
                        >
                          Retry checkpoint results
                        </button>
                      </>
                    )}
                  </div>
                )}
                <button
                  className="lab-train-button"
                  disabled={
                    trainingBusy ||
                    training?.status === "running" ||
                    training?.status === "queued"
                  }
                  onClick={() => void train()}
                >
                  {trainingBusy || training?.status === "running" ? (
                    <LoaderCircle size={13} className="spin" />
                  ) : (
                    <Activity size={13} />
                  )}
                  Train policy
                </button>
                <label>
                  Policy checkpoint
                  <select
                    aria-label="Policy checkpoint"
                    value={
                      controller === "baseline"
                        ? "baseline"
                        : (training?.id ?? "baseline")
                    }
                    onChange={(e) => {
                      setRunning(false);
                      runningRef.current = false;
                      if (e.target.value === "baseline")
                        setController("baseline");
                      else {
                        setController("trained");
                        setTraining(
                          trainings.find((r) => r.id === e.target.value) ??
                            null,
                        );
                      }
                    }}
                  >
                    <option value="baseline">
                      Scripted feedback baseline (not RL)
                    </option>
                    {trainings.map((t) => (
                      <option value={t.id} key={t.id}>
                        {typeof t.summary?.label === "string"
                          ? t.summary.label
                          : t.id.slice(0, 8)}{" "}
                        · {t.status}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  className="lab-train-button"
                  disabled={
                    busy ||
                    training?.status !== "completed" ||
                    !checkpointCompatible
                  }
                  onClick={() => void initialize(false, true)}
                >
                  <Plus size={13} />
                  New matched episode
                </button>
                {checkpointArchived && (
                  <div
                    className="policy-context archived-checkpoint"
                    role="status"
                  >
                    <strong>Archived model checkpoint</strong>
                    <p>
                      This checkpoint’s model version or mechanics fingerprint
                      differs from the running engine. Its recorded evaluation
                      remains available; train a current policy for playback.
                    </p>
                  </div>
                )}
                <p>
                  Uses the selected checkpoint’s environment parameters and
                  training horizon. A new seed is an evaluation sample, not
                  evidence of transfer to clinical conditions.
                </p>
                {policyContext && (
                  <div
                    className={`policy-context ${policyContext.out_of_distribution ? "outside" : ""}`}
                    role="status"
                  >
                    <strong>
                      {!policyContext.comparison_available
                        ? "Context comparison unavailable"
                        : policyContext.out_of_distribution
                          ? "Outside training conditions"
                          : "Environment parameters match checkpoint"}
                    </strong>
                    <p>
                      {policyContext.reasons.length
                        ? policyContext.reasons.join("; ")
                        : "No environment parameter differences were found."}
                    </p>
                    {!!policyContext.differing_parameters?.length && (
                      <dl>
                        {policyContext.differing_parameters.map(
                          (difference) => (
                            <div key={difference.parameter}>
                              <dt>
                                {difference.parameter.replaceAll("_", " ")}
                              </dt>
                              <dd>
                                {String(difference.training)} →{" "}
                                {String(difference.session)}
                              </dd>
                            </div>
                          ),
                        )}
                      </dl>
                    )}
                  </div>
                )}
                {contextError && (
                  <p className="policy-context-error">
                    Policy context could not be checked: {contextError}
                    <button
                      onClick={() => {
                        setRunning(false);
                        runningRef.current = false;
                        setContextRetry((value) => value + 1);
                      }}
                    >
                      Retry policy context
                    </button>
                  </p>
                )}
                {training && (
                  <>
                    <div className="learning-progress">
                      <span style={{ width: `${training.progress * 100}%` }} />
                    </div>
                    <span className="training-status">
                      {training.status} · {fmt(training.progress * 100, 0)}%
                    </span>
                    {training.summary && (
                      <dl>
                        {Object.entries(training.summary)
                          .filter(
                            ([, v]) =>
                              typeof v === "number" || typeof v === "string",
                          )
                          .slice(0, 10)
                          .map(([key, v]) => (
                            <div key={key}>
                              <dt>{key.replaceAll("_", " ")}</dt>
                              <dd>
                                {typeof v === "number" ? fmt(v, 3) : String(v)}
                              </dd>
                            </div>
                          ))}
                      </dl>
                    )}
                    <a
                      className="learning-export"
                      href={`/api/learning/runs/${training.id}/results`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Inspect training results
                    </a>
                    {learningResult && (
                      <>
                        {referenceProvenance && (
                          <p className="reference-provenance">
                            <strong>
                              Bundled reference · selected on validation
                              rollouts.
                            </strong>{" "}
                            {typeof referenceProvenance.scientific_note ===
                            "string"
                              ? referenceProvenance.scientific_note
                              : "Selection performance is not an untouched final test or evidence of clinical validity."}
                          </p>
                        )}
                        <p className="learning-method">
                          {learningResult.metadata?.training_method ??
                            learningResult.algorithm}
                        </p>
                        <LearningCurve points={learningResult.learning_curve} />
                        <button
                          className="lab-train-button"
                          onClick={() => setConsoleTab("learning")}
                        >
                          Show held-out evaluation
                        </button>
                      </>
                    )}
                  </>
                )}
              </div>
              <h2>Reward feedback</h2>
              <div className="reward-total">
                <span>Current transition</span>
                <strong>{fmt(inspectedEpisode?.reward, 3)}</strong>
                <span>Episode return</span>
                <strong>{fmt(inspectedEpisode?.episode_reward, 3)}</strong>
              </div>
              <div className="reward-components">
                {Object.entries(inspectedEpisode?.reward_components ?? {}).map(
                  ([key, value]) => (
                    <div key={key}>
                      <span>{key.replaceAll("_", " ")}</span>
                      <strong className={value < 0 ? "negative" : ""}>
                        {value > 0 ? "+" : ""}
                        {fmt(value, 4)}
                      </strong>
                    </div>
                  ),
                )}
              </div>
              <MiniTrace
                data={trace}
                field="reward"
                color="#5c7b9c"
                label="Observed transition reward"
              />
            </>
          )}
        </aside>
      </div>
      {recordedFrames.length > 1 && (
        <div
          className={`replay-transport ${replayIndex !== null ? "inspecting" : ""}`}
          role="group"
          aria-label="Recorded episode inspection"
        >
          <button
            disabled={busy || running || replayIndex === 0}
            onClick={() => {
              setReplayRunning(false);
              setReplayIndex(
                Math.max(0, (replayIndex ?? recordedFrames.length - 1) - 1),
              );
            }}
            aria-label="Previous recorded transition"
          >
            <ArrowLeft size={12} />
          </button>
          <button
            disabled={busy || running}
            onClick={() => {
              if (replayRunning) setReplayRunning(false);
              else {
                setReplayIndex((index) =>
                  index === null || index >= recordedFrames.length - 1
                    ? 0
                    : index,
                );
                setReplayRunning(true);
              }
            }}
          >
            {replayRunning ? <Pause size={12} /> : <Play size={12} />}{" "}
            {replayRunning ? "Pause recording" : "Play recording"}
          </button>
          <button
            disabled={
              busy ||
              running ||
              replayIndex === null ||
              replayIndex === recordedFrames.length - 1
            }
            onClick={() => {
              setReplayRunning(false);
              setReplayIndex(
                Math.min(
                  recordedFrames.length - 1,
                  (replayIndex ?? recordedFrames.length - 1) + 1,
                ),
              );
            }}
            aria-label="Next recorded transition"
          >
            <ArrowRight size={12} />
          </button>
          <label className="sr-only" htmlFor="recorded-transition">
            Recorded transition
          </label>
          <input
            id="recorded-transition"
            type="range"
            min="0"
            max={recordedFrames.length - 1}
            step="1"
            value={replayIndex ?? recordedFrames.length - 1}
            disabled={busy || running}
            onChange={(e) => {
              setReplayRunning(false);
              setReplayIndex(Number(e.target.value));
            }}
            aria-valuetext={`Recorded revision ${inspectedEpisode?.revision ?? 0}, ${fmt(state?.time_s)} seconds`}
          />
          <span>
            Revision {inspectedEpisode?.revision} · {fmt(state?.time_s)} s
          </span>
          <button
            disabled={replayIndex === null}
            onClick={() => {
              setReplayRunning(false);
              setReplayIndex(null);
            }}
          >
            Return to latest
          </button>
          <span className="replay-state">
            {replayIndex === null
              ? "Latest saved state"
              : "Read-only recorded frame"}
          </span>
        </div>
      )}
      <section className="procedure-console">
        <div className="console-heading">
          <h2>
            <Terminal size={13} />
            Episode diagnostics
          </h2>
          <div className="console-tabs">
            <button
              className={consoleTab === "events" ? "selected" : ""}
              onClick={() => setConsoleTab("events")}
            >
              Events
            </button>
            <button
              className={consoleTab === "rewards" ? "selected" : ""}
              onClick={() => setConsoleTab("rewards")}
            >
              Reward trace
            </button>
            <button
              className={consoleTab === "learning" ? "selected" : ""}
              onClick={() => setConsoleTab("learning")}
            >
              Policy evaluation
            </button>
          </div>
          <div>
            <label className="sr-only" htmlFor="procedure-history">
              Saved episode
            </label>
            <select
              id="procedure-history"
              disabled={busy || running}
              value={session?.id ?? ""}
              onChange={(e) => e.target.value && void load(e.target.value)}
            >
              <option value="">Saved episodes</option>
              {sessions.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.id.slice(0, 8)} · seed {s.seed} ·{" "}
                  {s.engine_compatible === false ? "archived" : s.status}
                </option>
              ))}
            </select>
            <span>
              {session?.state.termination_reason ||
                "Events reflect actual environment transitions"}
            </span>
          </div>
        </div>
        <div
          className={
            consoleTab === "events" ? "event-log" : "event-log hidden-console"
          }
          tabIndex={0}
          role="log"
          aria-label="Procedure event timeline"
        >
          {log.length ? (
            log.map((entry, i) => (
              <div key={i}>
                <span>{String(entry.step).padStart(4, "0")}</span>
                <p>{entry.text}</p>
              </div>
            ))
          ) : (
            <p>
              No episode loaded. Initialize an environment to record actions and
              tissue responses.
            </p>
          )}
        </div>
        {consoleTab === "rewards" && (
          <div className="console-reward-traces">
            <MiniTrace
              data={trace}
              field="reward"
              color="#9c4e47"
              label="Observed reward per transition"
            />
            <MiniTrace
              data={trace}
              field="total"
              color="#3e789f"
              label="Cumulative episode return"
            />
            <MiniTrace
              data={trace}
              field="damage"
              color="#ab6745"
              label="Peak material injury"
            />
          </div>
        )}
        {consoleTab === "learning" && (
          <div
            className="learning-evaluation"
            tabIndex={0}
            role="region"
            aria-label="Held-out policy evaluation"
          >
            {learningError ? (
              <p role="alert">
                Checkpoint results unavailable: {learningError}. Retry in the
                Policy learning inspector.
              </p>
            ) : !learningResult ? (
              <p>
                {training?.status === "completed"
                  ? "Loading checkpoint results…"
                  : training && ["queued", "running"].includes(training.status)
                    ? "Training and evaluating actual environment rollouts. Results appear after completion."
                    : "Select a completed policy checkpoint to inspect its held-out evaluation."}
              </p>
            ) : (
              <>
                <table>
                  <caption>
                    {referenceProvenance
                      ? "Training-disjoint validation rollouts · selected reference"
                      : "Held-out synthetic rollouts"}{" "}
                    · {training?.id.slice(0, 8)} · mean return with 95%
                    bootstrap interval
                  </caption>
                  <thead>
                    <tr>
                      <th>Controller</th>
                      <th>n</th>
                      <th>Mean return [95% CI]</th>
                      <th>Success [95% CI]</th>
                      <th>Severe harm [95% CI]</th>
                      <th>Handoff</th>
                      <th>Mean peak injury</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(
                      [
                        ["learned", "Learned policy"],
                        ["random", "Random actions"],
                        ["random_no_handoff", "Random without handoff"],
                        ["scripted", "Scripted feedback"],
                      ] as const
                    ).map(([key, label]) => {
                      const row = learningResult.evaluation[key];
                      return (
                        row && (
                          <tr key={key}>
                            <th scope="row">{label}</th>
                            <td>{row.n}</td>
                            <td>
                              {fmt(row.mean_return)}{" "}
                              {row.mean_return_ci &&
                                `[${fmt(row.mean_return_ci[0])}, ${fmt(row.mean_return_ci[1])}]`}
                            </td>
                            <td>
                              {fmt(row.success_rate * 100, 1)}%
                              {row.success_rate_ci &&
                                ` [${fmt(row.success_rate_ci[0] * 100, 1)}, ${fmt(row.success_rate_ci[1] * 100, 1)}]`}
                            </td>
                            <td>
                              {fmt(row.severe_harm_rate * 100, 1)}%
                              {row.severe_harm_rate_ci &&
                                ` [${fmt(row.severe_harm_rate_ci[0] * 100, 1)}, ${fmt(row.severe_harm_rate_ci[1] * 100, 1)}]`}
                            </td>
                            <td>{fmt(row.handoff_rate * 100, 1)}%</td>
                            <td>{fmt(row.mean_max_damage, 3)}</td>
                          </tr>
                        )
                      );
                    })}
                  </tbody>
                </table>
                <p>
                  All controllers use the same evaluation seeds. Binary
                  intervals use the Wilson method. Zero observed severe harm
                  does not establish safety; inspect harm, handoff, and success
                  separately.
                </p>
              </>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
function LearningCurve({
  points,
}: {
  points: LearningResult["learning_curve"];
}) {
  const values = points.filter((point) => Number.isFinite(point.return));
  if (!values.length) return null;
  const low = Math.min(...values.map((point) => point.return));
  const high = Math.max(...values.map((point) => point.return));
  const range = Math.max(1, high - low);
  const path = values
    .map(
      (point, index) =>
        `${index === 0 ? "M" : "L"}${8 + (index / Math.max(1, values.length - 1)) * 284},${76 - ((point.return - low) / range) * 62}`,
    )
    .join(" ");
  return (
    <div className="learning-curve">
      <span>Observed training episode return</span>
      <svg
        viewBox="0 0 300 86"
        role="img"
        aria-label={`Actual training returns over ${values.length} episodes; minimum ${fmt(low)}, maximum ${fmt(high)}`}
      >
        <line x1="8" x2="292" y1="76" y2="76" stroke="#c6d0db" />
        <path
          d={path}
          fill="none"
          stroke="#376b93"
          strokeWidth="1.5"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <div>
        <span>Episode 1</span>
        <span>
          {values.length} · return {fmt(values[values.length - 1].return)}
        </span>
      </div>
    </div>
  );
}
function MiniTrace({
  data,
  field,
  color,
  label,
}: {
  data: Trace[];
  field: keyof Trace;
  color: string;
  label: string;
}) {
  const values = data.map((d) => d[field]),
    low = Math.min(...values),
    high = Math.max(...values),
    range = Math.max(1, high - low);
  return (
    <div className="mini-trace">
      <span>{label}</span>
      <svg
        viewBox="0 0 260 54"
        role="img"
        aria-label={`${label}, ${data.length} recorded observations`}
      >
        <path d="M5 45H255M5 25H255M5 5H255" stroke="#e2e8ef" fill="none" />
        <polyline
          fill="none"
          stroke={color}
          strokeWidth="1.7"
          points={values
            .map(
              (v, i) =>
                `${5 + (values.length < 2 ? 0 : i / (values.length - 1)) * 250},${44 - ((v - low) / range) * 36}`,
            )
            .join(" ")}
        />
        {values.length === 1 && <circle cx="5" cy="44" r="2" fill={color} />}
      </svg>
      <span className="trace-range">
        {values.length
          ? `${fmt(low, 1)} – ${fmt(high, 1)} · ${values.length} samples`
          : "No recorded samples"}
      </span>
    </div>
  );
}
function TissueViewport({
  state: s,
  tool,
  onTarget,
}: {
  state: TissueState;
  tool: Tool;
  onTarget: (x: number, y: number) => void;
}) {
  const x = (v: number) => 450 + v * 9,
    y = (v: number) => 390 - v * 8,
    nodes = s.tissue.nodes,
    byId = new Map(nodes.map((n) => [n.id, n]));
  const g = s.tools.grasper,
    c = s.tools.cautery;
  const injury = (n: TissueNode) =>
    n.damage > 0.65
      ? "#5f2631"
      : n.temperature_c > 48
        ? "#e18c45"
        : n.damage > 0.15
          ? "#a8534b"
          : n.kind.includes("duct")
            ? "#b6b46a"
            : n.kind.includes("arter")
              ? "#b66a70"
              : "#9cb397";
  const onClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const svg = e.currentTarget,
      p = svg.createSVGPoint();
    p.x = e.clientX;
    p.y = e.clientY;
    const ctm = svg.getScreenCTM();
    if (ctm) {
      const local = p.matrixTransform(ctm.inverse());
      onTarget((local.x - 450) / 9, (390 - local.y) / 8);
    }
  };
  return (
    <svg
      className="tissue-scene"
      viewBox="0 0 900 680"
      role="img"
      aria-label="Illustrative XY tissue graph showing actual simulated node positions, instrument locations, temperature and damage"
      onClick={onClick}
      tabIndex={0}
    >
      <defs>
        <radialGradient id="cavity">
          <stop stopColor="#35443c" />
          <stop offset="1" stopColor="#111a21" />
        </radialGradient>
        <linearGradient id="liverFill" x1="0" y1="0" x2="1" y2="1">
          <stop stopColor="#8c5f55" />
          <stop offset="1" stopColor="#553d3b" />
        </linearGradient>
        <linearGradient id="gbFill" x1="0" y1="0" x2="1" y2="1">
          <stop stopColor="#899d80" />
          <stop offset="1" stopColor="#48634f" />
        </linearGradient>
        <radialGradient id="heat">
          <stop stopColor="#ffdf7a" stopOpacity=".8" />
          <stop offset=".4" stopColor="#ec7b36" stopOpacity=".5" />
          <stop offset="1" stopColor="#e04c29" stopOpacity="0" />
        </radialGradient>
        <pattern
          id="sceneGrid"
          width="45"
          height="40"
          patternUnits="userSpaceOnUse"
        >
          <path
            d="M45 0H0V40"
            stroke="#afc2bc"
            strokeOpacity=".055"
            fill="none"
          />
        </pattern>
      </defs>
      <rect width="900" height="680" fill="url(#cavity)" />
      <rect width="900" height="680" fill="url(#sceneGrid)" />
      <path
        d="M129 171C150 80 270 48 409 65C537 29 700 80 775 170C826 229 755 279 677 255C565 220 508 267 412 234C315 214 181 277 139 236Z"
        fill="url(#liverFill)"
        stroke="#a57f6f"
        strokeOpacity=".6"
      />
      <path
        d="M428 137C391 113 354 148 357 213C350 285 393 337 428 366C447 385 449 408 466 406C494 395 461 371 467 335C476 292 492 248 472 201C462 174 446 151 428 137Z"
        fill="url(#gbFill)"
        stroke="#a4b095"
        strokeWidth="2"
        opacity=".9"
      />
      <path
        d={`M${x(8)} ${y(-28)}L${x(8)} ${y(-12)}M${x(8)} ${y(-15)}L${x(2)} ${y(-8)}M${x(-2)} ${y(-16)}L${x(-4)} ${y(-10)}`}
        stroke="#b5ac71"
        strokeWidth="12"
        fill="none"
        strokeLinecap="round"
      />
      {s.tissue.edges.map(([a, b], i) => {
        const n = byId.get(a),
          m = byId.get(b);
        if (!n || !m) return null;
        return (
          <line
            key={i}
            x1={x(n.position_mm[0])}
            y1={y(n.position_mm[1])}
            x2={x(m.position_mm[0])}
            y2={y(m.position_mm[1])}
            stroke={Math.max(n.damage, m.damage) > 0.65 ? "#b84d45" : "#c2c7a8"}
            strokeWidth={Math.max(n.damage, m.damage) > 0.65 ? 1 : 2}
            strokeOpacity=".65"
            strokeDasharray={
              Math.max(n.damage, m.damage) > 0.7 ? "4 5" : undefined
            }
          />
        );
      })}
      {nodes.map((n) => (
        <g key={n.id}>
          {n.temperature_c > 40 && (
            <circle
              cx={x(n.position_mm[0])}
              cy={y(n.position_mm[1])}
              r={15 + Math.min(50, (n.temperature_c - 37) * 1.3)}
              fill="url(#heat)"
            />
          )}
          <circle
            cx={x(n.position_mm[0])}
            cy={y(n.position_mm[1])}
            r={n.kind.includes("duct") ? 7 : 6}
            fill={injury(n)}
            stroke="#d1d2b1"
            strokeWidth="1"
          >
            <title>
              {n.id}: damage {fmt(n.damage, 3)}, temperature{" "}
              {fmt(n.temperature_c, 1)}°C, strain {fmt(n.strain, 3)}
            </title>
          </circle>
          {n.damage > 0.65 && (
            <path
              d={`M${x(n.position_mm[0]) - 8} ${y(n.position_mm[1]) - 5}l5 7l6 -6l4 7`}
              stroke="#311921"
              strokeWidth="3"
              fill="none"
            />
          )}
          {s.physiology.bleeding_rate_ml_per_s > 0 &&
            n.damage > 0.1 &&
            n.kind.includes("arter") &&
            Array.from({ length: 10 }, (_, i) => (
              <circle
                key={i}
                cx={x(n.position_mm[0]) + Math.cos(i * 2.4 + s.time_s) * i * 3}
                cy={y(n.position_mm[1]) + i * 5 + Math.sin(s.time_s + i) * 6}
                r={3 + (i % 3)}
                fill="#a72f3d"
                opacity=".65"
              />
            ))}
          {s.physiology.bile_leak_rate_ml_per_s > 0 &&
            n.damage > 0.1 &&
            n.kind.includes("duct") &&
            Array.from({ length: 6 }, (_, i) => (
              <circle
                key={i}
                cx={x(n.position_mm[0]) + Math.sin(i + s.time_s) * i * 4}
                cy={y(n.position_mm[1]) + i * 8}
                r={3 + (i % 2)}
                fill="#b3b44c"
                opacity=".65"
              />
            ))}
        </g>
      ))}
      {c.power > 0 && (
        <circle
          cx={x(c.position_mm[0])}
          cy={y(c.position_mm[1])}
          r={30 + c.power * 35}
          fill="url(#heat)"
        />
      )}
      <g className={tool === "grasper" ? "selected-instrument" : ""}>
        <line
          x1="870"
          y1="660"
          x2={x(g.position_mm[0])}
          y2={y(g.position_mm[1])}
          stroke="#25313a"
          strokeWidth="17"
        />
        <line
          x1="870"
          y1="660"
          x2={x(g.position_mm[0])}
          y2={y(g.position_mm[1])}
          stroke="#b2bcc1"
          strokeWidth="11"
        />
        <path
          d={`M${x(g.position_mm[0]) + 8} ${y(g.position_mm[1]) + 9}l${g.grip > 0 ? -10 : -18} -18M${x(g.position_mm[0]) + 8} ${y(g.position_mm[1]) + 9}l${g.grip > 0 ? -8 : 2} -23`}
          stroke="#d3dce0"
          strokeWidth="5"
          fill="none"
          strokeLinecap="round"
        />
        <circle
          cx={x(g.position_mm[0])}
          cy={y(g.position_mm[1])}
          r="19"
          stroke="#8dc7ed"
          fill="none"
          strokeWidth="1"
          strokeDasharray="4 4"
          opacity={tool === "grasper" ? 0.8 : 0}
        />
      </g>
      <g>
        <line
          x1="25"
          y1="650"
          x2={x(c.position_mm[0])}
          y2={y(c.position_mm[1])}
          stroke="#263c42"
          strokeWidth="14"
        />
        <line
          x1="25"
          y1="650"
          x2={x(c.position_mm[0])}
          y2={y(c.position_mm[1])}
          stroke="#6b929b"
          strokeWidth="8"
        />
        <path
          d={`M${x(c.position_mm[0])} ${y(c.position_mm[1])}l7 -13l9 0`}
          stroke={c.power > 0 ? "#f8bd6d" : "#cbd9dc"}
          strokeWidth="4"
          fill="none"
        />
        <circle
          cx={x(c.position_mm[0])}
          cy={y(c.position_mm[1])}
          r="19"
          stroke="#ecc18e"
          fill="none"
          strokeWidth="1"
          strokeDasharray="4 4"
          opacity={tool === "cautery" ? 0.9 : 0}
        />
      </g>
      <g className="scene-scale">
        <path d="M54 59H144M54 54V64M144 54V64" stroke="#c3d2d8" />
        <text x="99" y="45" textAnchor="middle">
          10 mm model scale
        </text>
        <text x="54" y="610">
          X →
        </text>
        <text x="54" y="628">
          Y ↑
        </text>
      </g>
      <g className="scene-labels">
        <text x="650" y="133">
          LIVER (ILLUSTRATIVE)
        </text>
        <text x="517" y="270">
          GALLBLADDER NODE GRAPH
        </text>
        <text x="584" y="531">
          DUCT NETWORK
        </text>
      </g>
      <g className="scene-legend">
        <rect
          x="652"
          y="570"
          width="219"
          height="80"
          fill="#15222c"
          fillOpacity=".8"
          stroke="#49606d"
        />
        <circle cx="666" cy="587" r="4" fill="#9cb397" />
        <text x="677" y="591">
          Tissue state
        </text>
        <circle cx="666" cy="607" r="4" fill="#e18c45" />
        <text x="677" y="611">
          Thermal injury
        </text>
        <circle cx="666" cy="627" r="4" fill="#a72f3d" />
        <text x="677" y="631">
          Damage / bleeding
        </text>
      </g>
    </svg>
  );
}
