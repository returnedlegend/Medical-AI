import { useEffect, useState } from "react";
import { request } from "./api";
interface Spec {
  version: string;
  engine: string;
  observation_size: number;
  node_count: number;
  action_names: string[];
  action_semantics: string;
  objective: string;
  default_config: Record<string, number | boolean>;
  config_limits: Record<string, number[]>;
  reward: Record<string, number>;
  assumptions: string[];
  physics_fingerprint?: string;
}
export default function ModelDocumentation() {
  const [spec, setSpec] = useState<Spec | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    const a = new AbortController();
    request<Spec>("/procedure/spec", { signal: a.signal })
      .then(setSpec)
      .catch((e) => {
        if (!a.signal.aborted) setError(e.message);
      });
    return () => a.abort();
  }, []);
  return (
    <div className="academic-document model-documentation">
      <div className="academic-page-title">
        <h1>Stateful model, methods, and provenance</h1>
        <span>
          {spec
            ? `${spec.version} · ${spec.observation_size} observations`
            : "Local engine specification"}
        </span>
      </div>
      <div className="documentation-layout">
        <nav aria-label="Model documentation contents">
          <a href="#model-loop">Agent–environment loop</a>
          <a href="#model-tools">Instrument commands</a>
          <a href="#model-injury">Material and heat memory</a>
          <a href="#model-physiology">Bleeding, leakage, and vitals</a>
          <a href="#model-reward">Reward and termination</a>
          <a href="#model-parameters">Parameters and units</a>
          <a href="#model-provenance">Replay and model boundaries</a>
        </nav>
        <article>
          <section id="model-loop">
            <h2>1. Agent–environment loop</h2>
            <p>
              The procedure laboratory implements action → tissue response →
              observation and reward. Its 23 material sites form a planar,
              overdamped spring network: 16 gallbladder sites, two arterial
              sites, two cystic-duct sites, and three common-duct sites. Tool
              contact transfers force into this network. Tissue injury persists
              and changes future mechanics.
            </p>
            <p>
              The displayed 3D surface interpolates the network’s XY positions;
              depth, liver context, surface shading, and fluid particles are
              illustrative. The underlying mechanics have no depth/contact
              geometry and are separate from the optional SOFA diagnostic scene.
            </p>
            <div className="documentation-equation">
              aₜ → [tool motion, contact, springs, heat, injury, flow] → (oₜ₊₁,
              rₜ, terminated, truncated)
            </div>
            <p>
              The fully observed environment exposes material, tool, patient,
              configuration, and physiological state. A camera-only policy would
              be a different, partially observed problem.
            </p>
          </section>
          <section id="model-tools">
            <h2>2. Instrument commands and keyboard controls</h2>
            <div
              className="documentation-table"
              tabIndex={0}
              role="region"
              aria-label="Instrument action specification"
            >
              <table>
                <thead>
                  <tr>
                    <th>Action</th>
                    <th>Command semantics</th>
                    <th>Model response</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Grasper X/Y</td>
                    <td>Normalized −1…+1 velocities; up to 10 mm/s</td>
                    <td>
                      Moves within the bounded XY workspace; contact traction
                      deforms the material network.
                    </td>
                  </tr>
                  <tr>
                    <td>Jaw closure</td>
                    <td>
                      Positive sets closure; zero holds; below −0.2 releases
                    </td>
                    <td>
                      Acquires a nearby intact site within 3 mm when closure
                      ≥0.25. Closure above 0.75 can accumulate crush injury.
                    </td>
                  </tr>
                  <tr>
                    <td>Cautery X/Y and power</td>
                    <td>
                      Separate position commands; positive power heats for one
                      interval
                    </td>
                    <td>
                      Gaussian heat delivery with local temperature,
                      irreversible dose, material injury, and simplified
                      coagulation response.
                    </td>
                  </tr>
                  <tr>
                    <td>Handoff</td>
                    <td>Action above 0.5 terminates the episode</td>
                    <td>
                      Preserves the injured state and records a handoff; it does
                      not simulate healing or successful rescue.
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p>
              <kbd>G</kbd> selects the grasper; <kbd>C</kbd> selects cautery.
              Arrow keys move the selected instrument; <kbd>Space</kbd> toggles
              grasp or delivers one cautery pulse when the viewport has focus.{" "}
              <kbd>Escape</kbd> pauses playback. Click the viewport for one
              bounded movement toward a point. Drag and wheel change only the 3D
              camera.
            </p>
            <p>
              Cutting/removing material, clips, suturing, irrigation, and
              suction are not implemented.
            </p>
          </section>
          <section id="model-injury">
            <h2>3. Material injury and thermal memory</h2>
            <p>
              Mechanical injury m and accumulated thermal dose q never decrease
              within an episode. Cooling reduces temperature but does not
              reverse stored dose or stiffness loss.
            </p>
            <div className="documentation-equation">
              h = 1 − exp(−q / 0.8)
              <br />D = 1 − (1 − m)(1 − h)
              <br />k = k₀(1 − 0.9D)
            </div>
            <p>
              Restoring forces, same-material neighbor springs, and grasper
              coupling determine nodal movement in five fixed substeps. Injury
              reduces both restoring stiffness and contact traction. Material
              contact fails at combined injury 0.9. Excessive strain, force, and
              jaw compression add mechanical injury.
            </p>
            <div className="documentation-equation">
              Δm ∝ Δt · max(strain / limit − 1, 0)² + contact overload + crush
              injury
              <br />
              ΔT = Δt · [160 · power · Gaussian(distance, 2.4 mm) − (T − 37) /
              3]
              <br />
              Δq = Δt · max(T − 43, 0) / 100
            </div>
            <p>
              Heat is zero outside 7.2 mm. The thermal dose is a simple
              degree-second surrogate, not CEM43 or an Arrhenius burn model. The
              20-N saturation and workspace clamps are numerical bounds, not an
              instrument safety controller.
            </p>
          </section>
          <section id="model-physiology">
            <h2>
              4. Persistent bleeding, bile leakage, and synthetic vital signs
            </h2>
            <p>
              Arterial injury above 0.08 drives ongoing bleeding. A bounded
              coagulation state can accumulate during powered heating at 50–70
              °C; hotter tissue can lose that sealing state. Duct injury drives
              persistent bile leakage, weighted more strongly for the common
              duct.
            </p>
            <div className="documentation-equation">
              bleeding ∝ Σ max(Dartery − 0.08, 0) · (1 − coagulation) · pressure
              fraction
              <br />
              MAP = clip[90 · (1 − 2.5 · blood_loss / 5000)]
              <br />
              HR = clip[72 + 180 · blood_loss / 5000]
            </div>
            <p>
              Accumulated blood loss feeds back into pressure and the next
              bleeding rate. Blood and bile also obscure the exposure proxy.
              These are transparent toy feedback equations, not human
              hemodynamics; compensation timing, anesthesia, replacement fluids,
              infection, and systemic inflammatory response are absent.
            </p>
          </section>
          <section id="model-reward">
            <h2>
              5. Reward decomposition, controllers, and episode boundaries
            </h2>
            <p>
              The event console and policy inspector show rewards computed from
              actual transitions. Progress is credited only for a new exposure
              record under the model’s acceptance conditions; reopening the same
              view cannot repeatedly collect that reward.
            </p>
            <div
              className="documentation-table"
              tabIndex={0}
              role="region"
              aria-label="Reward components"
            >
              <table>
                <thead>
                  <tr>
                    <th>Component</th>
                    <th>Illustrative weight</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>
                      New record-high exposure under acceptance conditions
                    </td>
                    <td>+4 × gain</td>
                  </tr>
                  <tr>
                    <td>New material injury</td>
                    <td>−12 × summed increment</td>
                  </tr>
                  <tr>
                    <td>New artery / duct injury</td>
                    <td>Additional −40 × increment</td>
                  </tr>
                  <tr>
                    <td>Blood / bile loss</td>
                    <td>−0.005 / mL blood; −0.05 / mL bile</td>
                  </tr>
                  <tr>
                    <td>Force excess and elapsed action</td>
                    <td>Force penalty plus −0.01 per step</td>
                  </tr>
                  <tr>
                    <td>Successful modeled hold / severe-injury stop</td>
                    <td>+15 / −20 once</td>
                  </tr>
                  <tr>
                    <td>Handoff / time horizon</td>
                    <td>−2 / −1 once</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p>
              A scripted feedback baseline uses the public observation and is
              explicitly separate from reinforcement learning. The local NumPy
              Double DQN learns among declared tool-control primitives using
              transitions, replay, and environment reward. The alternative
              Double-Q table bins current contact, exposure error, force,
              strain, and mechanical injury; this reduced state omits
              information and is not Markov. Training checkpoints, comparator
              rollouts, and seed/configuration metadata are inspectable.
              Learning does not establish a useful or safe surgical policy.
            </p>
            <p>
              Playback reacts to current observations with fixed checkpoint
              weights. New training jobs perform the learning updates; saved
              manual episodes are not automatically demonstrations. The declared
              action primitives use a fixed 0.55 grasp command and 0.35 cautery
              command, so these policies do not learn continuous jaw-pressure or
              power optimization.
            </p>
            <p>
              Response protocols are scripted controlled interventions: gentle
              traction, sustained jaw closure, artery or duct contact heating,
              and a remote-heat control. Each action passes through the same
              stateful engine and is retained with its protocol phase.
              Intervention completion is separate from episode success. Manual
              tool actions interrupt an active protocol; Reset creates a new
              protocol episode.
            </p>
            <p>
              The modeled success gate requires sustained target exposure, low
              injury, force/strain acceptance, cautery off, and all sites at or
              below 43 °C. Severe injury, blood-loss stop, handoff, and success
              terminate; the time horizon truncates. Further actions are
              rejected until reset. All thresholds are research assumptions.
            </p>
          </section>
          <section id="model-parameters">
            <h2>6. Live environment defaults and bounds</h2>
            <p>
              These values are read from the current local engine specification.
              Episode-specific patient settings and limits are also available in
              the laboratory’s Tissue / tools inspector.
            </p>
            {error && (
              <p className="documentation-error" role="alert">
                Specification unavailable: {error}
              </p>
            )}
            {spec ? (
              <>
                <p>
                  <strong>Objective:</strong> {spec.objective}
                </p>
                <div
                  className="documentation-table"
                  tabIndex={0}
                  role="region"
                  aria-label="Live environment parameters"
                >
                  <table>
                    <thead>
                      <tr>
                        <th>Parameter</th>
                        <th>Default</th>
                        <th>Allowed range</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(spec.default_config).map(
                        ([name, value]) => (
                          <tr key={name}>
                            <td>{name.replaceAll("_", " ")}</td>
                            <td>{String(value)}</td>
                            <td>
                              {spec.config_limits[name]?.join(" … ") ??
                                "true / false"}
                            </td>
                          </tr>
                        ),
                      )}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              !error && <p>Reading local specification…</p>
            )}
          </section>
          <section id="model-provenance">
            <h2>7. Reproducibility and interpretation</h2>
            <p>
              Episode exports retain the engine version, configuration, seed,
              patient, observation, actions, transition sources, reward
              components, events, termination state, and the exact model-source
              fingerprint. Loading a saved episode restores its nodal state and
              recorded history. The recording slider inspects stored transition
              states without changing the episode. Reset creates a new episode
              and retains the previous record. Archived model revisions remain
              inspectable and exportable; their actions and policies cannot run
              against changed mechanics.
            </p>
            <p>
              A bundled reference, when available, is selected using
              training-disjoint validation rollouts and is labeled accordingly.
              Those results are not an untouched final test. Matching a
              checkpoint’s simulator settings does not establish generalization
              to a different tissue model or clinical use.
            </p>
            <p>
              The cohort study remains a separate static analytic strategy sweep
              with train / held-out comparisons. SOFA diagnostics run a separate
              upstream scene with a connectivity wrapper; they do not implement
              the stateful procedure model or validate its equations.
            </p>
            <p>
              <strong>Research boundary:</strong> geometry, material properties,
              heat dose, harm thresholds, exposure, and physiological responses
              are illustrative and uncalibrated. Passing software invariants
              does not validate real tissue mechanics or clinical safety. No
              patient data are used.
            </p>
            <div className="documentation-provenance">
              <span>
                Engine: {spec?.engine ?? "local specification unavailable"}
              </span>
              <span>Model version: {spec?.version ?? "—"}</span>
              <span title={spec?.physics_fingerprint}>
                Mechanics source:{" "}
                {spec?.physics_fingerprint?.slice(0, 16) ?? "not recorded"}
              </span>
              <span>
                Observation dimension: {spec?.observation_size ?? "—"}
              </span>
            </div>
          </section>
        </article>
      </div>
    </div>
  );
}
