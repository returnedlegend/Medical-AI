# Desktop procedure and reinforcement-learning milestone

This milestone turns the research workbench into a native desktop application and adds a sequential simulated procedure. The Windows executable is the primary distribution target. A hosted page or a shortcut opening an external browser does not satisfy the desktop requirement.

## Acceptance criteria

1. A native application window embeds the complete interface, launches its own private local backend, stores data in a writable platform user-data directory, and shuts down its server/children when closed. The distributable carries its Python runtime and frontend so users do not need npm, a Python installation, or shell commands.
2. The interface uses the conventions of academic simulation software: a large procedure viewport, explicit tools and parameter controls, labelled units, an event/reward trace, experiment configuration, and model provenance. Marketing panels and business metric styling are removed.
3. Grasper and electrosurgical actions update persistent simulated tissue state. Excess traction/crushing and thermal dose cause irreversible damage, change future mechanical response, and can injure specific vessel/duct regions. Injury produces observable bleeding/leak or other explicitly synthetic physiology signals.
4. The simulator implements reset/action/observation/reward/termination with reproducible seeds. Reward components are inspectable. Tissue cannot heal instantly when a tool releases, repeated stationary exposure cannot farm progress reward, and terminal states reject further actions.
5. A policy is actually trained from observed transitions. Random and scripted comparisons use held-out seeds. A checkpoint records architecture, model/reward versions, seeds, and learning/evaluation outcomes. A failed or inconclusive training run is reported as such.
6. The procedure can be controlled manually and by the trained policy. The same transition function drives API, rendering, training, and exported episode logs; UI animation is not a separate source of invented injury state.
7. Tool-response equations, contact regions, units, assumptions, damage thresholds, reward terms, and termination criteria are documented. Physical validation is distinguished from software tests. Synthetic vitals, bleeding estimates, and damage thresholds are illustrative, not clinical outcome predictions.

## Scientific boundaries

The portable sequential tissue environment is a transparent mechanistic surrogate suitable for developing a feedback loop. The separate SOFA scene remains a distinct physical runtime. The app must identify which engine produced a result and must not imply that illustrative body responses are validated human physiology. The optional SOFA investigation will check whether irreversible element softening can be coupled to measured strain without modifying upstream assets; it does not establish calibrated rupture or bleeding mechanics.
