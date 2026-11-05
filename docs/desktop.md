# Native desktop application

Retraction Lab's desktop application embeds its research interface in a native
Qt window. It does not open the application in an external web browser. A private
local research service starts with the window, chooses an unused loopback port,
and stops when the application closes. The interface and models are bundled;
normal cohort and procedure workflows do not require Python, Node, or an Internet
connection on the user's computer.

## Windows application

The Windows build produces `RetractionLab-windows-x64.zip`. Extract the **entire**
archive to a writable location, then launch `RetractionLab/RetractionLab.exe`.
Keep the accompanying `_internal` directory beside the executable. Copying just
the `.exe` out of the application folder will not work.

An NSIS build also produces `RetractionLab-0.2.0-windows-x64-Setup.exe`.
This single installer places the full application in your user account's
programs directory and adds a Start menu shortcut without administrator access.
Uninstalling removes the installed program files and preserves studies and logs.

This is a 64-bit application targeting Windows 10/11. The desktop package is
unsigned. Its build manifest and SHA256 sidecar identify the delivered artifact;
the source and build workflow are available for independent rebuilding.
The optional Linux SOFA runtime is not included in the Windows desktop package.
The procedure laboratory is the stateful synthetic tissue environment described
under Help → Tissue response model. The finite cohort study remains a separate
analytic surrogate. Neither is a validated clinical model.

The Windows build workflow is `.github/workflows/desktop.yml`. It compiles with
Windows Python and includes a packaged-application smoke test. A configured
workflow is not evidence that a particular Windows build has run; use the
artifact's build manifest, smoke JSON, and workflow outcome for that claim.

## Local studies and exports

Studies, procedure histories, policy checkpoints, and diagnostic logs live in
the operating system's per-user application-data directory, not in the program
installation directory. On Windows this is under `%LOCALAPPDATA%`; on Linux it is
under the standard user-data directory. File → Open study data folder shows the
actual location. Updating or moving the program folder does not erase this data.

Only one application instance may open a particular study directory at a time.
A native lock prevents a second window from accidentally recovering or modifying
the first window's active work. The local service also requires an unpredictable
per-launch session header supplied only by the embedded window. The service is
bound to loopback, and the embedded page blocks external HTTP navigation.

Use the interface's export controls to save procedure episodes, training results,
cohort JSON, cohort CSV, and reports through the native Save dialog. The File menu
also exports the most recent completed cohort study. Canceling a Save dialog
leaves the study unchanged. Help → Open diagnostic logs opens the local log folder.

Help contains offline documentation for tissue response, reinforcement learning,
the cohort model, the desktop application, and the optional SOFA integration.
These documents are bundled with the application and require no repository
checkout or network connection.

## Rebuild from source

On Windows, install Git, Python 3.12, and Node 22, then run from the checkout:

```powershell
py -3.12 -m venv .desktop-venv
.desktop-venv\Scripts\python -m pip install -r backend/requirements-desktop.txt
.desktop-venv\Scripts\python scripts/build-desktop.py
```

The same Python build script runs on Linux to produce a Linux executable. A
Windows executable must be frozen using Windows Python; invoking PyInstaller
with Linux Python does not cross-compile a Windows application. The controlled
Wine build path uses actual Windows CPython and Windows wheels, and must be
distinguished from validation on a real Windows installation.

The build generates icons locally, bundles `frontend/dist` and offline docs,
collects backend model modules, and creates an application ZIP plus SHA256 and a
manifest recording Python, source fingerprints, Git revision/dirty status, and
model versions. `--skip-frontend` uses an existing frontend build and should only
be used after that build has been verified.

Install NSIS 3.11 and append `--installer --makensis "C:\\Program Files (x86)\\NSIS\\makensis.exe"`
to create the installer as well as the portable archive. Each build first copies
an explicit source snapshot so edits during packaging cannot change the
application midway through analysis. Exact source hashes and the environment's
physics fingerprint are bundled in `build-info.json` and the external manifest.

## Development and checks

```bash
# Run the native host from the checkout after building the frontend.
.desktop-venv/bin/python desktop/app.py

# Exercise resource selection, private authentication, actual research requests,
# clean shutdown, and durable episode replay after a service restart.
.desktop-venv/bin/python -m pytest desktop/tests -q

# Native GUI check: renders the embedded interface, initializes an episode via
# the actual New Episode button, presses Step, runs a terminal-handoff episode,
# exports it, and completes a 40-patient cohort. It writes JSON and a screenshot.
.desktop-venv/bin/python desktop/app.py --smoke-test /tmp/desktop-smoke.json \
  --data-dir /tmp/desktop-smoke-data --software-rendering

# Driver-independent service/model diagnosis: actual cohort, persistent episode
# replay, 20 DQN training episodes, held-out evaluation and a trained-policy step.
.desktop-venv/bin/python desktop/app.py --diagnose /tmp/desktop-diagnose.json
```

Graphical checks on this cloud host use an authenticated Xvfb display. The
container's test-only WebEngine environment may require Chromium's sandbox
compatibility flag; that flag is not enabled by the application in production.
Headless evidence does not replace testing the delivered package on the user's
Windows hardware. `--software-rendering` is available for machines whose graphics
drivers cannot create the normal browser rendering context.

## Runtime validation status

The standalone Linux application has passed real embedded-interface, manual
step, terminal episode, cohort, and offline Help checks using Qt 6.8.3, launched
outside the checkout. Its packaged diagnosis also passed source fingerprint,
restart/replay, DQN training, held-out evaluation, and trained-policy checks.

The native Windows build and packaged graphical smoke test passed for source
checkpoint `79bcf56` in [Windows workflow run 42068968304](https://github.com/returnedlegend/Medical-AI/actions/runs/42068968304).
That workflow separately enforces both packaged service/model diagnosis and
the actual Qt window, including interface controls and offline Help. The run
contains the Windows application and smoke evidence artifacts.

Actual Windows CPython and Windows wheels have also produced a PE32+ x64
application on this cloud host; its packaged model diagnosis passes under Wine.
Wine itself stalls on QtWebEngine initialization in a minimal example, so the
graphical validation above comes from the native Windows runner. Each later
artifact needs its own recorded build and smoke outcome.
