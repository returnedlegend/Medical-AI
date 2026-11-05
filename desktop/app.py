"""Native Qt window with an embedded interface and private local research API."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

# Source checkout entry point; frozen builds bundle retraction as Python modules.
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desktop.runtime import LocalServer, TOKEN_HEADER, resource_root
from retraction import __version__


def configure_logging(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "application.log"
    handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    return path


def main() -> int:
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Retraction Lab native desktop application")
    parser.add_argument("--data-dir", type=Path, help="Override the writable per-user study directory")
    parser.add_argument("--smoke-test", type=Path, help="Run a native UI/API smoke check and write evidence JSON")
    parser.add_argument("--diagnose", type=Path, help="Check bundled service/models and write JSON; does not test the graphical window")
    parser.add_argument("--software-rendering", action="store_true", help="Use software OpenGL on machines with graphics-driver issues")
    args = parser.parse_args()
    if args.diagnose:
        from desktop.diagnostics import diagnose
        return diagnose(args.diagnose, args.data_dir)
    if args.software_rendering:
        os.environ.setdefault("QT_OPENGL", "software")
        os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")

    from PySide6.QtCore import QLockFile, QStandardPaths, Qt, QTimer, QUrl
    from PySide6.QtGui import QAction, QDesktopServices, QIcon
    from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QMainWindow, QMessageBox, QTextBrowser, QVBoxLayout
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineUrlRequestInterceptor
    from PySide6.QtWebEngineWidgets import QWebEngineView

    app = QApplication(sys.argv[:1])
    app.setOrganizationName("RetractionLab")
    app.setApplicationName("Retraction Lab")
    app.setApplicationVersion(__version__)
    icon_path = resource_root() / "desktop/assets/icon.png"
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))
    data_directory = args.data_dir or Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation))
    data_directory.mkdir(parents=True, exist_ok=True)
    log_path = configure_logging(data_directory / "logs")
    lock = QLockFile(str(data_directory / "desktop.lock"))
    if not lock.tryLock(0):
        QMessageBox.information(None, "Retraction Lab is already running", "Another Retraction Lab window is using this study directory. Close it before starting a second instance.")
        return 2
    server = LocalServer(data_directory)
    try:
        server.start()
    except Exception:
        logging.exception("Application startup failed")
        QMessageBox.critical(None, "Retraction Lab could not start", f"The local research service could not start.\n\nDiagnostic log: {log_path}")
        lock.unlock()
        return 1

    class LocalRequestInterceptor(QWebEngineUrlRequestInterceptor):
        def interceptRequest(self, info):
            url = info.requestUrl()
            if url.scheme() in ("http", "https"):
                if url.scheme() == "http" and url.host() == "127.0.0.1" and url.port() == urlparse(server.url).port:
                    info.setHttpHeader(TOKEN_HEADER.encode(), server.token.encode())
                else:
                    info.block(True)

    class ApplicationPage(QWebEnginePage):
        def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
            if url.scheme() in ("about", "data", "blob"):
                return True
            if url.scheme() == "http" and url.host() == "127.0.0.1" and url.port() == urlparse(server.url).port:
                return True
            return False

        def javaScriptConsoleMessage(self, level, message, line_number, source_id):
            logging.getLogger("desktop.javascript").info("%s:%s %s", urlparse(source_id).path, line_number, message)

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Retraction Lab — Research Workbench")
            self.resize(1440, 960)
            self.setMinimumSize(1000, 700)
            self.view = QWebEngineView(self)
            self.profile = QWebEngineProfile("retraction-desktop", self)
            self.profile.setPersistentStoragePath(str(data_directory / "web-profile"))
            self.profile.setCachePath(str(data_directory / "web-cache"))
            self.interceptor = LocalRequestInterceptor(self)
            self.profile.setUrlRequestInterceptor(self.interceptor)
            self.page = ApplicationPage(self.profile, self.view)
            self.view.setPage(self.page)
            self.view.loadFinished.connect(lambda ok: logging.getLogger("desktop.webview").info("Interface load finished: %s", ok))
            self.profile.downloadRequested.connect(self.download)
            self.setCentralWidget(self.view)
            self.statusBar().showMessage("Local research workspace · synthetic simulation")
            self.make_menu()
            self.view.load(QUrl(server.url))

        def make_menu(self):
            file_menu = self.menuBar().addMenu("&File")
            for label, suffix, format_name in (("Export latest results as &JSON…", "json", "json"), ("Export latest cohort as &CSV…", "csv", "csv"), ("Export latest &report…", "md", "report")):
                action = QAction(label, self)
                action.triggered.connect(lambda checked=False, ext=suffix, fmt=format_name: self.export_latest(ext, fmt))
                file_menu.addAction(action)
            file_menu.addSeparator()
            action = QAction("Open study data &folder", self)
            action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(data_directory))))
            file_menu.addAction(action)
            file_menu.addSeparator()
            exit_action = QAction("E&xit", self)
            exit_action.setShortcut("Ctrl+Q")
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)
            view_menu = self.menuBar().addMenu("&View")
            reload_action = QAction("&Reload interface", self)
            reload_action.setShortcut("Ctrl+R")
            reload_action.triggered.connect(self.view.reload)
            view_menu.addAction(reload_action)
            for title, factor in (("Zoom in", 1.1), ("Zoom out", 1 / 1.1)):
                action = QAction(title, self)
                action.triggered.connect(lambda checked=False, f=factor: self.view.setZoomFactor(max(0.6, min(1.8, self.view.zoomFactor() * f))))
                view_menu.addAction(action)
            help_menu = self.menuBar().addMenu("&Help")
            for label, document in (("Tissue response model", "tissue-response.md"), ("Reinforcement learning", "rl-training.md"), ("Analytic cohort model", "model.md"), ("Desktop application guide", "desktop.md")):
                action = QAction(label, self)
                action.triggered.connect(lambda checked=False, name=document: self.show_documentation(name))
                help_menu.addAction(action)
            help_menu.addSeparator()
            logs = QAction("Open diagnostic &logs", self)
            logs.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_path.parent))))
            help_menu.addAction(logs)
            about = QAction("&About Retraction Lab", self)
            about.triggered.connect(self.show_about)
            help_menu.addAction(about)

        def show_about(self):
            provenance = resource_root() / "build-info.json"
            build = json.loads(provenance.read_text(encoding="utf-8")) if provenance.is_file() else {}
            revision = (build.get("git_commit") or "source checkout")[:12]
            versions = build.get("model_versions", {})
            description = f"Retraction Lab {__version__}\nBuild: {revision}\n"
            if versions:
                description += f"Cohort model: {versions.get('cohort')}\nProcedure model: {versions.get('procedure')}\n"
            description += "\nA local computational research workbench with a stateful synthetic tissue laboratory and an analytic cohort study.\n\nResults are synthetic research outputs, not evidence of clinical safety.\n\nHelp contains the offline model documentation. Studies and logs remain in your local user-data folder."
            QMessageBox.about(self, "About Retraction Lab", description)

        def show_documentation(self, initial):
            dialog = QDialog(self)
            dialog.setWindowTitle("Retraction Lab — Offline documentation")
            dialog.resize(1000, 760)
            layout = QVBoxLayout(dialog)
            selector = QComboBox(dialog)
            documents = [("Tissue response model", "tissue-response.md"), ("Reinforcement learning", "rl-training.md"), ("Analytic cohort model", "model.md"), ("Desktop application guide", "desktop.md"), ("SOFA integration", "sofa-integration.md")]
            for label, name in documents:
                selector.addItem(label, name)
            viewer = QTextBrowser(dialog)
            viewer.setOpenExternalLinks(False)
            viewer.setOpenLinks(False)
            viewer.anchorClicked.connect(lambda url: viewer.scrollToAnchor(url.fragment()) if url.fragment() else None)
            def load_document(index):
                name = selector.itemData(index)
                path = resource_root() / "docs" / name
                viewer.setMarkdown(path.read_text(encoding="utf-8") if path.is_file() else "This document is unavailable in this build.")
            selector.currentIndexChanged.connect(load_document)
            selector.setCurrentIndex(next((i for i, item in enumerate(documents) if item[1] == initial), 0))
            load_document(selector.currentIndex())
            layout.addWidget(selector)
            layout.addWidget(viewer)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            dialog.exec()

        def download(self, request):
            destination, _ = QFileDialog.getSaveFileName(self, "Save research export", str(Path.home() / request.downloadFileName()))
            if not destination:
                request.cancel()
                return
            path = Path(destination)
            request.setDownloadDirectory(str(path.parent))
            request.setDownloadFileName(path.name)
            request.accept()
            request.isFinishedChanged.connect(lambda: self.statusBar().showMessage(f"Export saved: {path.name}" if request.state().name == "DownloadCompleted" else "Export interrupted", 8000))

        def export_latest(self, extension, format_name):
            try:
                completed = [record for record in server.request("/api/runs?limit=200")["runs"] if record["status"] == "completed"]
                if not completed:
                    QMessageBox.information(self, "No completed study", "Run a cohort study before exporting its results.")
                    return
                record = completed[0]
                name = f"retraction-{record['id']}.{extension}"
                destination, _ = QFileDialog.getSaveFileName(self, "Export latest completed study", str(Path.home() / name))
                if not destination:
                    return
                route = f"/api/runs/{record['id']}/report" if format_name == "report" else f"/api/runs/{record['id']}/export?format={format_name}"
                Path(destination).write_bytes(server.request_bytes(route))
                self.statusBar().showMessage(f"Export saved: {Path(destination).name}", 8000)
            except Exception:
                logging.exception("Native export failed")
                QMessageBox.warning(self, "Export failed", "The export could not be saved. Check the destination folder and diagnostic log.")

    window = MainWindow()
    window.show()
    smoke_executor = None
    if args.smoke_test:
        evidence = {"status": "running", "checks": {}, "packaged": bool(getattr(sys, "frozen", False))}
        smoke_executor = ThreadPoolExecutor(max_workers=1)

        def backend_check():
            # Exercise the stateful procedure, a terminal handoff, and a replay
            # read from the durable episode store as well as the cohort workflow.
            session = server.request("/api/procedure/sessions", {"seed": 42})
            for action in ([0, 0, .55, 0, 0, 0, 0], [.15, 0, .55, 0, 0, 0, 0], [0, .15, .55, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]):
                session = server.request(f"/api/procedure/sessions/{session['id']}/step", {"action": action, "expected_revision": session["revision"]})
            assert session["status"] == "terminated"
            restored = server.request(f"/api/procedure/sessions/{session['id']}")
            assert restored["revision"] == 4
            episode_export = server.request(f"/api/procedure/sessions/{session['id']}/export")
            assert episode_export
            evidence["checks"]["procedure_workflow"] = {"session_id": session["id"], "transitions": 4, "status": session["status"]}
            created = server.request("/api/runs", {"n_patients": 40, "seed": 812})
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                record = server.request(f"/api/runs/{created['id']}")
                if record["status"] == "completed":
                    results = server.request(f"/api/runs/{created['id']}/results")
                    assert len(results["patients"]) == 40
                    return {"run_id": record["id"], "patient_count": 40, "status": "completed"}
                if record["status"] == "failed":
                    raise RuntimeError("Native smoke experiment failed")
                time.sleep(0.1)
            raise TimeoutError("Native smoke experiment timed out")

        future = smoke_executor.submit(backend_check)
        smoke_timer = QTimer(window)
        smoke_timer.setInterval(300)
        started = time.monotonic()
        documentation_checked = False
        documentation_busy = False

        def finish_smoke(dom):
            nonlocal documentation_checked, documentation_busy
            if evidence["status"] != "running":
                return
            if isinstance(dom, str):
                try:
                    dom = json.loads(dom)
                except ValueError:
                    dom = None
            if isinstance(dom, dict):
                evidence["last_dom"] = dom
            if isinstance(dom, dict) and dom.get("rendered") and dom.get("manualStep") and future.done():
                if documentation_busy:
                    return
                if not documentation_checked:
                    documentation_busy = True
                    documentation_checked = True
                    def inspect_documentation():
                        nonlocal documentation_busy
                        dialog = app.activeModalWidget()
                        viewer = dialog.findChild(QTextBrowser) if dialog else None
                        files = ("tissue-response.md", "rl-training.md", "model.md", "desktop.md")
                        available = all((resource_root() / "docs" / name).is_file() for name in files)
                        readable = viewer is not None and len(viewer.toPlainText()) > 100
                        evidence["checks"]["offline_documentation"] = {"opened": readable, "all_documents_bundled": available}
                        if dialog:
                            dialog.accept()
                        documentation_busy = False
                    QTimer.singleShot(100, inspect_documentation)
                    window.show_documentation("tissue-response.md")
                try:
                    assert evidence["checks"]["offline_documentation"] == {"opened": True, "all_documents_bundled": True}
                    evidence["checks"]["native_webview"] = True
                    evidence["checks"]["cohort_workflow"] = future.result()
                    evidence["checks"]["frontend_root"] = dom
                    evidence["status"] = "passed"
                    args.smoke_test.parent.mkdir(parents=True, exist_ok=True)
                    screenshot = args.smoke_test.with_suffix(".png")
                    assert window.grab().save(str(screenshot))
                    evidence["screenshot"] = str(screenshot)
                except Exception as exc:
                    evidence["status"] = "failed"
                    evidence["error"] = str(exc)
                args.smoke_test.write_text(json.dumps(evidence, indent=2) + "\n")
                smoke_timer.stop()
                QTimer.singleShot(100, lambda: app.exit(0 if evidence["status"] == "passed" else 1))

        def poll_smoke():
            if time.monotonic() - started > 90:
                evidence["status"] = "failed"
                evidence["error"] = "Native interface did not render within 90 seconds"
                args.smoke_test.parent.mkdir(parents=True, exist_ok=True)
                args.smoke_test.write_text(json.dumps(evidence, indent=2) + "\n")
                smoke_timer.stop()
                app.exit(1)
                return
            window.page.runJavaScript("""(() => {
                const root = document.querySelector('#root');
                if (!window.__nativeSmokeStarted) {
                    const start = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'New episode');
                    if (start && !start.disabled) { start.click(); window.__nativeSmokeStarted = true; }
                }
                const caption = document.querySelector('.viewport-caption')?.textContent || '';
                const active = caption.includes('active');
                if (active && !window.__nativeSmokeStepped) {
                    const step = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Step');
                    if (step && !step.disabled) { step.click(); window.__nativeSmokeStepped = true; }
                }
                const clock = document.querySelector('.procedure-clock')?.textContent || '';
                return JSON.stringify({title:document.title, rendered:(root?.innerText?.length || 0) > 100,
                    episodeActive:active, manualStep:/step [1-9][0-9]*/.test(clock), clock,
                    excerpt:document.body?.innerText?.slice(0,160)});
            })()""", finish_smoke)

        smoke_timer.timeout.connect(poll_smoke)
        smoke_timer.start()

    result = app.exec()
    window.page.deleteLater()
    if smoke_executor:
        smoke_executor.shutdown(wait=True)
    server.stop()
    lock.unlock()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
