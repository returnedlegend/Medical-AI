"""A study remains exclusively owned until its background work has stopped."""

from pathlib import Path
import sys
import threading

from PySide6.QtCore import QLockFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from desktop.runtime import LocalServer
from retraction import main as api


def test_full_shutdown_keeps_study_locked_until_work_is_saved(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    stopping = threading.Event()
    stopped = threading.Event()
    shutdown_errors = []
    experiment = api.execute_experiment

    def held_experiment(config, progress):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("The test did not release the cohort calculation")
        return experiment(config, progress)

    monkeypatch.setattr(api, "execute_experiment", held_experiment)
    web = tmp_path / "frontend"
    web.mkdir()
    (web / "index.html").write_text("<html>Desktop shutdown regression</html>")
    directory = tmp_path / "study"
    directory.mkdir()
    owner = QLockFile(str(directory / "desktop.lock"))
    contender = QLockFile(str(directory / "desktop.lock"))
    assert owner.tryLock(0)
    service = LocalServer(directory, web_directory=web)
    reopened = None
    shutdown_thread = None

    def shutdown():
        stopping.set()
        try:
            service.stop(timeout=None)
        except BaseException as error:
            shutdown_errors.append(error)
        finally:
            owner.unlock()
            stopped.set()

    try:
        service.start()
        created = service.request("/api/runs", {"n_patients": 40, "seed": 621})
        assert entered.wait(5)
        shutdown_thread = threading.Thread(target=shutdown, name="desktop-shutdown-test")
        shutdown_thread.start()
        assert stopping.wait(5)
        assert not stopped.wait(0.1)
        assert not contender.tryLock(0), "A second window acquired a study still being written"

        release.set()
        assert stopped.wait(10)
        shutdown_thread.join(timeout=1)
        assert not shutdown_thread.is_alive()
        assert not shutdown_errors
        assert contender.tryLock(0)

        reopened = LocalServer(directory, web_directory=web)
        reopened.start()
        record = reopened.request(f"/api/runs/{created['id']}")
        assert record["status"] == "completed"
        assert record["error"] is None
        results = reopened.request(f"/api/runs/{created['id']}/results")
        assert len(results["patients"]) == 40
    finally:
        release.set()
        service.stop(timeout=None)
        if shutdown_thread is not None:
            shutdown_thread.join(timeout=10)
        if reopened is not None:
            reopened.stop(timeout=None)
        owner.unlock()
        contender.unlock()
