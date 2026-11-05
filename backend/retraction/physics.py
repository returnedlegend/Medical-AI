"""Optional upstream SOFA diagnostics, isolated from analytic research runs.

One child process owns one scene. Process completion is not model validation;
all public records explicitly retain model_ready=False.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .storage import Store

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER_TIMEOUT_SECONDS = 120
MAX_LOG_BYTES = 512_000
PURPOSE = "upstream-physics-diagnostic"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PhysicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=4294967295, strict=True)
    steps: int = Field(default=3, ge=1, le=50, strict=True)
    start_grasped: bool = False
    pan: float = Field(default=0, ge=-0.5, le=0.5)
    tilt: float = Field(default=0, ge=-0.5, le=0.5)
    spin: float = Field(default=0, ge=-0.5, le=0.5)
    insertion: float = Field(default=0, ge=-0.5, le=0.5)
    jaw: float = Field(default=0, ge=-0.5, le=0.5)


def runtime_root() -> Path:
    return Path(os.environ.get("RETRACTION_SOFA_DIR", "/workspace/.sofa-runtime"))


def runtime_status() -> dict:
    root = runtime_root()
    sentinels = [root / "venv/bin/python",
                 root / "SOFA_v24.06.00_Linux/plugins/SofaPython3/lib/python3/site-packages/Sofa",
                 root / "source/.upstream_commit",
                 PROJECT_ROOT / "scripts/run-sofa-smoke.sh"]
    available = all(path.exists() for path in sentinels)
    commit = None
    if available:
        candidate = (root / "source/.upstream_commit").read_text().strip()
        commit = candidate if re.fullmatch(r"[0-9a-f]{40}", candidate) else None
    return {
        "available": available, "purpose": PURPOSE, "model_ready": False,
        "reason": "Runtime files are installed; scene diagnostics still require inspection." if available else
                  "Optional SOFA runtime is not installed. Run scripts/setup-sofa.sh to prepare it.",
        "upstream_commit": commit, "sofa_version": "24.06.00" if available else None,
        "limits": {"max_steps": 50, "max_active_runs": 2, "max_workers": 1, "timeout_seconds": WORKER_TIMEOUT_SECONDS},
        "warning": "This upstream physics diagnostic does not implement the parameterized retraction study or validate surgical safety.",
    }


def worker_command(config: dict, output_path: Path) -> list[str]:
    config = PhysicsConfig.model_validate(config).model_dump()
    command = ["bash", str(PROJECT_ROOT / "scripts/run-sofa-smoke.sh"),
               "--output", str(output_path), "--seed", str(config["seed"]),
               "--steps", str(config["steps"]), "--size", "256"]
    if config["start_grasped"]:
        command.append("--start-grasped")
    for key in ("pan", "tilt", "spin", "insertion", "jaw"):
        command.extend([f"--{key}", str(config[key])])
    return command


def kill_process_group(process: subprocess.Popen) -> None:
    """Kill the whole scene/session, including timeout wrappers and children."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def sanitize_excerpt(value: str, directory: Path) -> str:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(value))
    for path, label in ((runtime_root(), "<sofa-runtime>"), (PROJECT_ROOT, "<project>"), (directory, "<run>")):
        text = text.replace(str(path), label)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    return " ".join(text.split())[:400]


def log_diagnostics(log_path: Path) -> dict:
    with log_path.open('rb') as handle:
        raw = handle.read(MAX_LOG_BYTES + 1)
    truncated = len(raw) > MAX_LOG_BYTES
    lines = raw[:MAX_LOG_BYTES].decode('utf-8', errors='replace').splitlines()
    errors = [line for line in lines if re.search(r"\[\s*(?:ERROR|FATAL)\s*\]", line, re.I)]
    warnings = [line for line in lines if re.search(r"\[\s*WARNING\s*\]", line, re.I)]
    return {"error_count": len(errors), "warning_count": len(warnings),
            "excerpts": [sanitize_excerpt(line, log_path.parent) for line in (errors + warnings)[:8]],
            "log_truncated": truncated}


class WorkerFailure(RuntimeError):
    pass


def execute_worker(config: dict, output_path: Path, *, process_started=None, process_finished=None) -> dict:
    """Run only the fixed local worker; no shell interpolation or network fetches."""
    log_path = output_path.with_suffix('.log')
    with log_path.open('wb') as log:
        process = subprocess.Popen(worker_command(config, output_path), cwd=PROJECT_ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   env={**os.environ, "RETRACTION_SOFA_DIR": str(runtime_root())})
        try:
            if process_started:
                process_started(process)
            try:
                code = process.wait(timeout=WORKER_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as error:
                kill_process_group(process)
                raise WorkerFailure(f"Physics diagnostic exceeded {WORKER_TIMEOUT_SECONDS} seconds; scene process group was stopped.") from error
            if code:
                raise WorkerFailure("Upstream physics worker failed. Inspect the server log and optional runtime setup.")
        finally:
            if process.poll() is None:
                kill_process_group(process)
            if process_finished:
                process_finished(process)
    if not output_path.is_file() or output_path.stat().st_size > 8_000_000:
        raise WorkerFailure("Physics worker did not produce a bounded result artifact.")
    try:
        result = json.loads(output_path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if not isinstance(result, dict) or not isinstance(result.get('steps'), list) or not result['steps']:
            raise ValueError("Missing step telemetry")
        image = output_path.with_suffix('.png')
        with image.open('rb') as handle:
            if handle.read(8) != PNG_SIGNATURE:
                raise ValueError("Missing PNG artifact")
        # Enforce JSON finiteness throughout telemetry, not only top-level fields.
        json.dumps(result, allow_nan=False)
    except (ValueError, OSError, TypeError) as error:
        raise WorkerFailure("Physics worker produced invalid telemetry or image artifacts.") from error
    recorded = result.get('diagnostics') or {}
    if not isinstance(recorded, dict):
        recorded = {}
    parsed = log_diagnostics(log_path)
    errors = max(int(recorded.get('error_count', 0)), parsed['error_count'])
    warnings = max(int(recorded.get('warning_count', 0)), parsed['warning_count'])
    messages = recorded.get('messages') or []
    result['diagnostics'] = {
        'error_count': errors, 'warning_count': warnings,
        'topology_error_count': int(recorded.get('topology_error_count', 0)),
        'messages': [{key: sanitize_excerpt(message.get(key, ''), output_path.parent)
                      for key in ('type', 'sender', 'message')}
                     for message in messages[:8] if isinstance(message, dict)],
        'excerpts': parsed['excerpts'], 'log_truncated': parsed['log_truncated'],
    }
    result.pop('image_path', None)
    result.pop('initial_image_path', None)
    result.update(purpose=PURPOSE, model_ready=False,
                  diagnostic_state='needs_review' if (errors or recorded.get('topology_error_count', 0)
                      or result.get('diagnostic_state') == 'needs_review') else 'infrastructure_checked')
    return result


class PhysicsService:
    def __init__(self, directory: Path):
        self.directory = directory
        self.store = Store(directory)
        self.store.recover_interrupted()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='physics')
        self.submit_lock = threading.Lock()
        self.process_lock = threading.Lock()
        self.processes: set[subprocess.Popen] = set()
        self.stopping = False

    def process_started(self, process):
        with self.process_lock:
            if self.stopping:
                kill_process_group(process)
                raise WorkerFailure('Server stopped before physics diagnostic could run.')
            self.processes.add(process)

    def process_finished(self, process):
        with self.process_lock:
            self.processes.discard(process)

    def work(self, run_id: str, config: dict):
        try:
            with self.process_lock:
                if self.stopping:
                    raise WorkerFailure('Server stopped before physics diagnostic could run.')
            self.store.update(run_id, status='running', progress=0.05)
            artifacts = self.directory / 'artifacts' / run_id
            artifacts.mkdir(parents=True, exist_ok=False)
            result = execute_worker(config, artifacts / 'result.json',
                                    process_started=self.process_started, process_finished=self.process_finished)
            result['image_url'] = f'/api/physics/runs/{run_id}/image'
            result['initial_image_url'] = f'/api/physics/runs/{run_id}/image?frame=initial'
            # Apply these again independently of the worker/stub's claimed state.
            result.update(purpose=PURPOSE, model_ready=False)
            summary = {key: result.get(key) for key in
                       ('diagnostic_state', 'sofa_version', 'sofa_env_commit', 'gallbladder_vertex_count', 'elapsed_seconds')}
            summary.update(purpose=PURPOSE, model_ready=False, steps_completed=len(result['steps']),
                           error_count=result.get('diagnostics', {}).get('error_count', 0),
                           warning_count=result.get('diagnostics', {}).get('warning_count', 0))
            self.store.update(run_id, status='completed', progress=1.0, summary=summary, result=result)
        except Exception as error:
            logger.exception('Physics diagnostic %s failed', run_id)
            message = str(error) if isinstance(error, WorkerFailure) else 'Physics diagnostic failed. Inspect the server log for details.'
            self.store.update(run_id, status='failed', error=message)

    def close(self):
        with self.process_lock:
            self.stopping = True
            processes = list(self.processes)
        for process in processes:
            kill_process_group(process)
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.store.recover_interrupted()


def initialize_physics(app, data_dir: Path | str):
    app.state.physics = PhysicsService(Path(data_dir) / 'physics')


def shutdown_physics(app):
    app.state.physics.close()


router = APIRouter(prefix='/api/physics', tags=['optional physics diagnostics'])


def service(request: Request) -> PhysicsService:
    return request.app.state.physics


def record_for(request: Request, run_id: UUID, *, include_result=False):
    record = service(request).store.get(str(run_id), include_result=include_result)
    if record is None:
        raise HTTPException(404, 'Physics run not found')
    return record


@router.get('/status')
def status():
    return runtime_status()


@router.post('/runs', status_code=202)
def create_run(config: PhysicsConfig, request: Request):
    if not runtime_status()['available']:
        raise HTTPException(503, 'Optional SOFA runtime is unavailable. Run scripts/setup-sofa.sh first.')
    jobs = service(request)
    with jobs.submit_lock:
        if jobs.stopping:
            raise HTTPException(503, 'Physics service is stopping')
        if jobs.store.active_count() >= 2:
            raise HTTPException(429, 'Two physics diagnostics are already active. Wait for a run to finish.')
        record = jobs.store.create(config.model_dump())
        jobs.executor.submit(jobs.work, record['id'], record['config'])
    return record


@router.get('/runs')
def list_runs(request: Request, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    return {'runs': service(request).store.list(limit=limit, offset=offset)}


@router.get('/runs/{run_id}')
def read_run(request: Request, run_id: UUID):
    return record_for(request, run_id)


@router.get('/runs/{run_id}/results')
def read_result(request: Request, run_id: UUID):
    record = record_for(request, run_id, include_result=True)
    if record['status'] != 'completed':
        raise HTTPException(409, 'Physics results are available after the diagnostic completes.')
    return record['result']


@router.get('/runs/{run_id}/image')
def read_image(request: Request, run_id: UUID, frame: Literal['initial', 'final'] = 'final'):
    record = record_for(request, run_id)
    if record['status'] != 'completed':
        raise HTTPException(409, 'Physics images are available after the diagnostic completes.')
    filename = 'result-initial.png' if frame == 'initial' else 'result.png'
    directory = (service(request).directory / 'artifacts' / str(run_id)).resolve()
    path = directory / filename
    if not path.is_file() or path.resolve().parent != directory:
        raise HTTPException(404, 'Physics image artifact is unavailable.')
    with path.open('rb') as handle:
        if handle.read(8) != PNG_SIGNATURE:
            raise HTTPException(404, 'Physics image artifact is invalid.')
    return FileResponse(path, media_type='image/png', headers={'Cache-Control': 'no-store'})
