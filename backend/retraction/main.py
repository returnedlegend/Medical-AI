"""Local research API, bounded background jobs, and optional production web UI."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .storage import Store
from .physics import router as physics_router, initialize_physics, shutdown_physics

logger = logging.getLogger(__name__)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    n_patients: int = Field(default=240, ge=40, le=2000)
    seed: int = Field(default=42, ge=0, le=4294967295)
    threshold_profile: Literal["conservative", "nominal", "permissive"] = "nominal"
    force_limit_n: float | None = Field(default=None, gt=0, le=20)
    strain_limit: float | None = Field(default=None, gt=0, le=3)
    exposure_threshold: float | None = Field(default=None, gt=0, le=1)
    attachment_min_mm: float = Field(default=8.0, ge=4, le=60)
    attachment_max_mm: float = Field(default=35.0, ge=4, le=60)
    adhesion_max_count: int = Field(default=6, ge=0, le=12)
    adhesion_stiffness_min: float = Field(default=0.03, ge=0, le=0.5)
    adhesion_stiffness_max: float = Field(default=0.20, ge=0, le=0.5)
    distension_min: float = Field(default=0.8, ge=0.5, le=2.5)
    distension_max: float = Field(default=1.8, ge=0.5, le=2.5)

    @model_validator(mode="after")
    def ordered_ranges(self):
        for lower, upper in (
            (self.attachment_min_mm, self.attachment_max_mm),
            (self.adhesion_stiffness_min, self.adhesion_stiffness_max),
            (self.distension_min, self.distension_max),
        ):
            if lower > upper:
                raise ValueError("Each patient range minimum must not exceed its maximum")
        return self


class RunRecord(BaseModel):
    id: str
    created_at: str
    updated_at: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: float
    config: dict
    error: str | None
    summary: dict | None


class PreviewPatient(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    attachment_length_mm: float = Field(default=24, ge=4, le=60)
    adhesion_count: int = Field(default=2, ge=0, le=12)
    adhesion_stiffness_n_per_mm: float = Field(default=0.1, ge=0, le=0.5)
    distension: float = Field(default=1.2, ge=0.5, le=2.5)


class PreviewStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    grasp_position: float = Field(default=0.75, ge=0, le=1)
    direction_deg: float = Field(default=35, ge=0, le=180)
    distance_mm: float = Field(default=12, ge=0, le=30)


class PreviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    patient: PreviewPatient = Field(default_factory=PreviewPatient)
    strategy: PreviewStrategy = Field(default_factory=PreviewStrategy)
    threshold_profile: Literal["conservative", "nominal", "permissive"] = "nominal"
    force_limit_n: float | None = Field(default=None, gt=0, le=20)
    strain_limit: float | None = Field(default=None, gt=0, le=3)
    exposure_threshold: float | None = Field(default=None, gt=0, le=1)


def execute_experiment(config: dict, progress_callback):
    from .simulation import run_experiment

    return run_experiment(config, progress_callback=progress_callback)


def create_app(data_dir: str | Path | None = None) -> FastAPI:
    directory = Path(data_dir or os.environ.get("RETRACTION_DATA_DIR", Path(__file__).resolve().parents[2] / ".data"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = Store(directory)
        app.state.store.recover_interrupted()
        app.state.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="retraction")
        app.state.submit_lock = threading.Lock()
        initialize_physics(app, directory)
        yield
        shutdown_physics(app)
        app.state.executor.shutdown(wait=True, cancel_futures=False)

    app = FastAPI(
        title="Retraction Lab API",
        version="0.1.0",
        description="Exploratory analytic surrogate, not a clinical tool or a SOFA tissue simulation.",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # FastAPI's default handler echoes the rejected input. NaN/Infinity in
        # Python's permissive JSON parser otherwise cause a second serialization
        # failure and an HTTP 500 instead of the intended validation response.
        errors = [{key: error[key] for key in ("type", "loc", "msg") if key in error}
                  for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    def get_run(run_id: UUID, include_result=False):
        record = app.state.store.get(str(run_id), include_result=include_result)
        if record is None:
            raise HTTPException(404, "Run not found")
        return record

    def work(run_id: str, config: dict):
        store = app.state.store
        try:
            store.update(run_id, status="running", progress=0.01)

            def progress(value, *args, **kwargs):
                store.update(run_id, progress=max(0.01, min(0.99, float(value))))

            result = execute_experiment(config, progress)
            store.update(run_id, status="completed", progress=1.0, summary=result["summary"], result=result)
        except Exception:
            logger.exception("Simulation run %s failed", run_id)
            store.update(run_id, status="failed", error="Simulation failed. See the server log for details.")

    @app.get("/api/health")
    def health():
        # Read the store as well: service health requires its durable job store.
        app.state.store.active_count()
        return {"status": "ok", "version": "0.1.0", "simulator": "analytic-surrogate", "clinical_use": False}

    @app.get("/api/presets")
    def presets():
        return {
            "default_config": RunConfig().model_dump(exclude_none=True),
            "threshold_profiles": ["conservative", "nominal", "permissive"],
            "limits": {"min_patients": 40, "max_patients": 2000, "max_active_runs": 8},
            "warning": "Synthetic exploratory results. Thresholds are modeling assumptions, not established clinical limits.",
        }

    @app.post("/api/preview")
    def preview(config: PreviewConfig):
        from .simulation import THRESHOLD_PROFILES, force_probe, simulate_patient

        thresholds = dict(THRESHOLD_PROFILES[config.threshold_profile])
        for field in ("force_limit_n", "strain_limit", "exposure_threshold"):
            value = getattr(config, field)
            if value is not None:
                thresholds[field] = value
        patient = config.patient.model_dump()
        strategy = config.strategy.model_dump()
        outcome = simulate_patient(patient, strategy, thresholds)
        return {
            "engine": "analytic-surrogate", "patient": patient, "strategy": strategy,
            "thresholds": thresholds, "outcome": outcome, "probe": force_probe(patient, thresholds),
        }

    @app.get("/api/runs")
    def list_runs(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
        return {"runs": app.state.store.list(limit=limit, offset=offset)}

    @app.post("/api/runs", response_model=RunRecord, status_code=202)
    def create_run(config: RunConfig):
        with app.state.submit_lock:
            if app.state.store.active_count() >= 8:
                raise HTTPException(429, "Eight runs are already active. Wait for a run to finish.")
            record = app.state.store.create(config.model_dump(exclude_none=True))
            app.state.executor.submit(work, record["id"], record["config"])
        return record

    @app.get("/api/runs/{run_id}", response_model=RunRecord)
    def read_run(run_id: UUID):
        return get_run(run_id)

    @app.get("/api/runs/{run_id}/results")
    def results(run_id: UUID):
        record = get_run(run_id, include_result=True)
        if record["status"] != "completed":
            raise HTTPException(409, f"Run is {record['status']}; results are available after completion")
        return record["result"]

    @app.get("/api/runs/{run_id}/patients/{patient_id}/strategies")
    def patient_strategies(run_id: UUID, patient_id: str):
        """Recompute the finite grid for one saved patient without storing a second matrix."""
        from .simulation import simulate_patient

        record = get_run(run_id, include_result=True)
        if record["status"] != "completed":
            raise HTTPException(409, "Complete the run before inspecting its strategy grid")
        result = record["result"]
        patient = next((p for p in result["patients"] if p["id"] == patient_id), None)
        if patient is None:
            raise HTTPException(404, "Patient not found in this run")
        thresholds = result["thresholds"]
        return {"patient_id": patient_id, "thresholds": thresholds,
                "strategies": [simulate_patient(patient, strategy, thresholds)
                               for strategy in result["strategies"]]}

    @app.get("/api/runs/{run_id}/report")
    def report(run_id: UUID):
        from .report import render_report

        record = get_run(run_id, include_result=True)
        if record["status"] != "completed":
            raise HTTPException(409, "Complete the run before exporting a report")
        return Response(render_report(record["result"], str(run_id)), media_type="text/markdown",
                        headers={"Content-Disposition": f'attachment; filename="retraction-{run_id}.md"'})

    @app.get("/api/runs/{run_id}/export")
    def export(run_id: UUID, format: Literal["json", "csv"] = "json"):
        record = get_run(run_id, include_result=True)
        if record["status"] != "completed":
            raise HTTPException(409, "Complete the run before exporting results")
        result = record.pop("result")
        headers = {"Content-Disposition": f'attachment; filename="retraction-{run_id}.{format}"'}
        if format == "json":
            payload = json.dumps({"run": record, "results": result}, indent=2, allow_nan=False)
            return Response(payload, media_type="application/json", headers=headers)
        rows = result.get("patients", [])
        output = io.StringIO(newline="")
        keys = list(dict.fromkeys(key for row in rows for key in row))
        # Configuration identifiers make independently downloaded tables traceable.
        writer = csv.DictWriter(output, fieldnames=["run_id", "seed", *keys])
        writer.writeheader()
        for row in rows:
            flat = {key: json.dumps(value, allow_nan=False) if isinstance(value, (dict, list)) else value for key, value in row.items()}
            writer.writerow({"run_id": str(run_id), "seed": record["config"]["seed"], **flat})
        return Response(output.getvalue(), media_type="text/csv", headers=headers)

    app.include_router(physics_router)

    # API routes are registered before the static catch-all. Build the web app to
    # enable a single-process production server; development uses Vite's proxy.
    web_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if (web_dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    return app


app = create_app()
