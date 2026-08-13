from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    id: str
    name: str
    task: Literal["classification", "regression"]
    target_col: str
    row_count: int
    column_count: int
    description: str = ""


class GeneratorSummary(BaseModel):
    id: str
    name: str
    family: str
    description: str
    available: bool = True
    reason: str | None = None
    auto_setup: bool = False


class GenerateRequest(BaseModel):
    dataset_id: str
    generator_ids: list[str] = Field(min_length=1)
    n_samples: int = Field(default=1000, ge=10, le=10000)
    seed: int = Field(default=42, ge=0)


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    dataset_id: str
    generator_ids: list[str]
    n_samples: int
    seed: int
    progress: float = 0.0
    message: str = ""
    results: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class PreviewResponse(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int
    quality_score: float | None = None
