"""Validated inputs for the two agents (also used as API request bodies)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

FUNCTIONS = ["Office", "Workstation", "Research Lab", "Teaching Lab", "Meeting", "Support & Storage"]


class SpaceRequest(BaseModel):
    request_type: Literal["new_office", "department_function", "change_assignment",
                          "vacancy_review", "temporary_relocation"] = "new_office"
    department: str | None = Field(None, max_length=120)
    function: str = "Office"
    min_area: float | None = Field(None, ge=0, le=50000)
    max_area: float | None = Field(None, ge=0, le=50000)
    seats: int = Field(1, ge=1, le=200)
    preferred_buildings: list[str] = Field(default_factory=list, max_length=12)
    exclude_buildings: list[str] = Field(default_factory=list, max_length=12)
    netid: str | None = Field(None, max_length=20)
    near: str | None = Field(None, max_length=120)      # person (name or NetID) whose office the space should be close to
    lab_requirements: list[str] = Field(default_factory=list, max_length=12)
    notes: str = Field("", max_length=2000)

    @field_validator("function")
    @classmethod
    def _known_function(cls, v: str) -> str:
        if v not in FUNCTIONS:
            raise ValueError(f"function must be one of {FUNCTIONS}")
        return v


class OutageRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=2_000_000)
    source: str = Field("pasted", max_length=200)
