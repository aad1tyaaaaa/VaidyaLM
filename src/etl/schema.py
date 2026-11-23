"""Pydantic schema for instruction pairs. Invalid rows go to quarantine with the reason."""
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator


class Pair(BaseModel):
    id: str = Field(min_length=3)
    source: Literal["medquad", "pubmedqa_artificial", "synthetic"]
    question: str = Field(min_length=10)
    context: str
    response: str = Field(min_length=2)
    label: Literal["yes", "no", "maybe"] | None = None

    @model_validator(mode="after")
    def _consistent(self):
        if self.label and not self.response.startswith(f"Answer: {self.label}\n"):
            raise ValueError("response does not start with its label")
        if self.source != "medquad" and not self.context:
            raise ValueError("PubMedQA-style pair has no context")
        return self


def validate(pairs):
    good, bad = [], []
    for p in pairs:
        try:
            good.append(Pair(**p).model_dump())
        except ValidationError as e:
            bad.append({**p, "reason": "; ".join(err["msg"] for err in e.errors())})
    return good, bad
