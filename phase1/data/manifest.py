"""Dataclasses and JSONL I/O for segments, variants and triplets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, TypeVar

T = TypeVar("T")


@dataclass
class Segment:
    segment_id: str
    source_id: str
    source_path: str
    start: float
    end: float
    sample_rate: int
    path: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Variant:
    variant_id: str
    segment_id: str
    transform: str
    params: dict[str, Any]
    path: str
    sample_rate: int


@dataclass
class Triplet:
    triplet_id: str
    anchor_id: str
    positive_id: str
    negative_id: str
    kind: str
    meta: dict[str, Any] = field(default_factory=dict)


def _dict_of(obj: Any) -> dict[str, Any]:
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


def write_jsonl(path: str | Path, items: Iterable[Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            payload = _dict_of(item) if hasattr(item, "__dataclass_fields__") else item
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path, cls: type[T] | None = None) -> list[Any]:
    path = Path(path)
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(cls(**row) if cls is not None else row)
    return rows


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def as_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj) if hasattr(obj, "__dataclass_fields__") else dict(obj)
