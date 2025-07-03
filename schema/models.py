from dataclasses import dataclass, field
from typing import List, Dict
import datetime


@dataclass
class SKU:
    id: str
    name: str


@dataclass
class Order:
    """Represents a single incoming customer order."""
    id: str
    items: Dict[str, int]  # SKU ID -> quantity
    created_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)


@dataclass
class SelectionResult:
    selected_skus: List[SKU]
    profit: float = 0.0
    # Other metrics can be added here


@dataclass
class SystemState:
    """Represents the entire state of the system to be persisted."""
    all_skus: List[SKU] = field(default_factory=list)
    previous_scores: Dict[str, float] = field(default_factory=dict)  # SKU ID -> score
    previous_selection: SelectionResult = None