from dataclasses import dataclass, field
from typing import List, Dict, Optional
import datetime
import pandas as pd


@dataclass
class SKU:
    id: str


@dataclass
class SelectionResult:
    selected_skus: List[SKU]
    score: float
    # Other metrics can be added here


@dataclass
class SystemState:
    """Represents the entire state of the system to be persisted."""
    previous_score:Optional[pd.Series] = None
    previous_score_time:Optional[datetime.datetime] = None
    previous_selection: Optional[SelectionResult] = None
    state_time: Optional[datetime.datetime] = None

if __name__ == '__main__':
    a = SelectionResult(
        selected_skus=[SKU(id='1'), SKU(id='2')],
        score=0.5,
    )
    print(a.__dict__)