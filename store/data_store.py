import json
import os
from typing import Optional
from schema.models import SystemState, SKU, SelectionResult
from config.config import DATA_STORAGE_PATH

class DataStore:
    @staticmethod
    def save_state(state: SystemState):
        """Saves the system state to a JSON file."""
        os.makedirs(os.path.dirname(DATA_STORAGE_PATH), exist_ok=True)
        with open(DATA_STORAGE_PATH, 'w') as f:
            # A custom encoder is needed for dataclasses
            json.dump(state, f, indent=4, cls=EnhancedJSONEncoder)

    @staticmethod
    def load_state() -> Optional[SystemState]:
        """Loads the system state from a JSON file."""
        if not os.path.exists(DATA_STORAGE_PATH):
            return None
        with open(DATA_STORAGE_PATH, 'r') as f:
            try:
                data = json.load(f)
                # Reconstruct the dataclasses from the dict
                all_skus = [SKU(**sku_data) for sku_data in data.get('all_skus', [])]
                previous_scores = data.get('previous_scores', {})
                
                previous_selection_data = data.get('previous_selection')
                previous_selection = None
                if previous_selection_data:
                    selected_skus = [SKU(**sku_data) for sku_data in previous_selection_data.get('selected_skus', [])]
                    previous_selection = SelectionResult(
                        selected_skus=selected_skus,
                        profit=previous_selection_data.get('profit', 0.0)
                    )

                return SystemState(
                    all_skus=all_skus,
                    previous_scores=previous_scores,
                    previous_selection=previous_selection
                )
            except (json.JSONDecodeError, TypeError):
                return None

class EnhancedJSONEncoder(json.JSONEncoder):
    """A custom JSON encoder for our dataclasses."""
    def default(self, o):
        if hasattr(o, '__dict__'):
            return o.__dict__
        return super().default(o)