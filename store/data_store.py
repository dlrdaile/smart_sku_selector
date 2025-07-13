import datetime
import json
import os
from typing import Optional
from schema.models import SystemState, SKU, SelectionResult
from config.config import settings
import pandas as pd
from loguru import logger
from pathlib import Path

class DataStore:
    @staticmethod
    def save_state(state: SystemState):
        logger.info(f"start saving state")
        state.state_time = datetime.datetime.now()
        """Saves the system state to a JSON file."""
        os.makedirs(os.path.dirname(settings.DATA_STORAGE_PATH), exist_ok=True)
        state_dict = {}
        if state.state_time is not None:
            state_dict["state_time"] = state.state_time.isoformat()
        if state.previous_score_time is not None:
            state_dict["previous_score_time"] = state.previous_score_time.isoformat()
        if state.previous_score is not None:
            state_dict["previous_score"] = state.previous_score.to_dict()
        if state.previous_selection is not None:
            state_dict["previous_selection"] = {
                "selected_skus": [sku.__dict__ for sku in state.previous_selection.selected_skus],
                "score": state.previous_selection.score
            }
        # 归档之前的state
        pre_state_path = Path(settings.DATA_STORAGE_PATH)
        if pre_state_path.exists() and pre_state_path.is_file():
            archive_dir_path = Path(settings.DATA_STORAGE_ARCHIVE_PATH)
            archive_dir_path.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir_path / f"state_{state.state_time.strftime('%Y_%m_%d_%H_%M_%S')}.json"
            pre_state_path.rename(archive_path)
            logger.info(f"old state archive path: {archive_path}")

        with open(settings.DATA_STORAGE_PATH, 'w') as f:
            # A custom encoder is needed for dataclasses
            json.dump(state_dict, f, indent=4)

        logger.info(f"successfully save new state to {settings.DATA_STORAGE_PATH}")

    @staticmethod
    def load_state() -> Optional[SystemState]:
        """Loads the system state from a JSON file."""
        logger.info(f"start loading state from {settings.DATA_STORAGE_PATH}")
        if os.path.exists(settings.DATA_STORAGE_PATH):
            with open(settings.DATA_STORAGE_PATH, 'r') as f:
                try:
                    data = json.load(f)
                    # Reconstruct the dataclasses from the dict
                    previous_scores = data.get('previous_score', {})
                    previous_scores_s = None
                    previous_scores_time = None
                    if previous_scores != {}:
                        previous_scores_s = pd.Series(previous_scores)

                    if data.get('previous_score_time'):
                        previous_scores_time = pd.to_datetime(data['previous_score_time'])

                    previous_selection_data = data.get('previous_selection')
                    previous_selection = None
                    if previous_selection_data:
                        selected_skus = [SKU(**sku_data) for sku_data in previous_selection_data.get('selected_skus', [])]
                        previous_selection = SelectionResult(
                            selected_skus=selected_skus,
                            score=previous_selection_data.get('score', 0),
                        )
                    state_time = None
                    if data.get('state_time'):
                        state_time = pd.to_datetime(data.get('state_time'))
                    logger.info(f"successfully load state from {settings.DATA_STORAGE_PATH}")
                    return SystemState(
                        previous_selection=previous_selection,
                        previous_score_time=previous_scores_time,
                        previous_score=previous_scores_s,
                        state_time=state_time
                    )
                except Exception as e:
                    logger.warning(f"Failed to load state from {settings.DATA_STORAGE_PATH}: {e}")

        return SystemState()

