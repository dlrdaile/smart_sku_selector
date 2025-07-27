from typing import List, Dict, Optional
from schema.models import SKU, SelectionResult, SystemState
import pandas as pd
from model.rough_select.processor import RoughSelectProcessor
from model.fine_select.solver import FineSelectProcessor
import numpy as np
from loguru import logger

class SkuOptimizer:
    """Represents the black-box SKU selection algorithm."""

    def __init__(self, config: dict,
                 prime_data_df: pd.DataFrame,
                 pre_state: Optional[SystemState] = None,
                 is_parallel: bool = True):
        self.config = config
        self.pre_state = pre_state
        self.rough_config = self.config.get("rough_config", {})
        self.final_config = self.config.get("fine_config", {})
        self.is_parallel = is_parallel
        if prime_data_df is None:
            raise ValueError("prime_data_df cannot be None")
        self.prime_data_df = prime_data_df
        self.rough_select_processor = RoughSelectProcessor(self.rough_config, self.prime_data_df)
        self.fine_select_processor = FineSelectProcessor(self.final_config, self.is_parallel)
        self.new_scores_df = None
        self.selection_result = None

    def score_skus(self) -> pd.DataFrame:
        """Generates scores for SKUs using exponential smoothing and new order data.

        This corresponds to 'sku打分' in the flowchart, now influenced by new orders.
        """
        return self.rough_select_processor.analyze_sku_data_with_date_range()

    def rough_selection(self, new_scores_df: pd.DataFrame) -> List[SKU]:
        """Sorts SKUs by score and selects the top N, boosting items in new orders.

        This corresponds to 'sku分数排序 (粗选品)' in the flowchart.
        """
        rough_selected_skus = []
        # 对先前的分数和当前的分数做加权处理
        if self.pre_state is not None:
            previous_score_s = self.pre_state.previous_score
            # 获取之前打分中top 100的sku
            if previous_score_s is not None:
                previous_score_s = previous_score_s.sort_values(ascending=False)
                pre_top_n = self.rough_config.get("pre_top_n", 100)
                pre_top_n_sku = previous_score_s.index[:pre_top_n].tolist()
                rough_selected_skus = rough_selected_skus + pre_top_n_sku
        self.new_scores_df = new_scores_df
        top_n = int(self.rough_config.get("top_n", 500))
        fine_tune_n = int(self.final_config.get("max_sku_count", 170))
        all_index = new_scores_df.sort_values(by="combined_score", ascending=False).index.tolist()
        top_n_skus = all_index[:top_n]
        all_index_array = np.array(all_index[top_n:])
        if fine_tune_n > top_n:
            top_n_skus = np.random.choice(all_index_array,fine_tune_n - top_n,replace=False).tolist() + top_n_skus
        rough_selected_skus = rough_selected_skus + top_n_skus
        logger.info(f"粗筛选共选中{len(set(rough_selected_skus))}个sku")
        # 去重
        rough_selected_skus = list(set(rough_selected_skus))
        return [SKU(sku) for sku in rough_selected_skus]

    def fine_selection(self, sku_data, order_df) -> Optional[SelectionResult]:
        """Optimizes for order completion rate, prioritizing new orders.

        This corresponds to '以最大化订单完成率进行优化 (细选品)' in the flowchart.
        """
        self.fine_select_processor.load_data(sku_data, order_df, None)
        # 构建模型
        self.fine_select_processor.build_model()

        # 求解模型
        result = self.fine_select_processor.solve()

        if result:
            selected_skus = self.fine_select_processor.selected_skus if self.fine_select_processor.selected_skus is not None else []
            selected_skus = [SKU(sku) for sku in selected_skus]
            self.selection_result = SelectionResult(selected_skus=selected_skus,
                                                    score=self.fine_select_processor.objective_value)
            return self.selection_result
        else:
            return None

    def get_random_select_result(self) -> Optional[SelectionResult]:
        top_n_skus = self.new_scores_df.sample(170).index.tolist()
        self.selection_result = SelectionResult(selected_skus=[SKU(sku) for sku in top_n_skus],score=0)
        return self.selection_result