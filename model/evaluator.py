from typing import Optional

import pandas as pd
from loguru import logger
from schema.models import SelectionResult
import random


class Evaluator:
    """Evaluates the profitability and costs of selection results."""

    def __init__(self, order_df, config):
        self.order_df = order_df
        self.date_num = self.order_df['date'].unique().shape[0]
        self.load_config(config)

    def load_config(self, config):
        self.config = config
        self.qty_comp_weight = config.get('qty_comp_weight', 0.5)
        self.switch_sku_time_use = config.get('switch_sku_time_use', 30)
        self.switch_sku_parallel = config.get('switch_sku_parallel', 10)

    def calculate_changeover_cost(self, old_selection: Optional[SelectionResult],
                                  new_selection: SelectionResult) -> float:
        """Calculates the cost of changing from the old selection to the new one.

        This corresponds to '换品成本' in the flowchart.
        """

        # 找出替换的sku
        def find_replaced_skus(old_selection, new_selection):
            new_sku_ids = set(sku.id for sku in new_selection.selected_skus)
            old_sku_ids = set(sku.id for sku in
                              old_selection.selected_skus) if old_selection is not None and old_selection.selected_skus is not None else set()
            same_skus = new_sku_ids & old_sku_ids
            new_change_skus = new_sku_ids - same_skus
            return new_change_skus

        replaced_skus = find_replaced_skus(old_selection, new_selection)
        # 计算换品成本
        changeover_cost = len(replaced_skus) * self.switch_sku_time_use / 3600 / self.switch_sku_parallel  # 假设一个换品需要30分钟，并发度是5
        return changeover_cost / self.date_num

    def simulate_warehouse_efficiency(self, selection: SelectionResult) -> float:
        """
        Calculates warehouse efficiency by simulating order fulfillment.

        This method evaluates the effectiveness of an SKU selection by simulating
        the order completion process. It calculates quantity and category (SKU)
        completion rates for each affected order, aggregates these into daily
        averages, and then computes a final weighted efficiency score.

        This corresponds to '优化仓库布局和设备执行效率' in the flowchart.
        """
        if selection is None or not selection.selected_skus:
            return 0.0

        selected_sku_ids = {sku.id for sku in selection.selected_skus}

        # Identify orders affected by the current SKU selection.
        # An order is affected if it contains at least one of the selected SKUs.
        affected_order_ids = self.order_df[self.order_df['sku_id'].isin(selected_sku_ids)]['order_id'].unique()
        if len(affected_order_ids) == 0:
            return 0.0

        original_affected_orders = self.order_df[self.order_df['order_id'].isin(affected_order_ids)].copy()

        # 1. Calculate the original total quantity and SKU variety for each affected order.
        # This serves as the baseline for calculating completion rates.
        original_order_agg = original_affected_orders.groupby('order_id').agg(
            original_quantity=('quantity', 'sum'),
            original_sku_count=('sku_id', 'nunique'),
            date=('date', 'first')
        ).reset_index()

        # 2. Determine the parts of the orders that can be fulfilled with the selected SKUs.
        fulfillable_parts = original_affected_orders[original_affected_orders['sku_id'].isin(selected_sku_ids)]

        # 3. Simulate the completion of these fulfillable parts.
        simulated_complete_df = self.simulate_order_completion(fulfillable_parts)

        # 4. Aggregate the simulated completion results by order.
        if not simulated_complete_df.empty:
            simulated_agg = simulated_complete_df.groupby('order_id').agg(
                completed_quantity=('quantity', 'sum'),
                completed_sku_count=('sku_id', 'nunique')
            ).reset_index()
            # 5. Merge original order data with simulated completion data.
            completion_df = pd.merge(original_order_agg, simulated_agg, on='order_id', how='left')
        else:
            completion_df = original_order_agg.copy()
            completion_df['completed_quantity'] = 0
            completion_df['completed_sku_count'] = 0

        # For orders that had fulfillable parts but nothing was completed in the simulation, fill NaNs with 0.
        completion_df[['completed_quantity', 'completed_sku_count']] = completion_df[
            ['completed_quantity', 'completed_sku_count']].fillna(0)

        # 6. Calculate quantity and SKU completion rates for each order.
        # Avoid division by zero, though original_quantity should not be zero.
        completion_df['quantity_completion'] = completion_df['completed_quantity'] / completion_df['original_quantity']
        completion_df['sku_completion'] = completion_df['completed_sku_count'] / completion_df['original_sku_count']

        # 7. Aggregate the completion rates by day to get daily averages.
        daily_completion = completion_df.groupby('date').agg(
            avg_quantity_completion=('quantity_completion', 'mean'),
            avg_sku_completion=('sku_completion', 'mean')
        ).reset_index()

        # 8. Calculate the final score as the mean of daily average completions, weighted.
        final_quantity_completion = daily_completion['avg_quantity_completion'].mean()
        final_sku_completion = daily_completion['avg_sku_completion'].mean()

        return final_quantity_completion * self.qty_comp_weight + final_sku_completion * (1 - self.qty_comp_weight)

    def should_switch(self, old_profit: float, new_profit: float, changeover_cost: float) -> bool:
        """Decides if the new selection should be adopted.

        This corresponds to '谁收益更大' in the flowchart.
        """
        net_new_profit = new_profit / (1 + changeover_cost)
        logger.info(f"Old Profit: {old_profit:.3f}, New Profit (Net): {net_new_profit:.3f}")
        return net_new_profit > old_profit

    def simulate_order_completion(self, selected_orders: pd.DataFrame) -> pd.DataFrame:
        """
        Simulates the order completion process.

        This is a placeholder implementation. In a real-world scenario, this
        would involve a more complex simulation engine. Here, we simulate that
        a random 95% of the selected order lines are completed successfully.
        """
        if selected_orders.empty:
            return selected_orders.copy()
        # Simulate by randomly "completing" a fraction of the order lines
        return selected_orders.sample(frac=0.95, random_state=42)
