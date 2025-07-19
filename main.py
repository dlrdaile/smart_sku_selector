import datetime
import time
from typing import Optional
from pathlib import Path

import pandas as pd
import ray

from store.data_store import DataStore
from schema.models import SystemState
from model.sku_optimizer import SkuOptimizer
from model.evaluator import Evaluator
from model.pre_process.processor import PrimerDataPreProcess
from loguru import logger
from config.config import settings
from copy import deepcopy
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

ray_is_initialized = False


def get_config(config) -> dict:
    default_config = deepcopy(settings.SKU_OPTIMIZER_CONFIG)
    if default_config is None:
        default_config = {}
    if "rough_config" not in default_config:
        default_config["rough_config"] = {}
    if "fine_config" not in default_config:
        default_config["fine_config"] = {}
    if type(config) == dict:
        for key, value in default_config.items():
            if key.startswith("rough_config_"):
                actual_key = key.replace("rough_config_", "")
                default_config["rough_config"][actual_key] = value
            elif key.startswith("fine_config_"):
                actual_key = key.replace("fine_config_", "")
                default_config["fine_config"][actual_key] = value
            else:
                default_config[key] = value
    return default_config


def run_once_optimization(config, pre_processor, pre_state) -> SkuOptimizer:
    """Runs a sweep of the SKU optimization process."""
    optimizer_config = get_config(config)

    logger.info("\n--- Running Optimization Cycle ---")
    # 1. SKU Scoring (now with new order data)
    logger.info("1. Scoring SKUs...")
    optimizer = SkuOptimizer(optimizer_config,
                             pre_processor.primer_data_df,
                             pre_state=pre_state,
                             is_parallel=not settings.IS_DEBUG)

    new_scores_df = optimizer.score_skus()

    # 2. Rough Selection
    logger.info("2. Performing rough selection...")
    rough_selected_skus = optimizer.rough_selection(new_scores_df)

    # 3. Fine Selection
    logger.info("3. Performing fine selection...")
    _ = optimizer.fine_selection(rough_selected_skus, pre_processor.fine_tune_structure_df)
    return optimizer


def tune_once_optimization(config, pre_processor, pre_state):
    """Runs a sweep of the SKU optimization process."""
    optimizer = run_once_optimization(config, pre_processor, pre_state)
    if optimizer.selection_result is None:
        return None
    return {"max_score": optimizer.selection_result.score}


def run_optimization_cycle(state: SystemState, pre_processor: PrimerDataPreProcess) -> Optional[SystemState]:
    """Runs one full cycle of the SKU optimization process."""
    default_config = deepcopy(settings.SKU_OPTIMIZER_CONFIG)
    state = deepcopy(state)
    # 如果非debug模式，则开启ray超参数搜索
    # settings.IS_DEBUG = True
    if not settings.IS_DEBUG:
        ray.init(include_dashboard=False)
        tuner = tune.Tuner(
            lambda config: tune_once_optimization(config,pre_processor,state),
            tune_config=tune.TuneConfig(
                num_samples=20, # 搜寻的参数组合数量
                max_concurrent_trials = 5,
                reuse_actors = True,
                scheduler=ASHAScheduler(metric="max_score", mode="max"), # best config 参考指标
                trial_dirname_creator= lambda trial: f"{trial.trial_id}"
            ),
            param_space=settings.SEARCH_SPACE, # 搜索空间配置
        )
        results = tuner.fit()
        best_result = results.get_best_result("max_score", mode="max")
        logger.info("\n--- Finished Optimization Cycle ---")
        logger.info(f"best config: {best_result.config}")

        best_config = best_result.config
    else:
        best_config = None
    # 3. New Selection
    optimizer = run_once_optimization(best_config, pre_processor, state)
    new_selection_result = optimizer.selection_result
    if new_selection_result is None:
        return None
    state.previous_score = optimizer.new_scores_df['combined_score']
    state.previous_score_time = pre_processor.last_date
    # 4. Evaluation
    evaluate_config = default_config.get("evaluate_config", {})
    evaluator = Evaluator(pre_processor.fine_tune_structure_df, evaluate_config)
    now_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    old_profit, is_full_load = evaluator.calc_actual_complete_rate(state.previous_selection)
    # 当模拟仿真全部满负载时，说明旧的选品方案已经达到仓库运行的上限了，不需要再优化了
    if is_full_load:
        logger.info("旧的选品方案已经达到仓库运行的上限了，不需要再优化了")
        return state

    logger.info("Evaluating new selection...")
    new_profit, is_full_load = evaluator.simulate_warehouse_efficiency(new_selection_result, "new", prefix=now_time_str)
    changeover_cost = evaluator.calculate_changeover_cost(state.previous_selection, new_selection_result)
    new_selection_result.profit = new_profit

    logger.info(f"   - Old Selection Profit: {old_profit:.3f}")
    logger.info(f"   - New Selection Gross Profit: {new_profit:.3f}")
    logger.info(f"   - Changeover Cost: {changeover_cost:.5f}")

    # 5. Decision
    logger.info("5. Making decision...")
    if evaluator.should_switch(old_profit, new_profit, changeover_cost):
        logger.info("   -> Decision: Switch to the new selection.")
        state.previous_selection = new_selection_result
    else:
        logger.info("   -> Decision: Keep the old selection.")

    # 6. Update and Save State
    logger.info("6. Saving state...")
    logger.info("--- Optimization Cycle Complete ---")
    return state


def main():
    """Main loop to listen for new orders and trigger optimization."""
    sku_master_data_path: Optional[Path] = Path(settings.SKU_MASTER_DATA_PATH)
    logger.info("System is running. Waiting for new orders...")
    time.sleep(1)
    try:
        while True:
            new_data_path = input("new_data_path: ")
            # Simulate the arrival of new orders
            data_path = Path(new_data_path)
            if not data_path.exists():
                logger.error(f"文件路径不存在")
                continue
            primer_data_path_list = list(data_path.rglob('*.csv'))
            if len(primer_data_path_list) == 0:
                logger.warning(f"没有有效数据文件")
                continue
            try:
                pre_processor = PrimerDataPreProcess(primer_data_path_list, sku_master_data_path, not settings.IS_DEBUG)
                pre_processor.run()
                # Trigger the optimization cycle
                state = DataStore.load_state()
                state = run_optimization_cycle(state, pre_processor)
                if state is not None:
                    DataStore.save_state(state)
            except Exception as e:
                if e is KeyboardInterrupt:
                    raise e
                logger.error(f"{e}")
                continue
    except KeyboardInterrupt:
        logger.info("\nSystem shutting down.")


if __name__ == "__main__":
    from warnings import filterwarnings
    from pandarallel import pandarallel

    filterwarnings('ignore')
    if not settings.IS_DEBUG:
        pandarallel.initialize(progress_bar=True)
    main()
