# Configuration for the SKU selection optimization
# Path for the data storage file
from ray import tune


class Config:
    # Configuration for the SKU selection optimization
    SKU_MASTER_DATA_PATH: str = r"D:\Code\Python\sku_select_optimize\data\raw_data\SKU_MD_0509.xlsx"

    SKU_OPTIMIZER_CONFIG: dict = {
        "rough_config": {
            "smart_sku_file_path" : r"D:\Code\Python\sku_select_optimize\data\raw_data\2025 618\4567月Smart SKU带Overall Score.csv", # smart sku分类结果文件路径
            "alpha" : 0.1, # 指数平滑参数
            "freq_weight" : 0.4, # 频次权重
            "cooc_weight" : 0.3, # 共现权重
            "quant_weight" : 0.3, # 数量权重
            "base_smart_ratio" : 0.5, # 基础评分与Smart评分的比例
            "top_n" : 1000, # 前top_n个sku
        },
        "fine_config": {
            'max_sku_count': 170,  # 最大选品数量N
            'weight': 0.5,  # 优化目标权重W
            'time_limit': 600,  # 求解时间限制(秒)
            'mip_gap': 0.01,  # MIP Gap
            'pallet_count': 170,  # 托盘数量
            'output_dir': 'output'  # 输出目录
        },
        "evaluate_config": {
            "qty_comp_weight": 0.6, # 计算订单完成率时品类完成率和箱数完成率的加权系数
            "switch_sku_time_use": 20, # 切换单个sku的时间成本，单位分钟
            "switch_sku_parallel": 8, # 切换sku过程的执行并发度
            "simulate_order_data_cache_dir_path": r"D:\Code\Python\sku_select_optimize\data\fulfillable_parts", # 提供给仿真的运行订单数据缓存目录
        }
    }

    SEARCH_SPACE:dict = {

        # rough selection
        "rough_config_alpha": tune.uniform(0.1, 0.5),
        "rough_config_freq_weight": tune.uniform(0.2, 0.5),
        "rough_config_cooc_weight": tune.uniform(0.2, 0.5),
        "rough_config_quant_weight": tune.uniform(0.2, 0.5),
        "rough_config_base_smart_ratio": tune.uniform(0.4, 0.6),

        # fine selection

    }

    IS_DEBUG = False

    DATA_STORAGE_PATH = "./data/storage.json"
    # 归档路径
    DATA_STORAGE_ARCHIVE_PATH = "./data/storage_archive"

settings = Config()
