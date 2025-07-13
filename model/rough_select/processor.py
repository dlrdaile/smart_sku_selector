from typing import Dict, List

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
import re
import os
import ast
from loguru import logger

class RoughSelectProcessor:
    def __init__(self, config:dict, prime_data_df:pd.DataFrame = None):
        self.smart_sku_df = None
        self.cur_config = dict()
        self.update_config(config)
        if prime_data_df is None:
            raise ValueError("prime_data_df cannot be None")
        self.prime_data_df:pd.DataFrame = prime_data_df
        while self.smart_sku_df is None:
            if self.smart_sku_file_path is None:
                self.smart_sku_file_path = input("请输入Smart SKU文件路径：")
            try:
                try:
                    self.smart_sku_df = pd.read_csv(self.smart_sku_file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    self.smart_sku_df = pd.read_csv(self.smart_sku_file_path, encoding='gbk')
            except Exception as e:
                logger.error(f"读取Smart SKU文件时发生错误: {e}")

        self.sku_categories,self.sku_overall_scores = self.load_smart_sku_data()
    def update_config(self,config:dict):
        """
        从配置文件中读取配置信息
        Args:
            config_file_path: 配置文件路径
        Returns:
            dict: 配置信息
        """
        if config is None:
            config = {}
        self.cur_config.update(config)
        self.smart_sku_file_path = config.get("smart_sku_file_path", None)
        self.alpha = config.get("alpha", 0.3)
        self.freq_weight = config.get("beta", 0.4)
        self.cooc_weight = config.get("gamma", 0.3)
        self.quant_weight = config.get("beta", 0.3)
        self.base_smart_ratio = config.get("base_smart_ratio", 0.5)


    def load_smart_sku_data(self, verbose=False):
        """
        读取Smart SKU数据，同时提取分类和Overall Score

        Args:
            smart_sku_file_path: Smart SKU CSV文件路径

        Returns:
            tuple: (sku_categories, sku_overall_scores)
                sku_categories: {sku: category} 其中category为'M1', 'M2', 'M3', 'M4'
                sku_overall_scores: {sku: overall_score}
        """
        if self.smart_sku_df is None:
            raise ValueError("Smart SKU数据未加载")

        sku_categories = {}
        sku_overall_scores = {}

        for idx, row in self.smart_sku_df.iterrows():
            if pd.isna(row['Item Number']):
                continue

            sku = str(row['Item Number']).strip()

            # 提取分类信息
            if not pd.isna(row['Explanation']):
                explanation = str(row['Explanation']).strip()
                if explanation.startswith('M1:'):
                    sku_categories[sku] = 'M1'
                elif explanation.startswith('M2:'):
                    sku_categories[sku] = 'M2'
                elif explanation.startswith('M3:'):
                    sku_categories[sku] = 'M3'
                elif explanation.startswith('M4:'):
                    sku_categories[sku] = 'M4'

            # 提取Overall Score
            if not pd.isna(row['Overall Score']):
                overall_score = float(row['Overall Score'])
                sku_overall_scores[sku] = overall_score
        if verbose:
            logger.info(f"Smart SKU数据加载完成，共{len(sku_categories)}个SKU分类，{len(sku_overall_scores)}个SKU评分")
            m1_count = sum(1 for cat in sku_categories.values() if cat == 'M1')
            m2_count = sum(1 for cat in sku_categories.values() if cat == 'M2')
            m3_count = sum(1 for cat in sku_categories.values() if cat == 'M3')
            m4_count = sum(1 for cat in sku_categories.values() if cat == 'M4')
            logger.info(f"M1: {m1_count}个, M2: {m2_count}个, M3: {m3_count}个, M4: {m4_count}个")
        self.sku_categories = sku_categories
        self.sku_overall_scores = sku_overall_scores
        return sku_categories, sku_overall_scores

    def cacl_sku_date_score(self,df:pd.DataFrame):
        # 初始化本天指标
        day_frequency = Counter()
        day_cooccurrence = defaultdict(lambda: defaultdict(int))
        day_quantity = defaultdict(int)
        # 初始化所有SKU集合
        all_skus = set()
        # 处理本天数据
        for idx, row in df.iterrows():

            # 提取SKU列表
            valid_skus = []
            if not pd.isna(row['sku_qty_in_cs_pairs']):
                try:
                    sku_qty_in_cs_pairs = row['sku_qty_in_cs_pairs']
                    if type(sku_qty_in_cs_pairs) == str:
                        sku_qty_in_cs_pairs = ast.literal_eval(sku_qty_in_cs_pairs)
                    sku_qty_dict = sku_qty_in_cs_pairs
                    for sku, quantity in sku_qty_dict.items():
                        sku_str = str(sku)
                        if re.match(r'^\d+$', sku_str):
                            day_quantity[sku_str] += quantity
                            day_frequency[sku_str] += 1
                            valid_skus.append(sku_str)
                except (ValueError, SyntaxError):
                    continue
            # 统计共现关系
            if len(valid_skus) > 1:
                for sku1, sku2 in combinations(valid_skus, 2):
                    day_cooccurrence[sku1][sku2] += 1
                    day_cooccurrence[sku2][sku1] += 1
            all_skus.update(valid_skus)

        # 计算本天每个SKU的指标
        day_sku_metrics = {}
        for sku in all_skus:
            frequency_score = day_frequency.get(sku, 0)

            # 计算共现分数
            cooccurrence_score = 0
            if sku in day_cooccurrence:
                cooccurrence_score = sum(day_cooccurrence[sku].values())

            quantity_score = day_quantity.get(sku, 0)

            day_sku_metrics[sku] = {
                'frequency': frequency_score,
                'cooccurrence': cooccurrence_score,
                'quantity': quantity_score
            }
        return day_sku_metrics
    def cacl_weight_decay_coefficients(self,alpha, days:List[pd.Timestamp]):
        """
        计算指数衰减系数向量
        Args:
            alpha: 指数平滑参数
            days: 日期列表
        Returns:
            list: 指数衰减系数向量
        """
        sorted_days = sorted(days, key=lambda day: day.toordinal(), reverse=True)
        last_day = sorted_days[0]
        decay_coefficients = {
            day: alpha * (1 - alpha) ** (last_day - day).days for day in sorted_days
        }
        return decay_coefficients
    def analyze_sku_data_with_date_range(self) -> pd.DataFrame:
        """
        使用指数平滑加权分析SKU数据，支持按日期范围读取数据

        Args:
            day_data_dir: 天数据目录路径
            start_file_date: 开始日期文件格式 (如: 2025_04_01)
            end_file_date: 结束日期文件格式 (如: 2025_04_30)
            alpha: 指数平滑参数
            freq_weight: 频次权重
            cooc_weight: 共现权重
            quant_weight: 数量权重
            base_smart_ratio: 基础评分与Smart评分的比例

        Returns:
            dict: {sku: score_info} SKU评分字典
        """
        # 按天分别读取和处理数据
        if self.sku_overall_scores is None:
            raise ValueError("<UNK>smart_sku_overall_scores cannot be None")
        if self.sku_categories is None:
            raise ValueError("<UNK>smart_sku_categories cannot be None")
        daily_metrics = {}  # {day: {sku: {frequency, cooccurrence, quantity}}}
        prime_data_df_grouped = self.prime_data_df.groupby('date')
        for date, _ in prime_data_df_grouped.groups.items():
            day_df = prime_data_df_grouped.get_group(date)
            day_sku_metrics = self.cacl_sku_date_score(day_df)
            # 将日期转为天维度的对象
            date = pd.to_datetime(date)
            daily_metrics[date] = day_sku_metrics

        days = sorted(daily_metrics.keys())
        decay_coefficients = self.cacl_weight_decay_coefficients(self.alpha, days)


        # 使用指数平滑加权计算每个SKU的综合指标
        logger.info(f"\n=== 计算指数平滑加权指标 ===")
        raw_metrics = defaultdict(lambda: defaultdict(int))

        for day in days:
            for sku, metrics in daily_metrics[day].items():
                raw_metrics[sku]['frequency'] += metrics['frequency'] * decay_coefficients[day]
                raw_metrics[sku]['cooccurrence'] += metrics['cooccurrence'] * decay_coefficients[day]
                raw_metrics[sku]['quantity'] += metrics['quantity'] * decay_coefficients[day]

        raw_metrics_df = pd.DataFrame(raw_metrics).T

        active_metrics_df = raw_metrics_df.query('(frequency > 0) and (cooccurrence > 0) and (quantity > 0)')
        # Z-score标准化（避免除零错误）
        active_metrics_df['frequency_zscore'] = (active_metrics_df['frequency'] - active_metrics_df['frequency'].mean()) / active_metrics_df['frequency'].std()
        active_metrics_df['cooccurrence_zscore'] = (active_metrics_df['cooccurrence'] - active_metrics_df['cooccurrence'].mean()) / active_metrics_df['cooccurrence'].std()
        active_metrics_df['quantity_zscore'] = (active_metrics_df['quantity'] - active_metrics_df['quantity'].mean()) / active_metrics_df['quantity'].std()
        active_metrics_df['base_score'] = self.freq_weight * active_metrics_df['frequency_zscore'] + self.cooc_weight * active_metrics_df['cooccurrence_zscore'] + self.quant_weight * active_metrics_df['quantity_zscore']
        active_metrics_df['smart_sku_score'] = active_metrics_df.index.map(self.sku_overall_scores)
        active_metrics_df['smart_sku_category'] = active_metrics_df.index.map(self.sku_categories)

        active_metrics_df['base_score_normalized'] = (active_metrics_df['base_score'] - active_metrics_df['base_score'].mean()) / active_metrics_df['base_score'].std()
        active_metrics_df['smart_sku_score_normalized'] = (active_metrics_df['smart_sku_score'] - active_metrics_df['smart_sku_score'].mean()) / active_metrics_df['smart_sku_score'].std()
        # 计算综合评分
        active_metrics_df['combined_score'] = self.base_smart_ratio * active_metrics_df['base_score_normalized'] + (1 - self.base_smart_ratio) * active_metrics_df['smart_sku_score_normalized']

        return active_metrics_df

