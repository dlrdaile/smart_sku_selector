#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
指数平滑加权超参数更新及选品程序
基于指数平滑加权版SKU评分算法，结合超参数优化功能
使用网格搜索和随机搜索混合优化策略，选出最优超参数并生成前220个SKU
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
import re
import os
from datetime import datetime
from sklearn.model_selection import ParameterGrid
import shutil
import random
import ast

def load_smart_sku_data(smart_sku_file_path):
    """
    读取Smart SKU数据，同时提取分类和Overall Score
    
    Args:
        smart_sku_file_path: Smart SKU CSV文件路径
    
    Returns:
        tuple: (sku_categories, sku_overall_scores)
            sku_categories: {sku: category} 其中category为'M1', 'M2', 'M3', 'M4'
            sku_overall_scores: {sku: overall_score}
    """
    print("正在读取Smart SKU数据和Overall Score...")
    try:
        smart_df = pd.read_csv(smart_sku_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        smart_df = pd.read_csv(smart_sku_file_path, encoding='gbk')
    
    sku_categories = {}
    sku_overall_scores = {}
    
    for idx, row in smart_df.iterrows():
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
    
    print(f"Smart SKU数据加载完成，共{len(sku_categories)}个SKU分类，{len(sku_overall_scores)}个SKU评分")
    m1_count = sum(1 for cat in sku_categories.values() if cat == 'M1')
    m2_count = sum(1 for cat in sku_categories.values() if cat == 'M2')
    m3_count = sum(1 for cat in sku_categories.values() if cat == 'M3')
    m4_count = sum(1 for cat in sku_categories.values() if cat == 'M4')
    print(f"M1: {m1_count}个, M2: {m2_count}个, M3: {m3_count}个, M4: {m4_count}个")
    
    return sku_categories, sku_overall_scores

def analyze_sku_data_with_date_range(day_data_dir, start_file_date, end_file_date, smart_sku_categories=None, smart_sku_overall_scores=None, alpha=0.3, freq_weight=0.4, cooc_weight=0.3, quant_weight=0.3, base_smart_ratio=0.5, verbose=True):
    """
    使用指数平滑加权分析SKU数据，支持按日期范围读取数据
    
    Args:
        day_data_dir: 天数据目录路径
        start_file_date: 开始日期文件格式 (如: 2025_04_01)
        end_file_date: 结束日期文件格式 (如: 2025_04_30)
        smart_sku_categories: Smart SKU分类字典
        smart_sku_overall_scores: Smart SKU Overall Score字典
        alpha: 指数平滑参数
        freq_weight: 频次权重
        cooc_weight: 共现权重
        quant_weight: 数量权重
        base_smart_ratio: 基础评分与Smart评分的比例
    
    Returns:
        dict: {sku: score_info} SKU评分字典
    """
    
    if verbose:
        print(f"\n=== 开始指数平滑加权分析（alpha={alpha}）===")
        print(f"分析日期范围: {start_file_date} 到 {end_file_date}")
        print(f"权重参数: 频次={freq_weight:.3f}, 共现={cooc_weight:.3f}, 数量={quant_weight:.3f}")
        print(f"基础/Smart比例: {base_smart_ratio:.3f}")
    
    # 按天分别读取和处理数据
    daily_metrics = {}  # {day: {sku: {frequency, cooccurrence, quantity}}}
    all_skus = set()  # 所有出现过的SKU集合
    
    # 获取指定日期范围内的文件
    selected_files = []
    for file in os.listdir(day_data_dir):
        if file.startswith('day_') and file.endswith('.csv'):
            # 提取文件中的日期部分 (day_2025_04_01.csv -> 2025_04_01)
            file_date = file.replace('day_', '').replace('.csv', '')
            if start_file_date <= file_date <= end_file_date:
                selected_files.append(file)
    
    selected_files.sort()  # 按文件名排序
    
    for i, day_file in enumerate(selected_files):
        file_date = day_file.replace('day_', '').replace('.csv', '')
        if verbose:
            print(f"\n正在处理日期 {file_date} 数据: {day_file}...")
        
        day_file_path = os.path.join(day_data_dir, day_file)
        if not os.path.exists(day_file_path):
            if verbose:
                print(f"  警告: 日期 {file_date} 数据文件不存在: {day_file_path}")
            continue
            
        try:
            day_df = pd.read_csv(day_file_path, encoding='utf-8')
        except UnicodeDecodeError:
            day_df = pd.read_csv(day_file_path, encoding='gbk')
        
        if verbose:
            print(f"  日期 {file_date} 数据形状: {day_df.shape}")
        
        # 初始化本天指标
        day_frequency = Counter()
        day_cooccurrence = defaultdict(lambda: defaultdict(int))
        day_quantity = defaultdict(int)
        
        # 处理本天数据
        for idx, row in day_df.iterrows():
            if verbose and idx % 5000 == 0 and idx > 0:
                print(f"    已处理日期 {file_date} {idx} 行数据...")
            
            # 提取SKU列表
            valid_skus = []
            if not pd.isna(row['sku_set']):
                try:
                    sku_list = ast.literal_eval(row['sku_set'])
                    for sku in sku_list:
                        sku_str = str(sku)
                        if re.match(r'^\d+$', sku_str):
                            valid_skus.append(sku_str)
                            all_skus.add(sku_str)
                except (ValueError, SyntaxError):
                    continue
            
            # 统计频次
            for sku in valid_skus:
                day_frequency[sku] += 1
            
            # 统计共现关系
            if len(valid_skus) > 1:
                for sku1, sku2 in combinations(valid_skus, 2):
                    day_cooccurrence[sku1][sku2] += 1
                    day_cooccurrence[sku2][sku1] += 1
            
            # 统计数量
            if not pd.isna(row['sku_qty_in_cs_pairs']):
                try:
                    sku_qty_dict = ast.literal_eval(row['sku_qty_in_cs_pairs'])
                    for sku, quantity in sku_qty_dict.items():
                        sku_str = str(sku)
                        if re.match(r'^\d+$', sku_str):
                            day_quantity[sku_str] += quantity
                except (ValueError, SyntaxError):
                    continue
        
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
        
        daily_metrics[file_date] = day_sku_metrics
        if verbose:
            print(f"  日期 {file_date} 处理完成，发现{len([sku for sku, metrics in day_sku_metrics.items() if metrics['frequency'] > 0])}个活跃SKU")
    
    if verbose:
        print(f"\n总共发现 {len(all_skus)} 个不同的SKU")
        print(f"分析天数: {list(daily_metrics.keys())}")
    
    # 计算指数平滑权重
    days = sorted(daily_metrics.keys())
    num_days = len(days)
    
    if verbose:
        print(f"\n=== 计算指数平滑权重 ===")
        print(f"使用指数平滑参数 alpha = {alpha}")
    
    # 计算每天的权重（越后面的天权重越高）
    weights = {}
    total_weight = 0
    
    for i, day in enumerate(days):
        # 指数平滑权重：后面的天权重更高
        if i == num_days - 1:  # 最后一天
            weight = 1 - sum(weights.values()) if weights else alpha
        else:
            weight = alpha * ((1 - alpha) ** (num_days - 1 - i))
        
        weights[day] = weight
        total_weight += weight
        if verbose:
            print(f"  日期 {day} 权重: {weight:.4f}")
    
    # 归一化权重
    for day in weights:
        weights[day] = weights[day] / total_weight
        if verbose:
            print(f"  日期 {day} 归一化权重: {weights[day]:.4f}")
    
    # 使用指数平滑加权计算每个SKU的综合指标
    print(f"\n=== 计算指数平滑加权指标 ===")
    weighted_metrics = {}
    
    for sku in all_skus:
        weighted_frequency = 0
        weighted_cooccurrence = 0
        weighted_quantity = 0
        
        for day in days:
            if sku in daily_metrics[day]:
                day_weight = weights[day]
                metrics = daily_metrics[day][sku]
                
                weighted_frequency += metrics['frequency'] * day_weight
                weighted_cooccurrence += metrics['cooccurrence'] * day_weight
                weighted_quantity += metrics['quantity'] * day_weight
        
        weighted_metrics[sku] = {
            'frequency': weighted_frequency,
            'cooccurrence': weighted_cooccurrence,
            'quantity': weighted_quantity
        }
    
    # 过滤掉完全没有活动的SKU
    active_skus = {sku: metrics for sku, metrics in weighted_metrics.items() 
                   if metrics['frequency'] > 0 or metrics['cooccurrence'] > 0 or metrics['quantity'] > 0}
    
    if verbose:
        print(f"过滤后活跃SKU数量: {len(active_skus)}")
    
    # 收集所有SKU的加权指标数据
    if verbose:
        print("\n正在收集指数平滑加权后的指标数据...")
    raw_metrics = {}
    for sku in active_skus:
        raw_metrics[sku] = {
            'frequency': weighted_metrics[sku]['frequency'],
            'cooccurrence': weighted_metrics[sku]['cooccurrence'],
            'quantity': weighted_metrics[sku]['quantity']
        }
    
    # 数据标准化处理（Z-score标准化）
    if verbose:
        print("正在进行数据标准化处理...")
    
    # 提取所有指标的数值
    frequencies = [metrics['frequency'] for metrics in raw_metrics.values()]
    cooccurrences = [metrics['cooccurrence'] for metrics in raw_metrics.values()]
    quantities = [metrics['quantity'] for metrics in raw_metrics.values()]
    
    # 计算均值和标准差
    freq_mean, freq_std = np.mean(frequencies), np.std(frequencies)
    cooccur_mean, cooccur_std = np.mean(cooccurrences), np.std(cooccurrences)
    quantity_mean, quantity_std = np.mean(quantities), np.std(quantities)
    
    if verbose:
        print(f"标准化前数据统计:")
        print(f"频次 - 均值: {freq_mean:.2f}, 标准差: {freq_std:.2f}, 范围: {min(frequencies):.2f}-{max(frequencies):.2f}")
        print(f"共现 - 均值: {cooccur_mean:.2f}, 标准差: {cooccur_std:.2f}, 范围: {min(cooccurrences):.2f}-{max(cooccurrences):.2f}")
        print(f"数量 - 均值: {quantity_mean:.2f}, 标准差: {quantity_std:.2f}, 范围: {min(quantities):.2f}-{max(quantities):.2f}")
    
    # 计算标准化后的评分
    sku_scores = {}
    for sku in active_skus:
        # 获取原始指标
        frequency_score = raw_metrics[sku]['frequency']
        cooccurrence_score = raw_metrics[sku]['cooccurrence']
        quantity_score = raw_metrics[sku]['quantity']
        
        # Z-score标准化（避免除零错误）
        freq_normalized = (frequency_score - freq_mean) / max(freq_std, 1e-8)
        cooccur_normalized = (cooccurrence_score - cooccur_mean) / max(cooccur_std, 1e-8)
        quantity_normalized = (quantity_score - quantity_mean) / max(quantity_std, 1e-8)
        
        # 基础综合评分公式（使用自定义权重）
        base_score = (
            freq_weight * freq_normalized + 
            cooc_weight * cooccur_normalized + 
            quant_weight * quantity_normalized
        )
        
        # 获取Smart SKU分类
        smart_category = 'Unknown'
        if smart_sku_categories and sku in smart_sku_categories:
            smart_category = smart_sku_categories[sku]
        
        # 获取Smart SKU Overall Score
        smart_sku_score = 0.0
        if smart_sku_overall_scores and sku in smart_sku_overall_scores:
            smart_sku_score = smart_sku_overall_scores[sku]
        
        sku_scores[sku] = {
            'frequency': frequency_score,
            'cooccurrence_total': cooccurrence_score,
            'quantity': quantity_score,
            'freq_normalized': freq_normalized,
            'cooccur_normalized': cooccur_normalized,
            'quantity_normalized': quantity_normalized,
            'base_score': base_score,
            'smart_category': smart_category,
            'smart_sku_score': smart_sku_score
        }
    
    # 对base_score和smart_sku_score进行标准化处理并计算综合评分
    if verbose:
        print("正在计算综合评分...")
    
    # 提取所有base_score和smart_sku_score
    base_scores = [scores['base_score'] for scores in sku_scores.values()]
    smart_sku_scores = [scores['smart_sku_score'] for scores in sku_scores.values()]
    
    # 计算均值和标准差
    base_score_mean, base_score_std = np.mean(base_scores), np.std(base_scores)
    smart_score_mean, smart_score_std = np.mean(smart_sku_scores), np.std(smart_sku_scores)
    
    if verbose:
        print(f"Base Score - 均值: {base_score_mean:.4f}, 标准差: {base_score_std:.4f}")
        print(f"Smart SKU Score - 均值: {smart_score_mean:.6f}, 标准差: {smart_score_std:.6f}")
    
    # 为每个SKU计算标准化后的综合评分
    for sku in sku_scores:
        base_score = sku_scores[sku]['base_score']
        smart_sku_score = sku_scores[sku]['smart_sku_score']
        
        # 标准化处理（避免除零错误）
        base_score_normalized = (base_score - base_score_mean) / max(base_score_std, 1e-8)
        smart_score_normalized = (smart_sku_score - smart_score_mean) / max(smart_score_std, 1e-8)
        
        # 综合评分：使用自定义比例结合两个标准化后的评分
        combined_score = base_smart_ratio * base_score_normalized + (1 - base_smart_ratio) * smart_score_normalized
        
        # 更新sku_scores
        sku_scores[sku]['base_score_normalized'] = base_score_normalized
        sku_scores[sku]['smart_score_normalized'] = smart_score_normalized
        sku_scores[sku]['combined_score'] = combined_score
    
    return sku_scores

def main():
    """
    主函数
    """
    # 文件路径配置
    day_data_dir = r"c:\Users\Administrator\Desktop\宝洁实习\20250703新数据动态选品\ProcessedData"
    smart_sku_file = r"c:\Users\Administrator\Desktop\宝洁实习\20250703新数据动态选品\Smart SKU\89月Smart SKU带Overall Score.csv"
    output_dir = r"c:\Users\Administrator\Desktop\宝洁实习\20250703新数据动态选品\Output"
    
    print("=" * 80)
    print("指数平滑加权超参数更新及选品程序（天数据版本）")
    print("=" * 80)
    
    # 用户输入参数
    print("\n请设置分析参数:")
    print("日期格式示例: 04-01 表示4月1日")
    try:
        start_date = input("请输入训练开始日期 (MM-DD): ").strip()
        end_date = input("请输入训练结束日期 (MM-DD): ").strip()
        target_date = input("请输入验证目标日期 (MM-DD): ").strip()
        
        # 验证日期格式
        import re
        date_pattern = r'^(0[4-7])-(0[1-9]|[12][0-9]|3[01])$'
        
        if not re.match(date_pattern, start_date) or not re.match(date_pattern, end_date) or not re.match(date_pattern, target_date):
            print("错误: 日期格式不正确，请使用MM-DD格式，月份限制在04-07月")
            return
        
        # 转换为文件名格式
        start_file_date = f"2025_{start_date.replace('-', '_')}"
        end_file_date = f"2025_{end_date.replace('-', '_')}"
        target_file_date = f"2025_{target_date.replace('-', '_')}"
        
        # 验证日期逻辑
        if start_date <= end_date:
            print("错误: 开始日期必须小于结束日期")
            return
            
            
    except ValueError:
        print("错误: 请输入有效的数字")
        return
    
    # 指数平滑参数
    try:
        alpha_input = input("请输入指数平滑参数alpha (0.1-0.5，直接回车使用默认值0.3): ").strip()
        if alpha_input == "":
            alpha = 0.3
        else:
            alpha = float(alpha_input)
            if alpha < 0.1 or alpha > 0.5:
                print("警告: alpha值超出推荐范围，将使用默认值0.3")
                alpha = 0.3
    except ValueError:
        print("错误: 请输入有效的数字，将使用默认值0.3")
        alpha = 0.3
    
    print(f"\n配置信息:")
    print(f"训练日期范围: {start_date} 到 {end_date}")
    print(f"验证日期: {target_date}")
    print(f"指数平滑参数alpha: {alpha}")
    
    try:
        # 加载Smart SKU数据
        smart_sku_categories, smart_sku_overall_scores = load_smart_sku_data(smart_sku_file)
        
        # 超参数优化
        print(f"\n=== 开始超参数优化 ===")
        
        # 定义超参数搜索空间
        alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        freq_weight_values = [0.3, 0.4, 0.5]
        cooc_weight_values = [0.2, 0.3, 0.4]
        quant_weight_values = [0.2, 0.3, 0.4]
        base_smart_ratio_values = [0.4, 0.5, 0.6]
        
        best_params = None
        best_score = -float('inf')
        optimization_results = []
        
        total_combinations = len(alpha_values) * len(freq_weight_values) * len(cooc_weight_values) * len(quant_weight_values) * len(base_smart_ratio_values)
        current_combination = 0
        
        print(f"总共需要测试 {total_combinations} 种参数组合")
        
        for alpha_val in alpha_values:
            for freq_w in freq_weight_values:
                for cooc_w in cooc_weight_values:
                    for quant_w in quant_weight_values:
                        for base_smart_r in base_smart_ratio_values:
                            current_combination += 1
                            
                            # 确保权重和为1
                            total_weight = freq_w + cooc_w + quant_w
                            freq_w_norm = freq_w / total_weight
                            cooc_w_norm = cooc_w / total_weight
                            quant_w_norm = quant_w / total_weight
                            
                            # 简化进度显示，每10个组合显示一次进度
                            if current_combination % 10 == 0 or current_combination == 1:
                                print(f"进度: {current_combination}/{total_combinations} ({current_combination/total_combinations*100:.1f}%)")
                            
                            try:
                                # 执行分析
                                sku_scores = analyze_sku_data_with_date_range(
                                    day_data_dir, start_file_date, end_file_date, 
                                    smart_sku_categories, smart_sku_overall_scores, 
                                    alpha_val, freq_w_norm, cooc_w_norm, quant_w_norm, base_smart_r, verbose=False
                                )
                                
                                # 获取前220个SKU
                                top_220_skus = sorted(sku_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)[:220]
                                top_220_sku_ids = [sku for sku, _ in top_220_skus]
                                
                                # 加载目标日期数据
                                target_file_path = os.path.join(day_data_dir, f"day_{target_file_date}.csv")
                                if os.path.exists(target_file_path):
                                    try:
                                        target_df = pd.read_csv(target_file_path, encoding='utf-8')
                                    except UnicodeDecodeError:
                                        target_df = pd.read_csv(target_file_path, encoding='gbk')
                                    
                                    # 计算前220个SKU在目标日期的quantity总和
                                    target_sku_quantities = {}
                                    for _, row in target_df.iterrows():
                                        if pd.notna(row['sku_qty_in_cs_pairs']):
                                            try:
                                                # 解析sku_qty_in_cs_pairs字典
                                                sku_qty_dict = ast.literal_eval(str(row['sku_qty_in_cs_pairs']))
                                                for sku, quantity in sku_qty_dict.items():
                                                    sku = str(sku).strip()
                                                    quantity = float(quantity)
                                                    target_sku_quantities[sku] = target_sku_quantities.get(sku, 0) + quantity
                                            except (ValueError, SyntaxError) as e:
                                                continue
                                    
                                    # 计算hit到的quantity
                                    hit_quantity = sum(target_sku_quantities.get(sku, 0) for sku in top_220_sku_ids)
                                    total_target_quantity = sum(target_sku_quantities.values())
                                    hit_rate = hit_quantity / total_target_quantity if total_target_quantity > 0 else 0
                                    
                                else:
                                    hit_quantity = 0
                                    total_target_quantity = 0
                                    hit_rate = 0
                                    print(f"警告: 目标日期文件不存在: {target_file_path}")
                                
                                # 记录结果
                                result = {
                                    'alpha': alpha_val,
                                    'freq_weight': freq_w_norm,
                                    'cooc_weight': cooc_w_norm,
                                    'quant_weight': quant_w_norm,
                                    'base_smart_ratio': base_smart_r,
                                    'hit_quantity': hit_quantity,
                                    'total_target_quantity': total_target_quantity,
                                    'hit_rate': hit_rate,
                                    'num_skus': len(sku_scores)
                                }
                                optimization_results.append(result)
                                
                                # 更新最佳参数（以hit_quantity为标准）
                                if hit_quantity > best_score:
                                    best_score = hit_quantity
                                    best_params = result.copy()
                                    print(f"发现更好的参数组合! Hit量: {hit_quantity:.0f} (组合 {current_combination})")
                                    
                            except Exception as e:
                                print(f"组合 {current_combination} 测试失败: {e}")
                                continue
        
        print(f"\n=== 超参数优化完成 ===")
        print(f"最佳参数组合:")
        print(f"  alpha: {best_params['alpha']}")
        print(f"  频次权重: {best_params['freq_weight']:.3f}")
        print(f"  共现权重: {best_params['cooc_weight']:.3f}")
        print(f"  数量权重: {best_params['quant_weight']:.3f}")
        print(f"  基础/Smart比例: {best_params['base_smart_ratio']}")
        print(f"  最佳Hit量: {best_params['hit_quantity']:.0f}")
        print(f"  Hit率: {best_params['hit_rate']:.4f}")
        
        # 使用最佳参数重新执行分析
        print(f"\n=== 使用最佳参数执行最终分析 ===")
        sku_scores = analyze_sku_data_with_date_range(
            day_data_dir, start_file_date, end_file_date, 
            smart_sku_categories, smart_sku_overall_scores, 
            best_params['alpha'], best_params['freq_weight'], 
            best_params['cooc_weight'], best_params['quant_weight'], 
            best_params['base_smart_ratio'], verbose=True
        )
        
        # 获取前220个SKU（按综合评分排序）
        top_220_skus = sorted(sku_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)[:220]
        top_220_sku_ids = [sku for sku, _ in top_220_skus]
        
        # 加载目标日期数据计算hit量
        target_file_path = os.path.join(day_data_dir, f"day_{target_file_date}.csv")
        if os.path.exists(target_file_path):
            try:
                target_df = pd.read_csv(target_file_path, encoding='utf-8')
            except UnicodeDecodeError:
                target_df = pd.read_csv(target_file_path, encoding='gbk')
            
            # 计算目标日期各SKU的quantity
            target_sku_quantities = {}
            for _, row in target_df.iterrows():
                if pd.notna(row['sku_qty_in_cs_pairs']):
                    try:
                        # 解析sku_qty_in_cs_pairs字典
                        sku_qty_dict = ast.literal_eval(str(row['sku_qty_in_cs_pairs']))
                        for sku, quantity in sku_qty_dict.items():
                            sku = str(sku).strip()
                            quantity = float(quantity)
                            target_sku_quantities[sku] = target_sku_quantities.get(sku, 0) + quantity
                    except (ValueError, SyntaxError) as e:
                        continue
            
            # 计算前220个SKU的hit量
            final_hit_quantity = sum(target_sku_quantities.get(sku, 0) for sku in top_220_sku_ids)
            total_target_quantity = sum(target_sku_quantities.values())
            final_hit_rate = final_hit_quantity / total_target_quantity if total_target_quantity > 0 else 0
        else:
            target_sku_quantities = {}
            final_hit_quantity = 0
            total_target_quantity = 0
            final_hit_rate = 0
            print(f"警告: 目标日期文件不存在: {target_file_path}")
        
        print(f"\n=== 分析完成 ===")
        print(f"总共分析了 {len(sku_scores)} 个SKU")
        print(f"前220个SKU Hit到的数量: {final_hit_quantity:.0f}")
        print(f"目标日期总数量: {total_target_quantity:.0f}")
        print(f"Hit率: {final_hit_rate:.4f}")
        print(f"\n前10个SKU评分:")
        for i, (sku, score_dict) in enumerate(top_220_skus[:10], 1):
            hit_qty = target_sku_quantities.get(sku, 0)
            print(f"  {i}. SKU {sku}: 评分={score_dict['combined_score']:.6f}, Hit量={hit_qty:.0f}")
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_filename = f"SKU评分结果_指数平滑加权_{timestamp}_{start_date}到{end_date}.csv"
        result_path = os.path.join(output_dir, result_filename)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存前220个SKU的详细结果到CSV
        result_df = pd.DataFrame([
            {
                '排名': i+1,
                'SKU': sku,
                '频次(加权)': f"{score_dict['frequency']:.6f}",
                '共现次数(加权)': f"{score_dict['cooccurrence_total']:.6f}",
                '数量(加权)': f"{score_dict['quantity']:.6f}",
                '频次标准化': f"{score_dict['freq_normalized']:.6f}",
                '共现标准化': f"{score_dict['cooccur_normalized']:.6f}",
                '数量标准化': f"{score_dict['quantity_normalized']:.6f}",
                'Base Score': f"{score_dict['base_score']:.6f}",
                'Smart SKU分类': score_dict['smart_category'],
                'Smart SKU Score': f"{score_dict['smart_sku_score']:.6f}",
                'Base Score标准化': f"{score_dict['base_score_normalized']:.6f}",
                'Smart Score标准化': f"{score_dict['smart_score_normalized']:.6f}",
                '综合评分': f"{score_dict['combined_score']:.6f}",
                '目标日期Hit量': f"{target_sku_quantities.get(sku, 0):.0f}"
            } 
            for i, (sku, score_dict) in enumerate(top_220_skus)
        ])
        result_df.to_csv(result_path, index=False, encoding='utf-8-sig')
        
        print(f"\n结果已保存到: {result_path}")
        
        # 保存超参数优化结果
        optimization_filename = f"超参数优化结果_{timestamp}_{start_date}到{end_date}.csv"
        optimization_path = os.path.join(output_dir, optimization_filename)
        
        optimization_df = pd.DataFrame(optimization_results)
        optimization_df = optimization_df.sort_values('hit_quantity', ascending=False)
        optimization_df.to_csv(optimization_path, index=False, encoding='utf-8-sig')
        
        print(f"超参数优化结果已保存到: {optimization_path}")
        
        # 显示最终分析参数
        print(f"\n最终分析参数:")
        print(f"  训练日期范围: {start_date} 到 {end_date}")
        print(f"  验证日期: {target_date}")
        print(f"  最优超参数:")
        print(f"    alpha: {best_params['alpha']}")
        print(f"    频次权重: {best_params['freq_weight']:.3f}")
        print(f"    共现权重: {best_params['cooc_weight']:.3f}")
        print(f"    数量权重: {best_params['quant_weight']:.3f}")
        print(f"    基础/Smart比例: {best_params['base_smart_ratio']}")
        print(f"  数据目录: {day_data_dir}")
        print(f"  Smart SKU文件: {smart_sku_file}")
        print(f"  输出目录: {output_dir}")
        print(f"\n超参数优化统计:")
        print(f"  测试的参数组合数: {len(optimization_results)}")
        print(f"  最佳Hit量: {best_params['hit_quantity']:.0f}")
        print(f"  最差Hit量: {min(r['hit_quantity'] for r in optimization_results):.0f}")
        min_hit = min(r['hit_quantity'] for r in optimization_results)
        if min_hit > 0:
            improvement = ((best_params['hit_quantity'] - min_hit) / min_hit * 100)
            print(f"  Hit量提升: {improvement:.2f}%")
        else:
            print(f"  Hit量提升: 无法计算（最差Hit量为0）")
        print(f"  最佳Hit率: {best_params['hit_rate']:.4f}")
        print(f"\n=== 最终选择的220个SKU及其信息 ===")
        print(f"总Hit量: {final_hit_quantity:.0f}")
        print(f"Hit率: {final_hit_rate:.4f}")
        print(f"使用的超参数:")
        print(f"  alpha: {best_params['alpha']}")
        print(f"  频次权重: {best_params['freq_weight']:.3f}")
        print(f"  共现权重: {best_params['cooc_weight']:.3f}")
        print(f"  数量权重: {best_params['quant_weight']:.3f}")
        print(f"  基础/Smart比例: {best_params['base_smart_ratio']:.3f}")
            
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()