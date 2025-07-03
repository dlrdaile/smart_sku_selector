import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
import re
import os

def load_smart_sku_data(smart_sku_file_path):
    """
    读取Smart SKU数据，提取M1和M3分类的SKU
    
    Args:
        smart_sku_file_path: Smart SKU CSV文件路径
    
    Returns:
        dict: {sku: category} 其中category为'M1', 'M2', 'M3', 'M4'
    """
    try:
        smart_df = pd.read_csv(smart_sku_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        smart_df = pd.read_csv(smart_sku_file_path, encoding='gbk')
    
    sku_categories = {}
    
    for idx, row in smart_df.iterrows():
        if pd.isna(row['Item Number']) or pd.isna(row['Explanation']):
            continue
            
        sku = str(row['Item Number']).strip()
        explanation = str(row['Explanation']).strip()
        
        # 提取M1-M4分类
        if explanation.startswith('M1:'):
            sku_categories[sku] = 'M1'
        elif explanation.startswith('M2:'):
            sku_categories[sku] = 'M2'
        elif explanation.startswith('M3:'):
            sku_categories[sku] = 'M3'
        elif explanation.startswith('M4:'):
            sku_categories[sku] = 'M4'
    
    m1_count = sum(1 for cat in sku_categories.values() if cat == 'M1')
    m2_count = sum(1 for cat in sku_categories.values() if cat == 'M2')
    m3_count = sum(1 for cat in sku_categories.values() if cat == 'M3')
    m4_count = sum(1 for cat in sku_categories.values() if cat == 'M4')

    return sku_categories

def load_smart_sku_overall_scores(smart_sku_overall_score_file_path):
    """
    读取Smart SKU Overall Score数据
    
    Args:
        smart_sku_overall_score_file_path: Smart SKU Overall Score CSV文件路径
    
    Returns:
        dict: {sku: overall_score}
    """
    try:
        score_df = pd.read_csv(smart_sku_overall_score_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        score_df = pd.read_csv(smart_sku_overall_score_file_path, encoding='gbk')
    
    sku_overall_scores = {}
    
    for idx, row in score_df.iterrows():
        if pd.isna(row['Item Number']) or pd.isna(row['Overall Score']):
            continue
            
        sku = str(row['Item Number']).strip()
        overall_score = float(row['Overall Score'])
        sku_overall_scores[sku] = overall_score
    
    return sku_overall_scores

def analyze_sku_data(csv_file_path, smart_sku_categories=None, smart_sku_overall_scores=None):
    """
    分析SKU数据，基于出现频次和共现关系选出前240个SKU
    
    Args:
        csv_file_path: CSV文件路径
        smart_sku_categories: Smart SKU分类字典，{sku: category}
        smart_sku_overall_scores: Smart SKU Overall Score字典，{sku: overall_score}
    
    Returns:
        top_240_skus: 前240个SKU列表
    """
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file_path, encoding='gbk')
    
    print(f"数据行数: {len(df)}")
    print(f"列名: {list(df.columns)}")
    
    # 提取SKU数据
    sku_frequency = Counter()  # SKU出现频次
    sku_cooccurrence = defaultdict(lambda: defaultdict(int))  # SKU共现矩阵
    task_sku_sets = []  # 每个任务的SKU集合
    
    print("正在分析SKU数据...")
    
    for idx, row in df.iterrows():
        if pd.isna(row['sku_set']):
            continue
            
        # 解析SKU字符串
        sku_string = str(row['sku_set']).strip()
        
        # 处理不同的分隔符格式
        # 移除引号并按逗号分割
        sku_string = sku_string.replace('"', '').replace("'", '')
        sku_list = [sku.strip() for sku in sku_string.split(',') if sku.strip()]
        
        # 过滤掉非数字SKU（可能的异常数据）
        valid_skus = []
        for sku in sku_list:
            # 只保留数字SKU
            if re.match(r'^\d+$', sku):
                valid_skus.append(sku)
        
        if not valid_skus:
            continue
            
        task_sku_sets.append(set(valid_skus))
        
        # 统计SKU频次
        for sku in valid_skus:
            sku_frequency[sku] += 1
        
        # 统计SKU共现关系
        if len(valid_skus) > 1:
            for sku1, sku2 in combinations(valid_skus, 2):
                sku_cooccurrence[sku1][sku2] += 1
                sku_cooccurrence[sku2][sku1] += 1
    
    print(f"总共发现 {len(sku_frequency)} 个不同的SKU")
    print(f"总共分析了 {len(task_sku_sets)} 个有效任务")
    
    # 计算SKU综合评分
    print("正在计算SKU综合评分...")
    
    # 第一步：收集所有SKU的原始指标数据
    raw_metrics = {}
    for sku in sku_frequency:
        # 基础分数：出现频次
        frequency_score = sku_frequency[sku]
        
        # 共现分数：与其他SKU的共现强度
        cooccurrence_score = 0
        if sku in sku_cooccurrence:
            # 计算该SKU与所有其他SKU的共现次数总和
            cooccurrence_score = sum(sku_cooccurrence[sku].values())
            
            # 计算平均共现强度（避免只与少数SKU高频共现的情况）
            if len(sku_cooccurrence[sku]) > 0:
                avg_cooccurrence = cooccurrence_score / len(sku_cooccurrence[sku])
            else:
                avg_cooccurrence = 0
        else:
            avg_cooccurrence = 0
        
        # 多样性分数：与多少个不同SKU共现过
        diversity_score = len(sku_cooccurrence.get(sku, {}))
        
        raw_metrics[sku] = {
            'frequency': frequency_score,
            'cooccurrence': cooccurrence_score,
            'diversity': diversity_score
        }
    
    # 第二步：数据标准化处理（Z-score标准化）
    print("正在进行数据标准化处理...")
    
    # 提取所有指标的数值
    frequencies = [metrics['frequency'] for metrics in raw_metrics.values()]
    cooccurrences = [metrics['cooccurrence'] for metrics in raw_metrics.values()]
    diversities = [metrics['diversity'] for metrics in raw_metrics.values()]
    
    # 计算均值和标准差
    freq_mean, freq_std = np.mean(frequencies), np.std(frequencies)
    cooccur_mean, cooccur_std = np.mean(cooccurrences), np.std(cooccurrences)
    diversity_mean, diversity_std = np.mean(diversities), np.std(diversities)
    
    print(f"标准化前数据统计:")
    print(f"频次 - 均值: {freq_mean:.2f}, 标准差: {freq_std:.2f}, 范围: {min(frequencies)}-{max(frequencies)}")
    print(f"共现 - 均值: {cooccur_mean:.2f}, 标准差: {cooccur_std:.2f}, 范围: {min(cooccurrences)}-{max(cooccurrences)}")
    print(f"多样性 - 均值: {diversity_mean:.2f}, 标准差: {diversity_std:.2f}, 范围: {min(diversities)}-{max(diversities)}")
    
    # 第三步：计算标准化后的评分
    sku_scores = {}
    
    for sku in sku_frequency:
        # 获取原始指标
        frequency_score = raw_metrics[sku]['frequency']
        cooccurrence_score = raw_metrics[sku]['cooccurrence']
        diversity_score = raw_metrics[sku]['diversity']
        
        # Z-score标准化（避免除零错误）
        freq_normalized = (frequency_score - freq_mean) / max(freq_std, 1e-8)
        cooccur_normalized = (cooccurrence_score - cooccur_mean) / max(cooccur_std, 1e-8)
        diversity_normalized = (diversity_score - diversity_mean) / max(diversity_std, 1e-8)
        
        # 基础综合评分公式（使用标准化后的数据）
        # 频次权重0.5，共现强度权重0.3，多样性权重0.2
        base_score = (
            0.5 * freq_normalized + 
            0.3 * cooccur_normalized + 
            0.2 * diversity_normalized
        )
        
        # 获取Smart SKU分类
        smart_category = 'Unknown'
        if smart_sku_categories and sku in smart_sku_categories:
            smart_category = smart_sku_categories[sku]
        
        # 获取Smart SKU Overall Score
        smart_sku_score = 0.0
        if smart_sku_overall_scores and sku in smart_sku_overall_scores:
            smart_sku_score = smart_sku_overall_scores[sku]
        
        # 计算综合评分：将base_score和smart_sku_score标准化后等权重结合
        # 由于两个评分的量级可能不同，需要进行标准化处理
        # 这里先保存原始值，后续会进行全局标准化
        
        sku_scores[sku] = {
            'frequency': frequency_score,
            'cooccurrence': cooccurrence_score,
            'diversity': diversity_score,
            'freq_normalized': freq_normalized,
            'cooccur_normalized': cooccur_normalized,
            'diversity_normalized': diversity_normalized,
            'base_score': base_score,
            'smart_category': smart_category,
            'smart_sku_score': smart_sku_score
        }
    
    # 第四步：对base_score和smart_sku_score进行标准化处理并计算综合评分
    print("正在计算综合评分...")
    
    # 提取所有base_score和smart_sku_score
    base_scores = [scores['base_score'] for scores in sku_scores.values()]
    smart_sku_scores = [scores['smart_sku_score'] for scores in sku_scores.values()]
    
    # 计算均值和标准差
    base_score_mean, base_score_std = np.mean(base_scores), np.std(base_scores)
    smart_score_mean, smart_score_std = np.mean(smart_sku_scores), np.std(smart_sku_scores)
    
    print(f"Base Score - 均值: {base_score_mean:.4f}, 标准差: {base_score_std:.4f}")
    print(f"Smart SKU Score - 均值: {smart_score_mean:.6f}, 标准差: {smart_score_std:.6f}")
    
    # 为每个SKU计算标准化后的综合评分
    for sku in sku_scores:
        base_score = sku_scores[sku]['base_score']
        smart_sku_score = sku_scores[sku]['smart_sku_score']
        
        # 标准化处理（避免除零错误）
        base_score_normalized = (base_score - base_score_mean) / max(base_score_std, 1e-8)
        smart_score_normalized = (smart_sku_score - smart_score_mean) / max(smart_score_std, 1e-8)
        
        # 综合评分：等权重结合两个标准化后的评分
        combined_score = 0.5 * base_score_normalized + 0.5 * smart_score_normalized
        
        # 更新sku_scores
        sku_scores[sku]['base_score_normalized'] = base_score_normalized
        sku_scores[sku]['smart_score_normalized'] = smart_score_normalized
        sku_scores[sku]['combined_score'] = combined_score
    
    # 按综合评分排序
    sorted_skus = sorted(sku_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)
    
    # 选出所有SKU
    all_skus = sorted_skus
    
    print(f"\n=== 所有{len(all_skus)}个SKU分析结果（含数据标准化和综合评分）===")
    print(f"排名\tSKU\t\t频次\t共现次数\t多样性\tBase Score\tSmart Score\t分类\t综合评分")
    print("-" * 120)
    
    for i, (sku, scores) in enumerate(all_skus[:20], 1):  # 只显示前20个
        print(f"{i:3d}\t{sku:10s}\t{scores['frequency']:4d}\t{scores['cooccurrence']:6.0f}\t{scores['diversity']:4d}\t{scores['base_score']:8.4f}\t{scores['smart_sku_score']:8.6f}\t{scores['smart_category']:4s}\t{scores['combined_score']:8.4f}")
    
    if len(all_skus) > 20:
        print(f"... (还有 {len(all_skus) - 20} 个SKU)")
    
    # 保存结果到文件
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "Output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'aggregated89_all_skus_normalized.csv')
    
    result_df = pd.DataFrame([
        {
            'rank': i + 1,
            'sku': sku,
            'frequency': scores['frequency'],
            'cooccurrence_total': scores['cooccurrence'],
            'diversity': scores['diversity'],
            'freq_normalized': scores['freq_normalized'],
            'cooccur_normalized': scores['cooccur_normalized'],
            'diversity_normalized': scores['diversity_normalized'],
            'base_score': scores['base_score'],
            'smart_category': scores['smart_category'],
            'smart_sku_score': scores['smart_sku_score'],
            'base_score_normalized': scores['base_score_normalized'],
            'smart_score_normalized': scores['smart_score_normalized'],
            'combined_score': scores['combined_score']
        }
        for i, (sku, scores) in enumerate(all_skus)
    ])
    
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_file}")
    
    # 分析统计信息
    print("\n=== 统计信息 ===")
    frequencies = [scores['frequency'] for _, scores in all_skus]
    print(f"所有{len(all_skus)}个SKU频次范围: {min(frequencies)} - {max(frequencies)}")
    print(f"所有SKU平均频次: {np.mean(frequencies):.2f}")
    
    cooccurrences = [scores['cooccurrence'] for _, scores in all_skus]
    print(f"所有SKU共现次数范围: {min(cooccurrences):.0f} - {max(cooccurrences):.0f}")
    print(f"所有SKU平均共现次数: {np.mean(cooccurrences):.2f}")
    
    diversities = [scores['diversity'] for _, scores in all_skus]
    print(f"所有SKU多样性范围: {min(diversities)} - {max(diversities)}")
    print(f"所有SKU平均多样性: {np.mean(diversities):.2f}")
    
    # 标准化后的数据统计
    print("\n=== 标准化后数据统计 ===")
    freq_norm = [scores['freq_normalized'] for _, scores in all_skus]
    cooccur_norm = [scores['cooccur_normalized'] for _, scores in all_skus]
    diversity_norm = [scores['diversity_normalized'] for _, scores in all_skus]
    
    print(f"标准化频次范围: {min(freq_norm):.4f} - {max(freq_norm):.4f}")
    print(f"标准化共现范围: {min(cooccur_norm):.4f} - {max(cooccur_norm):.4f}")
    print(f"标准化多样性范围: {min(diversity_norm):.4f} - {max(diversity_norm):.4f}")
    print(f"标准化后各指标均值接近0，标准差接近1，实现了量级统一")
    
    # Smart SKU分类统计
    print("\n=== Smart SKU分类统计 ===")
    category_counts = {}
    for _, scores in all_skus:
        cat = scores['smart_category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    total_skus = len(all_skus)
    for category in ['M1', 'M2', 'M3', 'M4', 'Unknown']:
        count = category_counts.get(category, 0)
        percentage = (count / total_skus) * 100
        print(f"{category}: {count}个 ({percentage:.1f}%)")
    
    # 重点关注M1和M3
    m1_m3_count = category_counts.get('M1', 0) + category_counts.get('M3', 0)
    m1_m3_percentage = (m1_m3_count / total_skus) * 100
    print(f"\n重点SKU (M1+M3): {m1_m3_count}个 ({m1_m3_percentage:.1f}%)")
    
    # 综合评分统计
    print("\n=== 综合评分统计 ===")
    combined_scores = [scores['combined_score'] for _, scores in all_skus]
    smart_scores = [scores['smart_sku_score'] for _, scores in all_skus]
    base_scores_all = [scores['base_score'] for _, scores in all_skus]
    
    print(f"综合评分范围: {min(combined_scores):.4f} - {max(combined_scores):.4f}")
    print(f"综合评分平均值: {np.mean(combined_scores):.4f}")
    print(f"Smart SKU Score范围: {min(smart_scores):.6f} - {max(smart_scores):.6f}")
    print(f"Base Score范围: {min(base_scores_all):.4f} - {max(base_scores_all):.4f}")
    
    return [sku for sku, _ in all_skus]

def analyze_top_cooccurrences(csv_file_path, top_skus, top_n=10):
    """
    分析前N个最强的SKU共现关系
    """
    print(f"\n=== 分析前{top_n}个最强SKU共现关系 ===")
    
    # 重新读取数据分析共现关系
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file_path, encoding='gbk')
    
    cooccurrence_pairs = defaultdict(int)
    
    for idx, row in df.iterrows():
        if pd.isna(row['sku_set']):
            continue
            
        sku_string = str(row['sku_set']).strip()
        sku_string = sku_string.replace('"', '').replace("'", '')
        sku_list = [sku.strip() for sku in sku_string.split(',') if sku.strip()]
        
        valid_skus = [sku for sku in sku_list if re.match(r'^\d+$', sku) and sku in top_skus]
        
        if len(valid_skus) > 1:
            for sku1, sku2 in combinations(sorted(valid_skus), 2):
                cooccurrence_pairs[(sku1, sku2)] += 1
    
    # 排序并显示前N个共现关系
    sorted_pairs = sorted(cooccurrence_pairs.items(), key=lambda x: x[1], reverse=True)
    
    print(f"SKU1\t\tSKU2\t\t共现次数")
    print("-" * 40)
    for (sku1, sku2), count in sorted_pairs[:top_n]:
        print(f"{sku1:10s}\t{sku2:10s}\t{count:4d}")
    
    # 保存共现关系结果到文件
    cooccurrence_df = pd.DataFrame([
        {
            'rank': i + 1,
            'sku1': sku1,
            'sku2': sku2,
            'cooccurrence_count': count
        }
        for i, ((sku1, sku2), count) in enumerate(sorted_pairs)
    ])
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "Output")
    os.makedirs(output_dir, exist_ok=True)
    cooccurrence_file = os.path.join(output_dir, 'sku_cooccurrence_analysis.csv')
    cooccurrence_df.to_csv(cooccurrence_file, index=False, encoding='utf-8-sig')
    print(f"\n共现关系分析结果已保存到: {cooccurrence_file}")
    
    return sorted_pairs

def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 文件路径
    csv_file = os.path.join(current_dir, "aggregated89.csv")
    smart_sku_file = os.path.join(current_dir, "89月Smart SKU带Overall Score.csv")
    smart_sku_overall_score_file = os.path.join(current_dir, "89月Smart SKU带Overall Score.csv")
    

    print("开始分析SKU数据（含数据标准化和综合评分）...")
    print("=" * 60)
    
    # 加载Smart SKU分类数据
    smart_sku_categories = load_smart_sku_data(smart_sku_file)
    
    # 加载Smart SKU Overall Score数据
    smart_sku_overall_scores = load_smart_sku_overall_scores(smart_sku_overall_score_file)
    
    # 执行主要分析（含数据标准化和综合评分）
    all_analyzed_skus = analyze_sku_data(csv_file, smart_sku_categories, smart_sku_overall_scores)
    
    # 分析顶级共现关系
    analyze_top_cooccurrences(csv_file, set(all_analyzed_skus), top_n=15)
    
    print("\n数据标准化和综合评分分析完成！")
    print("已整合Base Score和Smart SKU Score，使用等权重方法计算综合评分")
    print("=" * 60)
