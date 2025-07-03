import json
import os
import time

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from logging import getLogger

logger = getLogger(__name__)


class SKUSelectionOptimizer:
    def __init__(self, config=None):
        """
        初始化SKU选品优化器

        参数:
            config (dict): 配置参数，如果为None则使用默认配置
        """
        # 默认配置
        self.default_config = {
            'max_sku_count': 200,  # 最大选品数量N
            'weight': 0.5,  # 优化目标权重W
            'time_limit': 600,  # 求解时间限制(秒)
            'mip_gap': 0.01,  # MIP Gap
            'pallet_count': 200,  # 托盘数量
            'output_dir': 'output'  # 输出目录
        }

        # 使用提供的配置或默认配置
        self.config = config if config else self.default_config

        # 初始化数据结构
        self.sku_data = None  # SKU数据
        self.order_data = None  # 订单数据
        self.pallet_data = None  # 托盘数据
        self.demand_data = None  # 需求数据

        # 初始化模型
        self.model = None
        self.x = None  # SKU选择决策变量
        self.y = None  # SKU-托盘分配决策变量
        self.z = None  # 订单-SKU需求指示变量

        # 初始化结果
        self.selected_skus = None
        self.sku_pallet_assignment = None
        self.objective_value = None
        self.solution_time = None

        # 创建输出目录
        if not os.path.exists(self.config['output_dir']):
            os.makedirs(self.config['output_dir'])

    def load_data(self, sku_data, order_data, pallet_file=None):
        """
        从文件加载SKU数据、订单数据和托盘数据

        参数:
            sku_file (str): SKU数据文件路径
            order_file (str): 订单数据文件路径
            pallet_file (str): 托盘数据文件路径，如果为None则生成默认托盘数据
        """
        logger.info("正在加载数据...")

        # 加载SKU数据
        if self.sku_data is pd.DataFrame:
            self.sku_data = sku_data
        else:
            self.sku_data = pd.read_csv(sku_data)
        logger.info(f"已加载 {len(self.sku_data)} 个SKU")

        # 加载订单数据
        if self.order_data is pd.DataFrame:
            self.order_data = order_data
        else:
            self.order_data = pd.read_csv(order_data)

        logger.info(f"已加载 {len(self.order_data)} 个订单")

        # 加载或生成托盘数据
        if pallet_file:
            self.pallet_data = pd.read_csv(pallet_file)
        else:
            # 生成默认托盘数据 - 假设每个托盘容量相同
            pallet_count = self.config['pallet_count']
            self.pallet_data = pd.DataFrame({
                'pallet_id': [f'P{i}' for i in range(1, pallet_count + 1)]
                # 'capacity': [50] * self.config['pallet_count']  # 假设每个托盘容量为50
            })
        logger.info(f"已配置 {len(self.pallet_data)} 个托盘")

        # 处理订单需求数据
        self._process_demand_data()

        return self

    def _process_demand_data(self):
        """处理订单需求数据，计算每个订单对每个SKU的需求量"""
        logger.info("正在处理订单需求数据...")

        # 创建需求矩阵: 订单 x SKU
        demand_data = self.order_data.pivot_table(
            index='order_id',
            columns='sku_id',
            values='quantity',
            fill_value=0
        )

        # 确保所有SKU都在需求矩阵中
        # for sku_id in self.sku_data['sku_id']:
        #     if sku_id not in demand_data.columns:
        #         demand_data[sku_id] = 0

        self.demand_data = demand_data
        logger.info(f"需求数据处理完成，共 {len(demand_data)} 个订单")

        # 计算每个SKU的总需求量和出现频率
        sku_total_demand = self.demand_data.sum()
        sku_frequency = (self.demand_data > 0).sum()

        # 将总需求量和频率添加到SKU数据中
        sku_stats = pd.DataFrame({
            'sku_id': sku_total_demand.index,
            'total_demand': sku_total_demand.values,
            'frequency': sku_frequency.values
        })

        self.sku_data = self.sku_data.merge(sku_stats, on='sku_id', how='left')
        self.sku_data['total_demand'] = self.sku_data['total_demand'].fillna(0)
        self.sku_data['frequency'] = self.sku_data['frequency'].fillna(0)

        return self

    def build_model(self):
        """构建Gurobi优化模型"""
        logger.info("正在构建优化模型...")

        # 创建模型
        self.model = gp.Model("SKU_Selection_Optimization")

        # 获取集合
        I = self.sku_data['sku_id'].tolist()  # SKU集合
        J = self.demand_data.index.tolist()  # 订单集合
        P = self.pallet_data['pallet_id'].tolist()  # 托盘集合

        # 创建决策变量
        logger.info("正在创建决策变量...")

        # x_i: 是否选择SKU i
        self.x = self.model.addVars(I, vtype=GRB.BINARY, name="x")

        # y_ip: 是否在托盘p上存放SKU i
        self.y = self.model.addVars([(i, p) for i in I for p in P], vtype=GRB.BINARY, name="y")

        # z_ij: 订单j是否需要SKU i
        self.z = {}
        for j in J:
            for i in I:
                demand = self.demand_data.loc[j, i]
                # 如果需求大于0，则z_ij = 1，否则z_ij = 0
                if demand > 0:
                    self.z[(i, j)] = 1
                else:
                    self.z[(i, j)] = 0

        # 添加约束条件
        logger.info("正在添加约束条件...")

        # 1. 最大选品数量约束
        self.model.addConstr(
            gp.quicksum(self.x[i] for i in I) <= self.config['max_sku_count'],
            name="max_sku_count"
        )

        # 2. 被选中的SKU至少被存放在一个托盘上
        for i in I:
            self.model.addConstr(
                gp.quicksum(self.y[i, p] for p in P) >= self.x[i],
                name=f"sku_{i}_assignment"
            )

        # 3. 未被选中的SKU不能被存放在任何托盘上
        for i in I:
            for p in P:
                self.model.addConstr(
                    self.y[i, p] <= self.x[i],
                    name=f"sku_{i}_pallet_{p}_consistency"
                )

        # 4. 托盘容量约束
        # for p in P:
        #     pallet_capacity = self.pallet_data.loc[self.pallet_data['pallet_id'] == p, 'capacity'].iloc[0]
        #     self.model.addConstr(
        #             gp.quicksum(self.y[i, p] for i in I) <= pallet_capacity,
        #             name=f"pallet_{p}_capacity"
        #     )

        # 5. 设置优化目标
        logger.info("正在设置优化目标...")

        # 计算每个订单的SKU种类满足率
        T1_terms = []
        for j in J:
            # 订单j中需要的SKU总数
            total_skus_needed = sum(1 for i in I if self.z[(i, j)] > 0)

            if total_skus_needed > 0:
                # 订单j中被选中的SKU数量
                selected_skus = gp.quicksum(self.x[i] for i in I if self.z[(i, j)] > 0)

                # 订单j的SKU种类满足率
                T1_terms.append(selected_skus / total_skus_needed)

        T1 = gp.quicksum(T1_terms) / len(J) if J else 0

        # 计算每个订单的箱数满足率
        T2_terms = []
        for j in J:
            # 订单j需要的总箱数
            total_boxes_needed = sum(self.demand_data.loc[j, i] for i in I)

            if total_boxes_needed > 0:
                # 订单j中被选中的SKU的箱数
                selected_boxes = gp.quicksum(self.demand_data.loc[j, i] * self.x[i] for i in I)

                # 订单j的箱数满足率
                T2_terms.append(selected_boxes / total_boxes_needed)

        T2 = gp.quicksum(T2_terms) / len(J) if J else 0

        # 总目标：最大化加权满足率
        W = self.config['weight']
        self.model.setObjective(W * T1 + (1 - W) * T2, GRB.MAXIMIZE)

        # 设置求解参数
        self.model.setParam('TimeLimit', self.config['time_limit'])
        self.model.setParam('MIPGap', self.config['mip_gap'])

        logger.info("优化模型构建完成")
        return self

    def solve(self):
        """求解优化模型"""
        if not self.model:
            raise ValueError("模型尚未构建，请先调用build_model()方法")

        logger.info("开始求解优化模型...")
        start_time = time.time()

        # 求解模型
        self.model.optimize()

        # 记录求解时间
        self.solution_time = time.time() - start_time

        # 检查求解状态
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.TIME_LIMIT:
            # 获取目标函数值
            self.objective_value = self.model.objVal

            # 获取选中的SKU
            I = self.sku_data['sku_id'].tolist()
            P = self.pallet_data['pallet_id'].tolist()

            self.selected_skus = [i for i in I if self.x[i].X > 0.5]

            # 获取SKU-托盘分配方案
            self.sku_pallet_assignment = {
                (i, p): 1 for i in I for p in P if self.y[i, p].X > 0.5
            }

            logger.info(f"优化完成，目标函数值: {self.objective_value:.4f}")
            logger.info(f"选中 {len(self.selected_skus)} 个SKU")

            # 计算订单满足率
            self._calculate_order_fulfillment()

            return True
        else:
            logger.error(f"优化求解失败，状态码: {self.model.status}")
            return False

    def _calculate_order_fulfillment(self):
        """ 计算订单满足率 """
        J = self.demand_data.index.tolist()  # 订单集合
        # 初始化订单满足率统计
        self.fulfillment_stats = {}
        total_fulfilled_sku_type_qty_rate = 0
        total_fulfilled_sku_qty_rate = 0
        # 计算订单满足率
        for j in J:
            # 每个订单j需要的SKU总数，及满足数量（只要选中就全部满足），获得满足数量rate，最后求平均
            order_total_skus_needed = (self.demand_data[j] > 0).sum()
            order_selected_skus = (
                        self.demand_data.loc[j, self.demand_data['sku_id'].isin(self.selected_skus)] > 0).sum()
            order_fulfilled_sku_type_qty_rate = order_selected_skus / order_total_skus_needed if order_total_skus_needed > 0 else 0
            total_fulfilled_sku_type_qty_rate += order_fulfilled_sku_type_qty_rate

            # 每个订单j需要的总箱数，及满足数量（只要选中就全部满足），获得满足数量rate，最后求平均
            order_total_boxes_needed = self.demand_data.loc[j].sum()
            order_selected_boxes = self.demand_data.loc[j, self.demand_data['sku_id'].isin(self.selected_skus)].sum()
            order_fulfilled_sku_qty_rate = order_selected_boxes / order_total_boxes_needed if order_total_boxes_needed > 0 else 0
            total_fulfilled_sku_qty_rate += order_fulfilled_sku_qty_rate

        self.fulfillment_stats['fulfilled_orders'] = len(J)
        self.fulfillment_stats['fulfilled_skus_qty_rate'] = total_fulfilled_sku_qty_rate / len(J)
        self.fulfillment_stats['fulfilled_sku_type_qty_rate'] = total_fulfilled_sku_type_qty_rate / len(J)
        return self

    def export_results(self, selected_sku_file='selected_skus.csv',
                       assignment_file='sku_pallet_assignment.csv',
                       stats_file='optimization_stats.json'):
        """
        导出优化结果

        参数:
            selected_sku_file (str): 选中的SKU列表文件名
            assignment_file (str): SKU-托盘分配方案文件名
            stats_file (str): 优化统计信息文件名
        """
        if not self.selected_skus:
            logger.error("尚未求解模型或未获得有效结果")
            return

        # 创建输出目录
        output_dir = self.config['output_dir']

        # 导出选中的SKU列表
        selected_skus_df = self.sku_data[self.sku_data['sku_id'].isin(self.selected_skus)]
        selected_skus_path = os.path.join(output_dir, selected_sku_file)
        selected_skus_df.to_csv(selected_skus_path, index=False, encoding='utf-8-sig')
        logger.info(f"选中的SKU列表已导出至: {selected_skus_path}")

        # 导出SKU-托盘分配方案
        assignment_data = []
        for (i, p), assigned in self.sku_pallet_assignment.items():
            if assigned > 0.5:
                assignment_data.append({
                    'sku_id': i,
                    'pallet_id': p
                })

        assignment_df = pd.DataFrame(assignment_data)
        assignment_path = os.path.join(output_dir, assignment_file)
        assignment_df.to_csv(assignment_path, index=False, encoding='utf-8-sig')
        logger.info(f"SKU-托盘分配方案已导出至: {assignment_path}")

        # 导出优化统计信息
        stats = {
            'objective_value': float(self.objective_value),
            'solution_time': float(self.solution_time),
            'selected_sku_count': len(self.selected_skus),
            'max_sku_count': self.config['max_sku_count'],
            'weight_parameter': self.config['weight'],
            'fulfillment_stats': self.fulfillment_stats
        }

        stats_path = os.path.join(output_dir, stats_file)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"优化统计信息已导出至: {stats_path}")

        return self


if __name__ == '__main__':
    # 或使用自定义配置
    config = {
        'max_sku_count': 50,
        'weight': 0.7,
        'time_limit': 300,
        'mip_gap': 0.05,
        'pallet_count': 220,
        'output_dir': 'results'
    }
    optimizer = SKUSelectionOptimizer(config)
    # 加载数据
    optimizer.load_data(
        sku_data=None,
        order_data=None,
        pallet_file='data/pallet_data.csv'  # 可选，如果没有可以不提供
    )

    # 构建模型
    optimizer.build_model()

    # 求解模型
    result = optimizer.solve()

    # 检查求解结果
    if result:
        print(f"优化成功，目标函数值: {optimizer.objective_value:.4f}")
        print(f"选中SKU数量: {len(optimizer.selected_skus)}")
    else:
        print("优化求解失败")

    # 导出结果
    optimizer.export_results(
        selected_sku_file='selected_skus.csv',
        assignment_file='sku_pallet_assignment.csv',
        stats_file='optimization_stats.json'
    )
