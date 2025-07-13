import pandas as pd
from pathlib import Path
from utils.string import fuzzy_match_str_list
import numpy as np
# 导入 pandarallel 并初始化
from pandarallel import pandarallel
from model.pre_process.utlis import aggregate_data,expand_dict_to_frame
from loguru import logger


def label_workdays(group):
    # 计算相邻记录的时间差
    time_diff = group['transaction_date'] - group['transaction_date'].shift(1)
    # 判断时间差是否大于 8 小时，若为第一行则默认标记为新工作日
    new_workday = (time_diff > pd.Timedelta(hours=6)) | group['transaction_date'].isnull()
    new_workday.iloc[0] = True

    # 为每个工作日分配唯一编号
    workday_id = new_workday.cumsum()

    # 确定每个工作日的开始时间
    workday_start = group['transaction_date'].groupby(workday_id).transform('min')
    # 确定每个工作日的结束时间
    workday_end = group['transaction_date'].groupby(workday_id).transform('max')

    group['workday_id'] = workday_id
    group['workday_start'] = workday_start
    group['workday_end'] = workday_end
    return group


def check_to_load_number_reuse(group: pd.DataFrame):
    # 判断 to_load_number 是否发生变化
    to_load_number_changed = group['to_load_number'] != group['to_load_number'].shift(1)
    to_load_number_changed.iloc[0] = True

    # 为每个 to_load_number 操作段分配唯一编号
    to_load_number_seg_id = to_load_number_changed.cumsum()

    # 计算每个 to_load_number 操作段的开始和结束时间
    start_time = group['transaction_date'].groupby(to_load_number_seg_id).transform('min')
    end_time = group['transaction_date'].groupby(to_load_number_seg_id).transform('max')

    # 检查是否存在 to_load_number 复用情况
    reuse_check = to_load_number_seg_id.groupby(group['to_load_number']).transform('nunique') > 1

    group['to_load_number_seg_id'] = to_load_number_seg_id
    group['to_load_number_start_time'] = start_time
    group['to_load_number_end_time'] = end_time
    group['to_load_number_reused'] = reuse_check

    return group


def assign_task_id(group_df: pd.DataFrame) -> pd.DataFrame:
    # 为每个 to_load_number 操作段分配唯一编号
    sorted_df = group_df.sort_values(by=['task_load_number', 'transaction_date'])
    # is_pick发生变化，如果是从0到1则说明是新任务
    is_pick_changed = sorted_df['is_pick'] - sorted_df['is_pick'].shift(1)
    is_pick_changed.iloc[0] = 0
    is_pick_changed[is_pick_changed == -1] = 0
    is_pick_changed = is_pick_changed.astype(bool)
    task_load_number_is_changed = sorted_df['task_load_number'] != sorted_df['task_load_number'].shift(1)
    # 为每个 to_load_number 操作段分配唯一编号
    to_load_number_seg_id = is_pick_changed | task_load_number_is_changed
    sorted_df['task_id'] = to_load_number_seg_id.cumsum()
    return sorted_df


def is_valid_task(df: pd.DataFrame):
    df['is_valid_task'] = False
    pick_count = df['is_pick'].sum()
    if pick_count * 2 == len(df):
        df['is_valid_task'] = True
    return df


def get_need_stat_task(df: pd.DataFrame):
    df['is_need_stat'] = False
    pick_count = df['is_pick'].sum()
    if pick_count * 2 >= len(df):
        df['is_need_stat'] = True
    return df


class PrimerDataPreProcess(object):
    exclude_activity_code_list = ["Inventory Delete",
                                  "Location Allocate to Empty Clear Automatic",
                                  "Location Allocate to Empty Set Automatic",
                                  "Location Override",
                                  "Location Status Change",
                                  "WQSO Change Priority",
                                  "WQSO Change Assigned User",
                                  "VC_CLS_TRK",
                                  "Remaining ASN Removed",
                                  "Master Receipt Close",
                                  "Mass Update",
                                  "Work Suspended",
                                  "Lot Date Changed",
                                  "Inventory Status Change"]
    duplicate_columns = ['load_number', 'to_load_number', 'item_number', 'transaction_date', 'quantity',
                         'order_number']

    def __init__(self, primer_data_path_list: list[Path], sku_data_path: Path,is_parallel=True) -> None:
        self.primer_data_path_list = primer_data_path_list
        self.sku_data_path = sku_data_path
        self.is_parallel = is_parallel
        self.primer_data_df = None
        self.sku_data_df = None
        self.last_date = None
        self.fine_tune_structure_df = None

    def run(self) -> pd.DataFrame:
        # 1.读取数据
        logger.info("start run -------")
        self.primer_data_df = self.read_raw_datas_and_concat(self.primer_data_path_list)
        self.sku_data_df = self.read_sku_master_data(self.sku_data_path)

        # 2. 去除无用的activity_code
        logger.info("start exclude activity code -------")
        self.primer_data_df = self.primer_data_df[
            ~self.primer_data_df['activity_code'].isin(self.exclude_activity_code_list)]

        # 3. 数据去重
        logger.info("start drop duplicate data -------")
        self.primer_data_df = self.primer_data_df.drop_duplicates(subset=self.duplicate_columns)

        # 4. 计算qty_in_cs
        logger.info("start calculate qty_in_cs -------")
        self.primer_data_df = self.calculate_qty_in_cs(self.primer_data_df)

        # 5. 散件数据处理
        logger.info("start process list pick data -------")
        self.primer_data_df = self.process_list_pick_data(self.primer_data_df)

        # 6. 标记工作日
        logger.info("start label workdays -------")
        self.primer_data_df = self.label_workdays(self.primer_data_df)

        # 7. 划分任务
        logger.info("start check to load number reuse -------")
        self.primer_data_df = self.label_task(self.primer_data_df)

        # 8. 数据聚合
        logger.info("start aggregate data -------")
        self.primer_data_df = self.aggregate_data(self.primer_data_df)

        self.last_date = self.primer_data_df['date'].max()

        self.fine_tune_structure_df = self.transfer_to_fine_select_data_structure(self.primer_data_df)

        return self.primer_data_df

    def _read_raw_data(self, filename: Path) -> pd.DataFrame:
        df = pd.read_csv(filename,
                         dtype={"Item Number": "str", "Lot Number": "str", "Order Number": "str", "Load Number": "str",
                                "To Load Number": "str", "Shipment": "str"})

        def col_rename(col_name: str):
            col_name = "_".join(col_name.strip().split(" "))
            return col_name.lower()

        df.columns = [col_rename(col) for col in df.columns]
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['task_load_number'] = df['to_load_number'].fillna(df['load_number'])
        df['is_pick'] = df['to_load_number'].notnull()
        return df

    def read_raw_datas_and_concat(self, file_list: list[Path]) -> pd.DataFrame:
        df_list = []
        for file in file_list:
            df = self._read_raw_data(file)
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)
        return df

    def read_sku_master_data(self, file_path: Path) -> pd.DataFrame:
        sku_md_df = pd.read_excel(file_path, dtype={'material_num': 'str'})
        sku_md_df.set_index('material_num', inplace=True)
        return sku_md_df

    def calculate_qty_in_cs(self, df: pd.DataFrame) -> pd.DataFrame:
        sku_buom_per_cs_s = self.sku_data_df.buom_per_cs
        sku_buom_per_cs_dict = sku_buom_per_cs_s.to_dict()
        df['qty_in_cs'] = df['quantity'] * df['item_number'].map(sku_buom_per_cs_dict)
        # 当qtt_in_cs为null时，赋值为quantity
        df['qty_in_cs'] = df['qty_in_cs'].fillna(df['quantity'])
        return df

    def process_list_pick_data(self, df):
        list_pick_match_activity_code = [r'Case Replenishment.*']
        list_pick_match_operation_code = [r'List Pick', 'Resume List Pick', 'Inventory Identifier Changed']
        all_activity_codes = df.activity_code.value_counts().index
        all_operation_codes = df.operation_code.value_counts().index
        list_pick_activity_code = fuzzy_match_str_list(all_activity_codes, list_pick_match_activity_code)
        list_pick_operation_code = fuzzy_match_str_list(all_operation_codes, list_pick_match_operation_code)
        list_pick_df = df[
            df.activity_code.isin(list_pick_activity_code) | df.operation_code.isin(list_pick_operation_code)]
        # 再次过滤
        second_activity_code_filter_list = ['List Pick', 'Case Replenishment', 'Logical Movement',
                                            'Inventory Identifier Changed']
        list_pick_df = list_pick_df[list_pick_df.activity_code.isin(second_activity_code_filter_list)]

        # 剔除异常qty_in_cs
        list_pick_df = list_pick_df[list_pick_df.qty_in_cs < 5000]

        return list_pick_df

    def label_workdays(self, primer_data_df):
        df = primer_data_df.copy()
        sorted_df = df.sort_values(by=['user_id', 'transaction_date'])
        labeled_df = sorted_df.groupby('user_id', group_keys=False).apply(label_workdays)
        labeled_df['workday_time_diff'] = labeled_df['workday_end'] - labeled_df['workday_start']
        return labeled_df

    def label_task(self, primer_data_df):
        df = primer_data_df.copy()
        task_label_df = df.groupby(['user_id', 'workday_id'], as_index=False, group_keys=False).apply(assign_task_id)

        # 标记任务是否是有效任务
        def get_unique_task_start_time(group_df: pd.DataFrame):
            sorted_df = group_df.sort_values(by=['transaction_date'])
            return sorted_df['transaction_date'].iloc[0]

        unique_df = task_label_df.groupby(['user_id', 'workday_id', 'task_id', 'task_load_number']).apply(
            get_unique_task_start_time)
        unique_df.name = 'start_time'
        unique_df = unique_df.reset_index()
        unique_df = unique_df.sort_values(by=['start_time'])
        unique_df['index'] = np.arange(len(unique_df)) + 1
        unique_df['unique_label'] = unique_df.apply(
            lambda x: f"{x['user_id']}_{x['workday_id']}_{x['task_id']}_{x['task_load_number']}", axis=1)

        unique_label_map_dict = unique_df.set_index('unique_label')['index'].to_dict()

        task_label_df['unique_label'] = task_label_df.apply(
            lambda x: f"{x['user_id']}_{x['workday_id']}_{x['task_id']}_{x['task_load_number']}", axis=1)
        task_label_df['unique_label_index'] = task_label_df['unique_label'].map(unique_label_map_dict)
        task_label_df = task_label_df.groupby(['unique_label_index'], as_index=False, group_keys=False).apply(
            is_valid_task)

        return task_label_df

    def aggregate_data(self, primer_data_df):
        df = primer_data_df.copy()
        stat_label_df = df.groupby(['unique_label_index'], as_index=False, group_keys=False).apply(
            get_need_stat_task)
        need_stat_label_df = stat_label_df.query('is_need_stat == True and is_pick == True')
        if self.is_parallel:
            aggregated_df = need_stat_label_df.groupby(['unique_label_index'], as_index=False,
                                                       group_keys=False).parallel_apply(
                aggregate_data)
        else:
            aggregated_df = need_stat_label_df.groupby(['unique_label_index'], as_index=False,
                                                       group_keys=False).apply(
                aggregate_data)

        return aggregated_df

    def transfer_to_fine_select_data_structure(self, aggregated_df: pd.DataFrame) -> pd.DataFrame:
        df: pd.DataFrame = aggregated_df.copy()
        if self.is_parallel:
            df2 = df.groupby(df.index).parallel_apply(expand_dict_to_frame)
        else:
            df2 = df.groupby(df.index).apply(expand_dict_to_frame)
        df2.reset_index(level=0, inplace=True)
        df2.index = np.arange(len(df2))
        df2.rename(columns={'level_0': 'order_id'}, inplace=True)

        return df2


if __name__ == '__main__':
    from warnings import filterwarnings

    filterwarnings('ignore')
    pandarallel.initialize(progress_bar=True,use_memory_fs=True)
    print("start run -------")
    data_path = Path(r'D:\Code\Python\sku_select_optimize\data\run_data')
    primer_data_path_list = list(data_path.rglob('*.csv'))
    sku_data_path = Path(r'D:\Code\Python\sku_select_optimize\data\raw_data\SKU_MD_0509.xlsx')
    pre_processor = PrimerDataPreProcess(primer_data_path_list, sku_data_path)
    primer_data_df = pre_processor.run()
    fine_tune_structure_df = pre_processor.transfer_to_fine_select_data_structure(primer_data_df)
    fine_tune_structure_df.to_excel("1.xlsx")
