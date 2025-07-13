import pandas as pd

def aggregate_data(df: pd.DataFrame):
    df = df.sort_values(by=['transaction_date'])
    activity_code_path = df['activity_code'].tolist()
    operation_code_path = df['operation_code'].tolist()
    item_number_path = df['item_number'].tolist()
    qty_in_cs_path = df['qty_in_cs'].tolist()
    order_number_path = df['order_number'].tolist()
    from_area_path = df['from_area'].tolist()
    from_location_path = df['from_location'].tolist()
    to_area_path = df['to_area'].tolist()
    to_location_path = df['to_location'].tolist()
    transaction_date_path = df['transaction_date'].tolist()
    start_time = transaction_date_path[0]
    end_time = transaction_date_path[-1]
    time_diff = end_time - start_time
    time_diff = time_diff.total_seconds()
    user_id = df.iloc[0]['user_id']
    workday_id = df.iloc[0]['workday_id']
    task_id = df.iloc[0]['task_id']
    task_load_number = df.iloc[0]['task_load_number']
    work_day_start_time = df.iloc[0]['workday_start']
    work_day_end_time = df.iloc[0]['workday_end']
    work_day_time_diff = work_day_end_time - work_day_start_time
    work_day_time_diff = work_day_time_diff.total_seconds()
    unique_label_index = df.iloc[0]['unique_label_index']
    sku_qty_in_cs_pairs = df.groupby('item_number')['qty_in_cs'].sum()
    sku_qty_in_cs_pairs = sku_qty_in_cs_pairs.to_dict()
    sku_set = list(set(item_number_path))
    aggregate_data = {
        'user_id': user_id,
        'workday_id': workday_id,
        'task_id': task_id,
        'task_load_number': task_load_number,
        'work_day_start_time': work_day_start_time,
        'work_day_end_time': work_day_end_time,
        'work_day_time_diff': work_day_time_diff,
        'unique_label_index': unique_label_index,
        'activity_code_path': activity_code_path,
        'operation_code_path': operation_code_path,
        'item_number_path': item_number_path,
        'qty_in_cs_path': qty_in_cs_path,
        'order_number_path': order_number_path,
        'from_area_path': from_area_path,
        'from_location_path': from_location_path,
        'to_area_path': to_area_path,
        'to_location_path': to_location_path,
        'transaction_date_path': transaction_date_path,
        'pick_time_diff': time_diff,
        'pick_start_time': start_time,
        'pick_end_time': end_time,
        'sku_set': sku_set,
        'sku_qty_in_cs_pairs': sku_qty_in_cs_pairs,
        'pick_times': len(from_location_path),
        'sku_nums': len(sku_set),
        'date': pd.to_datetime(work_day_start_time).date(),
    }
    return pd.Series(aggregate_data)


def expand_dict_to_frame(df):
    data = df.iloc[0]['sku_qty_in_cs_pairs']
    if type(data) == str:
        data = eval(data)
    new_df = pd.DataFrame(data.items(), columns=['sku_id', 'quantity'])
    new_df['date'] = df.iloc[0]['date']
    new_df['work_day_start_time'] = df.iloc[0]['work_day_start_time']
    return new_df