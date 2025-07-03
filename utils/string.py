import re


# 模糊匹配方法
def fuzzy_match_str(columns: list, match_str: str) -> list:
    return [col for col in columns if re.search(match_str, col)]


def fuzzy_match_str_list(columns: list, match_str_list: list) -> list:
    return [col for col in columns if any(re.search(match_str, col) for match_str in match_str_list)]