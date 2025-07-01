import re
import random

def modify_fortran_current_voltage(
    fortran_path,
    output_path=None,
    wu_range=(10, 15),
    wi_range=(35, 50)
):
    """
    修改 Fortran 文件中的电压（wu）和电流（wi）为随机值，并返回结果。

    参数:
        fortran_path (str): 原始 Fortran 文件路径
        output_path (str): 修改后保存的文件路径（如果为 None，不保存）
        wu_range (tuple): 电压随机范围，例如 (10, 15)
        wi_range (tuple): 电流随机范围，例如 (35, 50)

    返回:
        new_wu (float): 修改后的电压值
        new_wi (float): 修改后的电流值
        modified_lines (List[str]): 修改后的 Fortran 文件内容行
    """
    # 读取 Fortran 文件
    with open(fortran_path, 'r') as f:
        lines = f.readlines()

    # 生成随机新值
    new_wu = round(random.uniform(*wu_range), 2)
    new_wi = round(random.uniform(*wi_range), 2)

    # 正则匹配替换
    pattern_wu = re.compile(r'^\s*wu\s*=\s*[\d.Ee+-]+', re.IGNORECASE)
    pattern_wi = re.compile(r'^\s*wi\s*=\s*[\d.Ee+-]+', re.IGNORECASE)

    modified_lines = []
    for line in lines:
        if pattern_wu.match(line):
            line = f"      wu={new_wu}\n"
        elif pattern_wi.match(line):
            line = f"      wi={new_wi}\n"
        modified_lines.append(line)

    # 写入新文件（如指定）
    if output_path:
        with open(output_path, 'w') as f:
            f.writelines(modified_lines)

    return new_wu, new_wi, modified_lines
