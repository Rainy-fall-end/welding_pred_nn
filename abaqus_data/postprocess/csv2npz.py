# 98-50
# 196-148
# 294-256
# ...
# 6958-6910

# 6910-6958
# 50-98

# postprocess_displacement.py
# -*- coding: utf-8 -*-
"""
把 export_steps_last_frame.py 得到的 *_results.csv
整理为 dict[step_name][instance_name] = disp_matrix

disp_matrix 说明
--------------
▪ 行   : 预定义的节点对（顺序固定）
▪ 列   : [U1(node_A), U2(node_A), U3(node_A),
         U1(node_B), U2(node_B), U3(node_B)]
"""


import pandas as pd
import numpy as np
from collections import defaultdict
import os
import csv
# ------------------------------------------------------------
# 1. 预定义节点矩阵（两个 instance 分别一份）
# ------------------------------------------------------------
def build_node_matrix(instance):
    """
    返回 nodes_mat  (71, 49)
    """
    if instance == "P4_19-1":
        starts = np.arange(98, 6958 + 1, 98)          # 98 → 6958, 共 71 个
        rows = [np.arange(s, s - 49, -1) for s in starts]
    elif instance == "P4_19-2":
        starts = np.arange(6910, 50 - 1, -98)         # 6910 → 50, 共 71 个
        rows = [np.arange(s, s + 49, 1) for s in starts]
    else:
        raise ValueError("只支持 P4_19-1 / P4_19-2")
    return np.vstack(rows)                            # (71, 49)

def convert_matrix_res(csv_results):
    NODE_MAT = {
    inst: build_node_matrix(inst) for inst in ["P4_19-1", "P4_19-2"]
}
    TARGET_INSTANCES = list(NODE_MAT.keys())

    # ------------------------------------------------------------
    # 2. 读取结果 CSV
    # ------------------------------------------------------------
    results_csv = csv_results          # ← 改成你的文件名
    df = pd.read_csv(results_csv)

    # 只保留目标 instance，并把 node 设为索引方便查表
    df = df[df["instance"].isin(TARGET_INSTANCES)].set_index(["instance", "node"])

    # ----------------------------- 3. 组装 dict -----------------------------
    disp_dict = defaultdict(dict)

    for step_name, step_df in df.groupby("step"):
        for inst in TARGET_INSTANCES:
            # 构造节点矩阵（71, 49）
            nodes_mat = NODE_MAT[inst]  # shape=(71, 49)

            # 提取该 step + instance 的全部数据
            sub_df = step_df.loc[step_df.index.get_level_values(0) == inst].copy()
            sub_df.reset_index(inplace=True)

            # ========== 建立索引映射 ==========
            nodes_available = sub_df["node"].values
            node_to_index = dict(zip(nodes_available, range(len(nodes_available))))

            try:
                index_mat = np.vectorize(lambda nid: node_to_index[nid])(nodes_mat)
            except KeyError as e:
                raise KeyError(f"❌ 缺失节点 {e.args[0]} in instance {inst} of step {step_name}")

            # ========== 取出各字段矩阵 ==========
            # 位移字段
            u_data = sub_df[["U1", "U2", "U3"]].values.astype(float)
            u_mat = u_data[index_mat]  # shape = (71, 49, 3)

            # 应力字段（S11~S23）
            s_data = sub_df[["S11", "S22", "S33", "S12", "S13", "S23"]].values.astype(float)
            s_mat = s_data[index_mat]  # shape = (71, 49, 6)

            # 温度字段（NT11）
            nt_data = sub_df["NT11"].values.astype(float)
            nt_mat = nt_data[index_mat]  # shape = (71, 49)

            # ========== 写入字典 ==========
            disp_dict[step_name][inst] = {
                'u': u_mat,
                's': s_mat,
                'nt': nt_mat
            }
    return disp_dict
def read_step_metadata(meta_csv_path):
    """
    读取 *_steps.csv，返回 list[dict]，每个 dict 表示一步的元数据。
    """
    step_meta_list = []
    with open(meta_csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 类型转换
            step_meta_list.append({
                'step_name': row['step_name'],
                'description': row['description'],
                'domain': row['domain'],
                'timePeriod': float(row['timePeriod']),
                'frames': int(row['frames']),
                'startTime': float(row['startTime']),
                'endTime': float(row['endTime']),
            })
    return step_meta_list

def convert_matrix(csv_results,csv_steps):
    res = {}
    res["data"] = convert_matrix_res(csv_results)
    res["worker"] = read_step_metadata(csv_steps)
    try:
        os.remove(csv_results)
        print(f"🗑️ 已删除 CSV 文件: {csv_results}")
        os.remove(csv_steps)
        print(f"🗑️ 已删除 CSV 文件: {csv_steps}")
    except FileNotFoundError:
        print(f"⚠️ 找不到 CSV 文件: {csv_results}")
        print(f"⚠️ 找不到 CSV 文件: {csv_steps}")
    except Exception as e:
        print(f"❌ 删除 CSV 失败: {e}")
    return res