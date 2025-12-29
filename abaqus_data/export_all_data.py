# -*- coding: utf-8 -*-
"""
用法:
    abaqus python export_steps_last_frame.py Job.odb out_prefix
结果:
    out_prefix_results.csv  节点矩阵
    out_prefix_steps.csv    Step 元数据
"""
from __future__ import print_function
from odbAccess import openOdb
from abaqusConstants import NODAL, INTEGRATION_POINT
from sys import argv
import sys
import csv

# ------------------------------------------------------------
# 工具：兼容 Python 2/3 的 CSV 打开器
# ------------------------------------------------------------
def open_csv_writer(path):
    """
    返回一个 (file, writer) 二元组，兼容 Python 2 / 3。
    """
    if sys.version_info[0] < 3:
        f = open(path, "wb")
    else:
        f = open(path, "w", newline='')
    writer = csv.writer(f)
    return f, writer

# ------------------------------------------------------------
# 0. 解析参数
# ------------------------------------------------------------
if len(argv) != 3:
    raise RuntimeError("用法: abaqus python export_steps_last_frame.py Job.odb out_prefix")

odb_path, out_prefix = argv[1], argv[2]
res_csv  = out_prefix + "_results.csv"
meta_csv = out_prefix + "_steps.csv"

# ------------------------------------------------------------
# 1. 打开 ODB
# ------------------------------------------------------------
odb = openOdb(odb_path, readOnly=True)
asm = odb.rootAssembly

# ------------------------------------------------------------
# 2. 写 Step 元数据表（不改）
# ------------------------------------------------------------
fmeta, meta_writer = open_csv_writer(meta_csv)
meta_writer.writerow([
    "step_name", "description", "domain",
    "timePeriod", "frames", "startTime", "endTime"
])

for step in odb.steps.values():
    end_time   = step.frames[-1].frameValue if step.frames else 0.0
    start_time = step.frames[0].frameValue  if step.frames else 0.0
    meta_writer.writerow([
        step.name,
        step.description,
        step.domain,
        step.timePeriod,
        len(step.frames),
        start_time,
        end_time
    ])
fmeta.close()

# ------------------------------------------------------------
# 3. 写节点矩阵表（格式不变：一行一个 node，仍然输出 S11..S23 和 NT11）
# ------------------------------------------------------------
header = [
    "step", "instance", "node",
    "x", "y", "z",
    "U1", "U2", "U3",
    "S11", "S22", "S33", "S12", "S13", "S23",
    "NT11"
]

fres, writer = open_csv_writer(res_csv)
writer.writerow(header)

# -----------------------------
# 小工具：把积分点应力聚合到节点（最小侵入实现）
# -----------------------------
def build_nodal_stress_from_ip(frame):
    """
    返回 stress_dict: {(instanceName, nodeLabel): (S11,S22,S33,S12,S13,S23)}
    做法：
      IP 应力 -> 单元内积分点平均(centroid) -> 分摊到该单元所有节点 -> 节点平均
    说明：
      这不是 Abaqus CAE “精确同款” nodal extrapolation，但对你保持 CSV 格式最稳。
    """
    if 'S' not in frame.fieldOutputs:
        return {}

    s_fld = frame.fieldOutputs['S'].getSubset(position=INTEGRATION_POINT)

    # 1) element centroid stress: (inst, elemLabel) -> sum6, count_ip
    elem_sum = {}
    elem_cnt = {}

    for v in s_fld.values:
        inst_name = v.instance.name
        e = v.elementLabel
        key = (inst_name, e)
        s = v.data  # 6-tuple
        if key not in elem_sum:
            elem_sum[key] = [0.0] * 6
            elem_cnt[key] = 0
        for k in range(6):
            elem_sum[key][k] += float(s[k])
        elem_cnt[key] += 1

    # 2) distribute to nodes: (inst, nodeLabel) -> sum6, count_elem_contrib
    node_sum = {}
    node_cnt = {}

    for (inst_name, e), ssum in elem_sum.items():
        c = float(elem_cnt[(inst_name, e)])
        if c <= 0:
            continue
        s_centroid = [ssum[k] / c for k in range(6)]

        inst = asm.instances[inst_name]
        # element connectivity: tuple of node labels
        elem = inst.getElementFromLabel(e)
        conn = elem.connectivity

        for nlab in conn:
            nkey = (inst_name, nlab)
            if nkey not in node_sum:
                node_sum[nkey] = [0.0] * 6
                node_cnt[nkey] = 0
            for k in range(6):
                node_sum[nkey][k] += s_centroid[k]
            node_cnt[nkey] += 1

    # 3) average
    stress_dict = {}
    for nkey, ssum in node_sum.items():
        c = float(node_cnt[nkey])
        if c <= 0:
            continue
        stress_dict[nkey] = tuple(ssum[k] / c for k in range(6))

    return stress_dict


# 遍历每个 Step（不改结构）
for step in odb.steps.values():
    frame = step.frames[-1]  # 末帧

    # 位移（保持不变）
    disp_fld = frame.fieldOutputs['U'].getSubset(position=NODAL)
    disp_dict = {(v.instance.name, v.nodeLabel): v.data for v in disp_fld.values}

    # 应力（改：用 IP 聚合到节点，保证 (inst,node)->6 分量，CSV 格式不变）
    stress_dict = build_nodal_stress_from_ip(frame)

    # 温度（改：你 ODB 里是 NT11，不是 NT；仍按 NODAL 读）
    if 'NT11' in frame.fieldOutputs:
        temp_fld = frame.fieldOutputs['NT11'].getSubset(position=NODAL)
        temp_dict = {(v.instance.name, v.nodeLabel): v.data for v in temp_fld.values}
    else:
        temp_dict = {}

    # 写入所有 instance 的所有节点（保持你原本“全扫写入”的方式）
    for inst in asm.instances.values():
        inst_name = inst.name
        for node in inst.nodes:
            n = node.label
            key = (inst_name, n)
            x, y, z = node.coordinates
            u  = disp_dict.get(key, (0.0, 0.0, 0.0))
            s  = stress_dict.get(key, (0.0,) * 6)
            nt = temp_dict.get(key, 0.0)  # NT11 通常是标量 float

            writer.writerow([
                step.name, inst_name, n,
                x, y, z,
                float(u[0]), float(u[1]), float(u[2]),
                float(s[0]), float(s[1]), float(s[2]), float(s[3]), float(s[4]), float(s[5]),
                float(nt)
            ])

fres.close()
odb.close()
