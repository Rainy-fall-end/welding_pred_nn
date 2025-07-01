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
from abaqusConstants import NODAL
from sys import argv
import sys
import csv
import os

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
# 2. 写 Step 元数据表
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
# 3. 写节点矩阵表（考虑 instance 区分）
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

# 遍历每个 Step
for step in odb.steps.values():
    frame = step.frames[-1]  # 末帧

    # 位移
    disp_fld = frame.fieldOutputs['U'].getSubset(position=NODAL)
    disp_dict = {(v.instance.name, v.nodeLabel): v.data for v in disp_fld.values}

    # 应力
    if 'S' in frame.fieldOutputs:
        stress_fld = frame.fieldOutputs['S'].getSubset(position=NODAL)
        stress_dict = {(v.instance.name, v.nodeLabel): v.data for v in stress_fld.values}
    else:
        stress_dict = {}

    # 温度
    if 'NT11' in frame.fieldOutputs:
        temp_fld = frame.fieldOutputs['NT11'].getSubset(position=NODAL)
        temp_dict = {(v.instance.name, v.nodeLabel): v.data for v in temp_fld.values}
    else:
        temp_dict = {}

    # 写入所有 instance 的所有节点
    for inst in asm.instances.values():
        inst_name = inst.name
        for node in inst.nodes:
            n = node.label
            key = (inst_name, n)
            x, y, z = node.coordinates
            u  = disp_dict.get(key, (0.0, 0.0, 0.0))
            s  = stress_dict.get(key, (0.0,) * 6)
            nt = temp_dict.get(key, (0.0,))

            writer.writerow([
                step.name, inst_name, n,
                x, y, z,
                u[0], u[1], u[2],
                s[0], s[1], s[2], s[3], s[4], s[5],
                nt
            ])

fres.close()
odb.close()
