# -*- coding: utf-8 -*-
# export_last_frame.py
from odbAccess import openOdb
from sys import argv
import csv
if len(argv) != 3:
    raise RuntimeError("用法: abaqus python export_last_frame.py job.odb out.csv")

odb_file, csv_file = argv[1], argv[2]
odb = openOdb(odb_file)
# ——假设只有一个分析步；若有多个可改成 odb.steps['Step-2'] 等
step = list(odb.steps.values())[-1]          # 最后一个 Step
frame = step.frames[-1]                      # 最后一帧

disp_field = frame.fieldOutputs['U']         # 位移场

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["node", "x", "y", "z", "ux", "uy", "uz"])

    # 遍历所有实例
    asm = odb.rootAssembly
    for inst_name, inst in asm.instances.items():
        # 取出该实例所有节点的位移值
        # values 列表中的每个 element : .nodeLabel, .data(3)
        disp_vals = disp_field.getSubset(region=inst).values
        label2disp = {v.nodeLabel: v.data for v in disp_vals}

        for node in inst.nodes:
            x, y, z = node.coordinates
            ux, uy, uz = label2disp.get(node.label, (0.0, 0.0, 0.0))
            writer.writerow([node.label, x, y, z, ux, uy, uz])

print(f"导出完成: {csv_file}")
odb.close()
