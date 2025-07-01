import re
import random
import pathlib

def two_sided_gap_uniform(delta: float) -> float:
    """生成 [-delta, -delta/2] ∪ [delta/2, delta] 区间的随机数"""
    if random.random() < 0.5:
        return random.uniform(-delta, -delta/2)
    else:
        return random.uniform(delta/2, delta)

def format_like(src: str, new_val: float) -> str:
    """保留字符串原始格式（小数位/指数格式）"""
    return str(new_val)

def perturb_inp_z(inp_path, out_path, part_name, target_nodes=None, delta=1e-4):
    """
    将 inp_path 中 Part=part_name 的节点 Z 坐标进行扰动，并写入 out_path。
    
    Parameters:
        inp_path (str): 输入 INP 文件路径
        out_path (str): 输出 INP 文件路径
        part_name (str): 目标 Part 名称（区分大小写）
        target_nodes (list[int] or None): 指定扰动的节点号列表，留空表示全部扰动
        delta (float): 扰动幅度（对称分布）
    """
    if target_nodes is None:
        target_nodes = set()
    else:
        target_nodes = set(target_nodes)

    node_pat = re.compile(r"^\s*([0-9]+)\s*,\s*(.*)$")
    coord_split = re.compile(r"[,\s]+")

    inside_target_part = False
    inside_node_block  = False

    with open(inp_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for line in fin:
            upper = line.upper().lstrip()

            # ---- 识别 Part 区块 ----
            if upper.startswith("*PART"):
                part_match = re.search(r"NAME\s*=\s*([^\s,]+)", upper)
                inside_target_part = part_match and (part_match.group(1) == part_name.upper())
            elif upper.startswith("*END PART"):
                inside_target_part = False

            # ---- 识别 Node 块 ----
            if inside_target_part and upper.startswith("*NODE"):
                inside_node_block = True
                fout.write(line)
                continue
            if inside_node_block and upper.startswith("*"):
                inside_node_block = False

            # ---- 节点扰动 ----
            if inside_node_block:
                m = node_pat.match(line)
                if m:
                    idx = int(m.group(1))
                    if (not target_nodes) or (idx in target_nodes):
                        coord_tokens = [tok for tok in coord_split.split(m.group(2)) if tok]
                        if len(coord_tokens) >= 3:
                            z_old = float(coord_tokens[2])
                            z_offset = two_sided_gap_uniform(delta)
                            z_new = z_old + z_offset
                            coord_tokens[2] = format_like(coord_tokens[2], z_new)
                            line = f"      {idx:>4},  " + ", ".join(coord_tokens) + "\n"

            fout.write(line)

    print("✅ 扰动完成 →", pathlib.Path(out_path).resolve())
