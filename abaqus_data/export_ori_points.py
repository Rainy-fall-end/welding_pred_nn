import numpy as np
from postprocess.csv2npz import build_node_matrix
import re

def parse_inp_nodes(inp_path: str, part_name: str) -> dict:
    """
    从 .inp 文件中提取指定 Part 中的节点定义，返回 {node_id: (x, y, z)}
    仅处理 Part name == part_name 的区域。
    """
    node_dict = {}
    inside_target_part = False
    inside_node_block = False

    with open(inp_path, 'r') as f:
        for line in f:
            upper = line.strip().upper()

            # ---- 判断是否进入目标 Part 区块 ----
            if upper.startswith("*PART"):
                part_match = re.search(r"NAME\s*=\s*([^\s,]+)", upper)
                inside_target_part = part_match and (part_match.group(1) == part_name.upper())
                continue

            elif upper.startswith("*END PART"):
                inside_target_part = False
                inside_node_block = False
                continue

            # ---- 判断是否进入 Node 块 ----
            if inside_target_part and upper.startswith("*NODE"):
                inside_node_block = True
                continue

            elif inside_node_block and upper.startswith("*"):
                inside_node_block = False
                continue

            # ---- 读取节点坐标 ----
            if inside_node_block:
                parts = line.strip().split(",")
                if len(parts) >= 4:
                    try:
                        node_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        node_dict[node_id] = (x, y, z)
                    except ValueError:
                        continue  # 跳过无法解析的行

    return node_dict


def get_instance_pointclouds(node_dict: dict, node_mats: list[np.ndarray]) -> np.ndarray:
    pcs = []
    for mat in node_mats:
        node_ids = mat.flatten()
        points = []
        for nid in node_ids:
            nid_int = int(nid)
            # import pdb
            # pdb.set_trace()
            if nid_int not in node_dict:
                raise KeyError(f"节点 ID {nid_int} 不在 .inp 文件中")
            points.append(node_dict[nid_int])
        pcs.append(np.array(points))
    return np.stack(pcs, axis=0)


# 读取 inp 文件节点
node_dict = parse_inp_nodes("workdir_pipeline/Job-4.inp","P4_19")

# 已有两个 node_mat，比如：
node_mat1 = build_node_matrix("P4_19-1")  # (71,49)
node_mat2 = build_node_matrix("P4_19-2")  # (71,49)

# 获取两个实例的点云
pointcloud_array = get_instance_pointclouds(node_dict, [node_mat1, node_mat2])  # (2, N, 3)

print(pointcloud_array.shape)  # 应为 (2, 3479, 3)
