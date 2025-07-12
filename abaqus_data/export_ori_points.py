import numpy as np
from postprocess.csv2npz import build_node_matrix
import re
import numpy as np

def axis_angle_rotation_matrix(p1, p2, angle_deg):
    # 计算旋转轴单位向量
    axis = np.array(p2) - np.array(p1)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    angle_rad = np.radians(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1 - c

    # 罗德里格斯旋转公式构建旋转矩阵
    R = np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ]
    ])
    return R

def apply_instance_transform(points, pivot_point, axis_point, angle_deg):
    R = axis_angle_rotation_matrix(pivot_point, axis_point, angle_deg)
    # 中心化 → 旋转 → 平移回原始位置
    centered = points - pivot_point
    rotated = centered @ R.T
    transformed = rotated + pivot_point
    return transformed

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
# pointcloud_array = get_instance_pointclouds(node_dict, [node_mat1, node_mat2])  # (2, N, 3)
point_cloud_1 = get_instance_pointclouds(node_dict, [node_mat1]) 
point_cloud_2 = get_instance_pointclouds(node_dict, [node_mat2])
# print(pointcloud_array.shape)  # 应为 (2, 3479, 3)
# 第一个实例的变换（P4_19-2）
pivot1 = np.array([0.2, -0.00025, 0.0005])
axis1 = np.array([0.2, -0.00025, 1.0005])
angle1 = 180.0
transformed_pc1 = apply_instance_transform(point_cloud_1[0], pivot1, axis1, angle1)
pc1_transformed = transformed_pc1.reshape(71, 49, 3)
# 第二个实例的变换（P0_5MM-1）
pivot2 = np.array([0., 0.00025, 0.])
axis2 = np.array([0., 0.00025, -1.00000001268805])
angle2 = 89.9999992730282
transformed_pc2 = apply_instance_transform(point_cloud_2[0], pivot2, axis2, angle2)
pc2_transformed = transformed_pc2.reshape(71, 49, 3)
instance_dict = {
    "P4_19-1": pc1_transformed,
    "P4_19-2": pc2_transformed
}

# 4. 保存为 npz
np.savez("point_cloud.npz", **instance_dict)
# # for debug
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(12, 6))

# ax1 = fig.add_subplot(121, projection='3d')
# ax1.scatter(point_cloud_1[0][:,0], point_cloud_1[0][:,1], point_cloud_1[0][:,2], label='Original PC1')
# ax1.scatter(transformed_pc1[:,0], transformed_pc1[:,1], transformed_pc1[:,2], label='Transformed PC1')
# ax1.set_title('P4_19-2')
# ax1.legend()

# ax2 = fig.add_subplot(122, projection='3d')
# ax2.scatter(point_cloud_2[0][:,0], point_cloud_2[0][:,1], point_cloud_2[0][:,2], label='Original PC2')
# ax2.scatter(transformed_pc2[:,0], transformed_pc2[:,1], transformed_pc2[:,2], label='Transformed PC2')
# ax2.set_title('P0_5MM-1')
# ax2.legend()

# plt.tight_layout()
# plt.show()