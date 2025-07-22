import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import shutil

class WeldTensorFolderDataset(Dataset):
    def __init__(self, data_path, npz_name="weld_data.npz", device=None, step_num=44,max_samples=None):
        """
        :param data_path: 根目录，多个子文件夹，每个子文件夹有一个 weld_data.npz
        :param npz_name: 每个子文件夹中的 npz 文件名
        :param device: 数据加载到的设备，如 'cuda' 或 torch.device
        """
        self.data_paths = []
        self.device = torch.device(device) if device else torch.device("cpu")

        for subfolder in sorted(os.listdir(data_path)):
            folder_path = os.path.join(data_path, subfolder)
            npz_path = os.path.join(folder_path, npz_name)

            # (1) .npz 是否存在？
            if not os.path.isfile(npz_path):
                print(f"⚠️ 跳过并删除: {folder_path}, 未找到 {npz_name}")
                shutil.rmtree(folder_path)          # 直接删即可
                continue

            # (2) 读取 .npz，检查 data 长度
            try:
                with np.load(npz_path, allow_pickle=True) as weld_datas:
                    data_len = len(weld_datas["data"].item())  # 读取完立刻关闭文件
            except Exception as e:
                print(f"⚠️ 无法读取 {npz_path}: {e}")
                data_len = None                                # 标记为无效

            # (3) 决定是否保留
            if data_len != step_num:
                print(f"⚠️ 跳过并删除: {folder_path}, data 长度为 {data_len}")
                shutil.rmtree(folder_path)                      # 现在文件已关闭，可以安全删除
                continue

            # (4) 通过检查，加入数据集
            self.data_paths.append(npz_path)

            if max_samples is not None and max_samples > 0 and len(self.data_paths) >= max_samples:
                print(f"✅ 已收集到 {max_samples} 个样本，提前结束数据扫描")
                break
        sample_num = len(self.data_paths)
        print(f"共有{sample_num}条训练数据")
        self.instances = ["P4_19-1", "P4_19-2"]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        npz_path = self.data_paths[idx]
        weld_datas = np.load(npz_path, allow_pickle=True)
        data = weld_datas["data"].item()

        step_names = data.keys()

        tensors = []

        for step in step_names:
            inst_tensor = []
            for inst in self.instances:
                try:
                    u = data[step][inst]["u"]  # (71,49,3)
                    s = data[step][inst]["s"]  # (71,49,6)
                    nt = data[step][inst]["nt"]  # (71,49)
                    nt = nt[:, :, np.newaxis]  # → (71,49,1)

                    all_fields = np.concatenate([u, s, nt], axis=-1)  # (71,49,10)
                    inst_tensor.append(all_fields.transpose(2, 0, 1))  # (10,71,49)
                except KeyError as e:
                    raise KeyError(f"{npz_path} 缺失字段: {e}")

            step_tensor = np.stack(inst_tensor, axis=0)  # (2,10,71,49)
            tensors.append(step_tensor)

        out_tensor = np.stack(tensors, axis=0)  # (T,2,10,71,49)
        out_tensor = torch.from_numpy(out_tensor).float().to(self.device)

        data_time = weld_datas["worker"]
        start_times = []
        time_periods = []

        data_time = weld_datas["worker"]
        time_periods = [item["timePeriod"] for item in data_time]

        start_times = [0.0]
        for i in range(1, len(time_periods)):
            start_time = (
                start_times[-1] + time_periods[i - 1]
            )  # startTime = previous_endTime + previous_timePeriod
            start_times.append(start_time)
        start_times_tensor = torch.tensor(start_times, dtype=torch.float32)
        time_periods_tensor = torch.tensor(time_periods, dtype=torch.float32)
        para_dict = weld_datas["para"].item()
        ui = para_dict["ui"]
        vi = para_dict["vi"]
        para_tensor = torch.tensor([ui, vi], dtype=torch.float32)
        # out_tensor = np.stack(tensors, axis=0)         # (T, 2, 10, 71, 49)
        T = out_tensor.shape[0]
        out_tensor = out_tensor.permute(0, 2, 1, 3, 4)
        out_tensor = out_tensor.reshape(T, 10, 71, 49 * 2)   #  (T, 10, 71, 98)
        return (
            out_tensor.to(self.device),
            start_times_tensor.to(self.device),
            time_periods_tensor.to(self.device),
            para_tensor.to(self.device),
        )

    @property
    def shape(self, idx=0):
        npz_path = self.data_paths[idx]
        data = np.load(npz_path, allow_pickle=True)["data"].item()

        step_names = sorted(data.keys(), key=lambda x: x.lower())
        tensors = []

        for step in step_names:
            inst_tensor = []
            for inst in self.instances:
                try:
                    u = data[step][inst]["u"]  # (71,49,3)
                    s = data[step][inst]["s"]  # (71,49,6)
                    nt = data[step][inst]["nt"]  # (71,49)
                    nt = nt[:, :, np.newaxis]  # → (71,49,1)

                    all_fields = np.concatenate([u, s, nt], axis=-1)  # (71,49,10)
                    inst_tensor.append(all_fields.transpose(2, 0, 1))  # (10,71,49)
                except KeyError as e:
                    raise KeyError(f"{npz_path} 缺失字段: {e}")

            step_tensor = np.stack(inst_tensor, axis=0)  # (2,10,71,49)
            tensors.append(step_tensor)

        out_tensor = np.stack(tensors, axis=0)         # (T, 2, 10, 71, 49)
        T = out_tensor.shape[0]
        out_tensor = out_tensor.transpose(0, 2, 1, 3, 4)  # (T, 10, 2, 71, 49)
        out_tensor = out_tensor.reshape(T, 10, 71, 49 * 2)
        out_tensor = torch.from_numpy(out_tensor).float().to(self.device)

        return out_tensor.shape


def build_dataset(args):
    # 构建完整数据集
    dataset = WeldTensorFolderDataset(
        data_path=args.data_dir, 
        device=args.device, 
        step_num=args.step_nums,
        max_samples=args.max_samples
    )

    # 计算划分大小
    total_len = len(dataset)
    train_len = int(total_len * args.train_ratio)
    test_len = total_len - train_len

    # 随机划分数据集
    train_set, test_set = random_split(dataset, [train_len, test_len])

    # 构建 DataLoader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader
