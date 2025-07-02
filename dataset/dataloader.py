import os
import numpy as np
import torch
from torch.utils.data import Dataset

import os
import numpy as np
import torch
from torch.utils.data import Dataset
class WeldTensorFolderDataset(Dataset):
    def __init__(self, data_path, npz_name="weld_data.npz", device=None):
        """
        :param data_path: 根目录，多个子文件夹，每个子文件夹有一个 weld_data.npz
        :param npz_name: 每个子文件夹中的 npz 文件名
        :param device: 数据加载到的设备，如 'cuda' 或 torch.device
        """
        self.data_paths = []
        self.device = torch.device(device) if device else torch.device('cpu')

        for subfolder in sorted(os.listdir(data_path)):
            folder_path = os.path.join(data_path, subfolder)
            npz_path = os.path.join(folder_path, npz_name)
            if os.path.isfile(npz_path):
                self.data_paths.append(npz_path)
            else:
                print(f"⚠️ 跳过: {folder_path}, 未找到 {npz_name}")

        self.instances = ["P4_19-1", "P4_19-2"]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        npz_path = self.data_paths[idx]
        data = np.load(npz_path, allow_pickle=True)['data'].item()

        step_names = sorted(data.keys(), key=lambda x: x.lower())
        tensors = []

        for step in step_names:
            inst_tensor = []
            for inst in self.instances:
                try:
                    u = data[step][inst]['u']     # (71,49,3)
                    s = data[step][inst]['s']     # (71,49,6)
                    nt = data[step][inst]['nt']   # (71,49)
                    nt = nt[:, :, np.newaxis]     # → (71,49,1)

                    all_fields = np.concatenate([u, s, nt], axis=-1)  # (71,49,10)
                    inst_tensor.append(all_fields.transpose(2, 0, 1))  # (10,71,49)
                except KeyError as e:
                    raise KeyError(f"{npz_path} 缺失字段: {e}")

            step_tensor = np.stack(inst_tensor, axis=0)  # (2,10,71,49)
            tensors.append(step_tensor)

        out_tensor = np.stack(tensors, axis=0)  # (T,2,10,71,49)
        out_tensor = torch.from_numpy(out_tensor).float().to(self.device)
        return out_tensor

def build_dataset(args):
    pass