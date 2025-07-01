from torch.utils.data import DataLoader
from dataset.dataloader import WeldTensorFolderDataset
import torch
from utils.tensor_convert import flatten_middle_dimensions
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = WeldTensorFolderDataset(
    data_path=r"C:\works\codes\welding_pred_nn\abaqus_data\data",
    device=device   # or None / 'cpu'
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)

for batch in loader:
    print(flatten_middle_dimensions(batch).shape)  # [4, T, 2, 10, 71, 49]
