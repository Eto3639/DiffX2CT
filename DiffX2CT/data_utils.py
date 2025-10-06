# ファイル名: data_utils.py

import torch
import random
from torch.utils.data import Dataset

class Preprocessed_CT_DRR_Dataset(Dataset):
    def __init__(self, pt_file_paths, drr_dir, patch_size=None):
        super().__init__()
        self.pt_file_paths = pt_file_paths
        self.drr_dir = drr_dir
        self.patch_size = patch_size

    def __len__(self):
        return len(self.pt_file_paths)

    def __getitem__(self, idx):
        ct_tensor_path = self.pt_file_paths[idx]
        ct_scan_full = torch.load(ct_tensor_path, map_location='cpu', weights_only=True)

        if ct_scan_full.dim() == 3:
            ct_scan_full = ct_scan_full.unsqueeze(0)
        
        # patch_sizeが指定されている場合のみ、ランダムなパッチを切り出す
        if self.patch_size is not None:
            c, d, h, w = ct_scan_full.shape
            pd, ph, pw = self.patch_size
            if d < pd or h < ph or w < pw:
                 raise ValueError(f"Full resolution {ct_scan_full.shape} is smaller than patch size {self.patch_size}")
            d_start, h_start, w_start = random.randint(0, d - pd), random.randint(0, h - ph), random.randint(0, w - pw)
            ct_data = ct_scan_full[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
            pos_3d = torch.tensor([d_start, h_start, w_start], dtype=torch.float32)
        else:
            ct_data = ct_scan_full
            pos_3d = torch.tensor([0, 0, 0], dtype=torch.float32)
        
        drr_subdir_name = ct_tensor_path.stem
        drr_ap_path = self.drr_dir / drr_subdir_name / "AP.pt"
        drr_lat_path = self.drr_dir / drr_subdir_name / "LAT.pt"
        
        if not drr_ap_path.exists() or not drr_lat_path.exists():
            raise FileNotFoundError(f"DRR files not found in {self.drr_dir / drr_subdir_name}")
            
        drr1 = torch.load(drr_ap_path, map_location='cpu', weights_only=True)
        drr2 = torch.load(drr_lat_path, map_location='cpu', weights_only=True)
        
        # epsilon = 1e-6
        # drr1 = (drr1 - drr1.min()) / (drr1.max() - drr1.min() + epsilon)
        # drr2 = (drr2 - drr2.min()) / (drr2.max() - drr2.min() + epsilon)
        
        if drr1.dim() == 2: drr1 = drr1.unsqueeze(0)
        if drr2.dim() == 2: drr2 = drr2.unsqueeze(0)
        
        return ct_data, drr1, drr2, pos_3d