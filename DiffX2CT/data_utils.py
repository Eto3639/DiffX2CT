# ファイル名: data_utils.py

import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import math


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

class GridPatchDatasetWithCond(Dataset):
    """
    MONAIに依存せず、グリッドパッチと対応するDRR、位置エンコーディングを返すデータセット。
    """
    def __init__(self, data, drr_dir, patch_size, patch_overlap=(0, 0, 0)):
        self.drr_dir = Path(drr_dir)
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        
        # 1. 元のCTボリュームとDRRをロード
        self.ct_volumes = [torch.load(p, map_location='cpu', weights_only=True).unsqueeze(0) 
                           if torch.load(p).dim() == 3 else torch.load(p)
                           for p in data]
        self.drr_pairs = []
        for p in data:
            drr_subdir = self.drr_dir / p.stem
            drr1 = torch.load(drr_subdir / "AP.pt", map_location='cpu', weights_only=True).unsqueeze(0)
            drr2 = torch.load(drr_subdir / "LAT.pt", map_location='cpu', weights_only=True).unsqueeze(0)
            self.drr_pairs.append((drr1, drr2))


    def _get_positional_encoding(self, patch_start_coords, patch_size, volume_size):
        """
        3D位置エンコーディングを、3つの1D座標ベクトルのタプルとして生成します。
        モデル(SinusoidalPosEmb)が期待する形式に合わせます。
        """
        for i, (start, size, vol_size) in enumerate(zip(patch_start_coords, patch_size, volume_size)):
            end = start + size
            pos_vectors[i] = torch.linspace(-1.0, 1.0, vol_size)[start:end]
        return tuple(pos_vectors)

    def __len__(self):
        # ★ 修正: 1エポックの長さをパッチ総数ではなく、元のCTボリューム数とする
        return len(self.ct_volumes)

    def __getitem__(self, index):
        # ★ 修正: indexはボリュームを指す。そのボリュームからランダムなパッチを切り出す
        original_image_idx = index
        ct_full = self.ct_volumes[original_image_idx]
        spatial_size = ct_full.shape[1:] # (D, H, W)
        
        # ランダムなパッチの開始座標を計算
        d_max = spatial_size[0] - self.patch_size[0]
        h_max = spatial_size[1] - self.patch_size[1]
        w_max = spatial_size[2] - self.patch_size[2]
        d_start = random.randint(0, d_max) if d_max > 0 else 0
        h_start = random.randint(0, h_max) if h_max > 0 else 0
        w_start = random.randint(0, w_max) if w_max > 0 else 0
        patch_start_coords = (d_start, h_start, w_start)
        d, h, w = patch_start_coords # パッチ切り出し用に展開
        pd, ph, pw = self.patch_size
        ct_patch = ct_full[:, d:d+pd, h:h+ph, w:w+pw]
        
        # 4. コンディショニング情報と位置エンコーディングの取得
        drr1, drr2 = self.drr_pairs[original_image_idx]

        # 位置エンコーディングの計算
        pos_vectors = [torch.zeros(s) for s in self.patch_size]
        for i, (start, size, vol_size) in enumerate(zip(patch_start_coords, self.patch_size, spatial_size)):
            end = start + size
            pos_vectors[i] = torch.linspace(-1.0, 1.0, vol_size)[start:end]
        pos_3d_tuple = tuple(pos_vectors)
        pos_3d_stacked = torch.stack(pos_3d_tuple, dim=0)
        return ct_patch.float(), drr1.float(), drr2.float(), pos_3d_stacked.float()