# ファイル名: data_utils.py

import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import math
import torch.nn.functional as F
from utils import load_config

# --- ★ 追加: config.ymlからサイズ設定を読み込む ---
CONFIG = load_config()
TARGET_VOLUME_SIZE = tuple(CONFIG["DATA"]["TARGET_VOLUME_SIZE"]) # (D, H, W)
TARGET_DRR_SIZE = tuple(CONFIG["DATA"]["TARGET_DRR_SIZE"])       # (H, W)

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

        # (C, D, H, W)の形式に合わせる
        if ct_scan_full.dim() == 3:
            ct_scan_full = ct_scan_full.unsqueeze(0)
        
        # --- ★ 変更: サイズが異なる場合のみリサイズ ---
        if ct_scan_full.shape[1:] != TARGET_VOLUME_SIZE:
            ct_scan_full = F.interpolate(ct_scan_full.unsqueeze(0), size=TARGET_VOLUME_SIZE, mode='trilinear', align_corners=False).squeeze(0)

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
        
        if drr1.dim() == 2: drr1 = drr1.unsqueeze(0)
        if drr2.dim() == 2: drr2 = drr2.unsqueeze(0)
        # --- ★ 変更: サイズが異なる場合のみリサイズ ---
        # F.interpolateは(N, C, H, W)を期待するため、次元を追加してからリサイズし、元に戻す
        if drr1.shape[1:] != TARGET_DRR_SIZE:
            drr1 = F.interpolate(drr1.unsqueeze(0).unsqueeze(0), size=TARGET_DRR_SIZE, mode='bilinear', align_corners=False).squeeze(0)
            drr2 = F.interpolate(drr2.unsqueeze(0).unsqueeze(0), size=TARGET_DRR_SIZE, mode='bilinear', align_corners=False).squeeze(0)

        # epsilon = 1e-6
        # drr1 = (drr1 - drr1.min()) / (drr1.max() - drr1.min() + epsilon)
        # drr2 = (drr2 - drr2.min()) / (drr2.max() - drr2.min() + epsilon)
        
        return ct_data, drr1, drr2, pos_3d

class GridPatchDatasetWithCond(Dataset):
    """
    グリッドパッチと対応するDRR、位置エンコーディングを返すデータセット。
    """
    def __init__(self, data: list[Path], drr_dir: Path, patch_size: tuple, patch_overlap: tuple = (0, 0, 0)):
        # --- ★ 変更: 遅延読み込みのための準備 ---
        self.ct_file_paths = data
        self.drr_dir = Path(drr_dir)
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.patch_info_list = []
        
        # 1. パッチの「レシピ」を作成する
        print("⏳ Initializing GridPatchDatasetWithCond...")
        for ct_index, ct_path in enumerate(self.ct_file_paths):
            # 実際のボリューム読み込みは行わず、形状情報だけを使う (ここでは固定サイズを前提とする)
            vol_shape = TARGET_VOLUME_SIZE
            stride = [ps - po for ps, po in zip(self.patch_size, self.patch_overlap)]
            
            for d in range(0, vol_shape[0] - self.patch_size[0] + 1, stride[0]):
                for h in range(0, vol_shape[1] - self.patch_size[1] + 1, stride[1]):
                    for w in range(0, vol_shape[2] - self.patch_size[2] + 1, stride[2]):
                        patch_start_coords = (d, h, w)
                        # (CTファイルのインデックス, パッチの開始座標) をリストに保存
                        self.patch_info_list.append((ct_index, patch_start_coords))
        
        print(f"✅ GridPatchDatasetWithCond initialized with {len(self.patch_info_list)} patch recipes.")


    def __len__(self):
        return len(self.patch_info_list)

    def __getitem__(self, index):
        # --- ★ 変更: ここで初めて実際のデータを読み込む ---
        ct_index, patch_start_coords = self.patch_info_list[index]
        
        # 1. CTボリュームを読み込み、リサイズ
        ct_path = self.ct_file_paths[ct_index]
        ct_vol = torch.load(ct_path, map_location='cpu', weights_only=True)
        if ct_vol.dim() == 3: ct_vol = ct_vol.unsqueeze(0)
        if ct_vol.shape[1:] != TARGET_VOLUME_SIZE:
            ct_vol = F.interpolate(ct_vol.unsqueeze(0), size=TARGET_VOLUME_SIZE, mode='trilinear', align_corners=False).squeeze(0)

        # 2. パッチを切り出す
        d, h, w = patch_start_coords
        pd, ph, pw = self.patch_size
        ct_patch = ct_vol[:, d:d+pd, h:h+ph, w:w+pw]

        # 3. 対応するDRRを読み込み、リサイズ
        drr_subdir = self.drr_dir / ct_path.stem
        drr1 = torch.load(drr_subdir / "AP.pt", map_location='cpu', weights_only=True)
        drr2 = torch.load(drr_subdir / "LAT.pt", map_location='cpu', weights_only=True)
        if drr1.dim() == 2: drr1 = drr1.unsqueeze(0)
        if drr2.dim() == 2: drr2 = drr2.unsqueeze(0)
        if drr1.shape[1:] != TARGET_DRR_SIZE:
            drr1 = F.interpolate(drr1.unsqueeze(0), size=TARGET_DRR_SIZE, mode='bilinear', align_corners=False).squeeze(0)
            drr2 = F.interpolate(drr2.unsqueeze(0), size=TARGET_DRR_SIZE, mode='bilinear', align_corners=False).squeeze(0)

        # 4. 位置エンコーディングを計算
        pos_vectors = [torch.zeros(s) for s in self.patch_size]
        for j, (start, size, vol_size) in enumerate(zip(patch_start_coords, self.patch_size, TARGET_VOLUME_SIZE)):
            end = start + size
            pos_vectors[j] = torch.linspace(-1.0, 1.0, vol_size)[start:end]
        pos_3d = torch.stack(pos_vectors, dim=0)

        return ct_patch.float(), drr1.float(), drr2.float(), pos_3d.float()