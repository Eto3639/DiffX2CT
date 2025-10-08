from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def _compute_importance_map(patch_size: tuple[int, ...], sigma_scale: float, device: torch.device) -> torch.Tensor:
    """ガウシアン重みマップを計算します。"""
    importance_map = np.ones(patch_size, dtype=np.float32)
    for dim, size in enumerate(patch_size):
        sd = size / (2.0 * sigma_scale)
        x = np.linspace(-size / 2.0, size / 2.0, size)
        importance_map *= np.exp(-((x / sd) ** 2))
    return torch.from_numpy(importance_map).to(device)


class SlidingWindowInferer:
    def __init__(
        self,
        roi_size: Sequence[int] | int,
        sw_batch_size: int = 1,
        overlap: float = 0.5,
        mode: str = "constant",
        sigma_scale: Sequence[float] | float = 0.125,
        padding_mode: str = "constant",
        cval: float = 0.0,
        sw_device: torch.device | str = "cuda",  # ★ 修正: デフォルトをcudaに
        device: torch.device | str | None = None,
        progress: bool = False,
        cache_roi_weight_map: bool = False,
        cpu_thresh: int | None = None,
    ) -> None:
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode = mode.lower()
        self.sigma_scale = sigma_scale
        self.padding_mode = padding_mode
        self.cval = cval
        self.sw_device = sw_device
        self.device = device
        self.progress = progress
        self.cpu_thresh = cpu_thresh 
        self.cache_roi_weight_map = cache_roi_weight_map 

    def __call__(self, inputs: torch.Tensor, network: Callable[..., torch.Tensor], *args, **kwargs) -> torch.Tensor:
        device = self.device
        if device is None and isinstance(inputs, torch.Tensor):
            device = inputs.device
        if device is None:
            device = torch.device("cuda")
        
        if isinstance(self.sw_device, str):
            sw_device = torch.device(self.sw_device)
        else:
            sw_device = self.sw_device

        num_spatial_dims = len(inputs.shape) - 2
        
        if isinstance(self.roi_size, int):
            roi_size = (self.roi_size,) * num_spatial_dims
        else:
            roi_size = tuple(self.roi_size)

        image_size_original = list(inputs.shape[2:])
        roi_size = tuple(min(r, s) for r, s in zip(roi_size, image_size_original))

        if self.overlap < 0 or self.overlap >= 1:
            raise ValueError(f"overlap must be >= 0 and < 1, got {self.overlap}.")
        scan_interval = [int(r * (1 - self.overlap)) for r in roi_size]

        if self.progress:
            try:
                from tqdm import tqdm
                num_rois = [int(np.ceil(i / j)) for i, j in zip(image_size_original, scan_interval)]
                total_iterations = int(np.prod(num_rois))
                pbar = tqdm(total=total_iterations, desc=f"Sliding window inference ({self.mode})")
            except (ImportError, ModuleNotFoundError):
                self.progress = False
                warnings.warn("tqdm is not installed, disabling progress bar.")

        if self.mode == "gaussian":
            importance_map = _compute_importance_map(roi_size, self.sigma_scale, device)
        else:
            importance_map = None

        scan_num = [int(np.ceil(float(i) / j)) for i, j in zip(image_size_original, scan_interval)]

        total_pad = [
            max(scan_interval[i] * (scan_num[i] - 1) + roi_size[i] - image_size_original[i], 0) for i in range(num_spatial_dims)
        ]
        pad_size = []
        for p in total_pad:
            pad_size.extend([p // 2, p - (p // 2)])

        pad_size_start = [pad_size[i] for i in range(0, len(pad_size), 2)]

        if any(pad_size):
            inputs = F.pad(inputs, pad_size[::-1], mode=self.padding_mode, value=self.cval)

        image_size_padded = list(inputs.shape[2:])

        scan_strides: list[torch.Tensor] = []
        for i in range(num_spatial_dims):
            stride_range = torch.arange(0, image_size_padded[i] - roi_size[i] + 1, scan_interval[i], device=device)
            scan_strides.append(stride_range)

        scan_strides_prod = torch.stack(torch.meshgrid(scan_strides, indexing="ij"), dim=-1).view(-1, num_spatial_dims)

        output_image, count_map = torch.tensor(0.0, device=inputs.device), torch.tensor(0.0, device=inputs.device)
        _initialized = False

        for slice_g in range(0, len(scan_strides_prod), self.sw_batch_size):
            slice_range = range(slice_g, min(slice_g + self.sw_batch_size, len(scan_strides_prod)))
            
            # untrimmed_box_startsはパディングされたボリューム内の座標
            untrimmed_box_starts = scan_strides_prod[slice_range]

            # バッチデータを構築
            batch_data = []
            batch_pos_3d = [] # ★ 位置エンコーディング用のバッチリスト
            for i in range(len(untrimmed_box_starts)):
                box_start_padded = untrimmed_box_starts[i]

                # 各ROIのスライスを作成
                spatial_slices = [slice(start.item(), start.item() + size) for start, size in zip(box_start_padded, roi_size)]

                # 入力からROIを抽出
                window_data_i = inputs[0, :, spatial_slices[0], spatial_slices[1], spatial_slices[2]]

                # ★ パッチがROIサイズより小さい場合(画像の端など)にパディングする
                if window_data_i.shape[1:] != roi_size:
                    patch_pad_width = []
                    for j in range(num_spatial_dims):
                        diff = roi_size[j] - window_data_i.shape[j+1]
                        patch_pad_width.extend([diff // 2, diff - (diff // 2)])
                    window_data_i = F.pad(window_data_i, patch_pad_width[::-1], mode="constant", value=self.cval)
                batch_data.append(window_data_i)

                # --- ★ パッチごとの位置エンコーディングを計算 ---
                pos_vectors = []
                for dim in range(num_spatial_dims):
                    # パディング前のオリジナルボリュームを基準とした座標を計算
                    start_coord_original = box_start_padded[dim].item() - pad_size_start[dim]
                    end_coord_original = start_coord_original + roi_size[dim]

                    # -1から1の範囲で正規化された座標を生成
                    patch_indices = torch.arange(start_coord_original, end_coord_original, device=device, dtype=torch.float32)
                    patch_coords = -1.0 + 2.0 * patch_indices / (image_size_original[dim] - 1)
                    pos_vectors.append(patch_coords)
                
                batch_pos_3d.append(torch.stack(pos_vectors, dim=0))
            
            window_data = torch.stack(batch_data, dim=0).to(sw_device)
            pos_3d_tensor = torch.stack(batch_pos_3d, dim=0).to(sw_device) # (B, 3, N)

            # ★ 修正: ネットワーク呼び出しをシンプルにし、計算した位置エンコーディングを渡す
            seg_prob = network(x=window_data, pos_3d=pos_3d_tensor, **kwargs).to(device)

            if not _initialized:
                output_classes = seg_prob.shape[1]
                output_shape = [inputs.shape[0], output_classes] + list(inputs.shape[2:])
                output_image = torch.zeros(*output_shape, dtype=torch.float32, device=device)
                count_map = torch.zeros(*output_shape, dtype=torch.float32, device=device)
                _initialized = True

            if importance_map is not None:
                # importance_mapをバッチサイズに合わせて拡張
                expanded_importance_map = importance_map.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
                seg_prob = seg_prob * expanded_importance_map

            # 出力を正しい位置に追加
            for i in range(len(trim_s)):
                spatial_slices = []
                for dim in range(num_spatial_dims):
                    start = trim_s[i][dim].item()
                    end = trim_e[i][dim].item()
                    spatial_slices.append(slice(start, end))
                
                # 出力テンソルに追加
                output_slice = (0, slice(None)) + tuple(spatial_slices)
                count_slice = (0, slice(None)) + tuple(spatial_slices)
                
                output_image[output_slice] += seg_prob[i]
                if importance_map is not None:
                    count_map[count_slice] += importance_map
                else:
                    count_map[count_slice] += 1.0

            if self.progress:
                pbar.update(len(slice_range))

        if self.progress:
            pbar.close()

        if not _initialized:
            warnings.warn("Sliding window inference not run, returning original input.")
            return inputs

        output_image = output_image / count_map

        if any(pad_size):
            crop_start = [p // 2 for p in total_pad]
            crop_end = [s - (p - (p // 2)) for s, p in zip(output_image.shape[2:], total_pad)]
            slices = [slice(None)] * 2 + [slice(cs, ce) for cs, ce in zip(crop_start, crop_end)]
            output_image = output_image[slices]

        return output_image