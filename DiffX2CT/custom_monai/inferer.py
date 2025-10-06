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
            device = torch.device("cuda")  # ★ 修正: デフォルトをcudaに
        
        # ★ 修正: sw_deviceが文字列の場合の処理を追加
        if isinstance(self.sw_device, str):
            sw_device = torch.device(self.sw_device)
        else:
            sw_device = self.sw_device

        num_spatial_dims = len(inputs.shape) - 2
        
        if isinstance(self.roi_size, int):
            roi_size = (self.roi_size,) * num_spatial_dims
        else:
            roi_size = tuple(self.roi_size)

        image_size = list(inputs.shape[2:])
        roi_size = tuple(min(r, s) for r, s in zip(roi_size, image_size))

        if self.overlap < 0 or self.overlap >= 1:
            raise ValueError(f"overlap must be >= 0 and < 1, got {self.overlap}.")
        scan_interval = [int(r * (1 - self.overlap)) for r in roi_size]

        if self.progress:
            try:
                from tqdm import tqdm
                num_rois = [int(np.ceil(i / j)) for i, j in zip(image_size, scan_interval)]
                total_iterations = int(np.prod(num_rois))
                pbar = tqdm(total=total_iterations, desc=f"Sliding window inference ({self.mode})")
            except (ImportError, ModuleNotFoundError):
                self.progress = False
                warnings.warn("tqdm is not installed, disabling progress bar.")

        if self.mode == "gaussian":
            importance_map = _compute_importance_map(roi_size, self.sigma_scale, device)  # ★ 修正: deviceを使用
        else:
            importance_map = None

        scan_num = [int(np.ceil(float(i) / j)) for i, j in zip(image_size, scan_interval)]

        total_pad = [
            max(scan_interval[i] * (scan_num[i] - 1) + roi_size[i] - image_size[i], 0) for i in range(num_spatial_dims)
        ]
        pad_size = []
        for p in total_pad:
            pad_size.extend([p // 2, p - (p // 2)])

        if any(pad_size):
            inputs = F.pad(inputs, pad_size[::-1], mode=self.padding_mode, value=self.cval)

        scan_strides: list[torch.Tensor] = []
        for i in range(num_spatial_dims):
            stride_range = torch.arange(scan_num[i], dtype=torch.long, device=device)  # ★ 修正: deviceを使用
            scan_strides.append(stride_range)

        scan_strides_prod = torch.stack(torch.meshgrid(scan_strides, indexing="ij"), dim=-1).view(-1, num_spatial_dims)

        output_image, count_map = torch.tensor(0.0, device=inputs.device), torch.tensor(0.0, device=inputs.device)
        _initialized = False

        for slice_g in range(0, len(scan_strides_prod), self.sw_batch_size):
            slice_range = range(slice_g, min(slice_g + self.sw_batch_size, len(scan_strides_prod)))
            untrimmed_box_starts = scan_strides_prod[slice_range]
            untrimmed_box_ends = untrimmed_box_starts + torch.as_tensor(roi_size, device=device)  # ★ 修正: deviceを使用
            
            trim_s = torch.max(untrimmed_box_starts, torch.zeros_like(untrimmed_box_starts)).long()
            trim_e = torch.min(untrimmed_box_ends, torch.as_tensor(inputs.shape[2:], device=device)).long()

            # バッチデータを構築
            batch_data = []
            for i in range(len(trim_s)):
                # 各ROIのスライスを作成
                spatial_slices = []
                for dim in range(num_spatial_dims):
                    start = trim_s[i][dim].item()
                    end = trim_e[i][dim].item()
                    spatial_slices.append(slice(start, end))
                
                # 入力からROIを抽出 [C, D, H, W] の形で
                roi = inputs[0, :, spatial_slices[0], spatial_slices[1], spatial_slices[2]]
                batch_data.append(roi)
            
            # バッチ次元でスタック [B, C, D, H, W]
            window_data = torch.stack(batch_data, dim=0)
            # DataParallelに対応するため、モデルと同じデバイスにデータを送る
            if isinstance(network, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                window_data = window_data.to(device)

            # ★ 修正: networkの引数に 'x' があるかチェックして呼び出しを分ける
            import inspect
            sig = inspect.signature(network.forward if isinstance(network, torch.nn.Module) else network)
            if 'x' in sig.parameters:
                # DistributedUNetのようにキーワード引数 'x' を持つモデル
                seg_prob = network(x=window_data) # ★ 修正: 不要なkwargsを渡さない
            else:
                # 通常のモデル (第一引数が入力テンソル)
                seg_prob = network(window_data) # ★ 修正: 不要なkwargsを渡さない

            # ★ 修正: seg_probを適切なデバイスに移動
            if seg_prob.device != device:
                seg_prob = seg_prob.to(device)

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