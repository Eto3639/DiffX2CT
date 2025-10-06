# ファイル名: custom_models/conditioning_encoder.py (位置埋め込み対応版)

import torch
import torch.nn as nn
import torchvision.models as models
import math
from typing import Tuple

# --- ★ 変更点: 柔軟な2D正弦波位置埋め込み ---
class FlexibleSinusoidalPositionalEmbedding2D(nn.Module):
    """
    2D特徴マップに、指定された軸と方向で正弦波位置埋め込みを追加する。
    """
    def __init__(self, dim: int, dim_mapping: Tuple[str, str] = ('h', 'w')):
        """
        Args:
            dim (int): 埋め込み次元。
            dim_mapping (Tuple[str, str]): 特徴マップの2軸がCTのどの軸に対応するかを指定。
                                           例: ('d', 'w'), ('-d', 'h')。'-'は座標の反転を示す。
        """
        super().__init__()
        self.dim = dim
        self.dim_mapping = dim_mapping

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 形状 (B, C, H, W) の特徴マップ
        """
        b, c, dim1_size, dim2_size = x.shape
        device = x.device
        assert c == self.dim, "The channel dimension of the input must match the embedding dimension"

        div_term = torch.exp(torch.arange(0, self.dim, 2, device=device) * (-math.log(10000.0) / self.dim))

        total_pos_embedding = torch.zeros(dim1_size, dim2_size, self.dim, device=device)

        # 2つの軸に対してループ
        for i, dim_name in enumerate(self.dim_mapping):
            size = dim1_size if i == 0 else dim2_size
            
            # 座標を生成
            positions = torch.arange(size, device=device)
            if dim_name.startswith('-'):
                positions = torch.flip(positions, dims=[0]) # 座標を反転

            # 埋め込みを計算
            pos_emb_slice = positions.unsqueeze(1) * div_term
            pos_emb = torch.zeros(size, self.dim, device=device)
            pos_emb[:, 0::2] = torch.sin(pos_emb_slice)
            pos_emb[:, 1::2] = torch.cos(pos_emb_slice)

            # 正しい形状にブロードキャストして加算
            if i == 0: # Dim1 (e.g., D)
                total_pos_embedding += pos_emb.unsqueeze(1)
            else: # Dim2 (e.g., W or H)
                total_pos_embedding += pos_emb.unsqueeze(0)

        pos_embedding = total_pos_embedding.permute(2, 0, 1).unsqueeze(0)
        return x + pos_embedding

class ConditioningEncoderResNet(nn.Module):
    def __init__(self, output_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])        
        in_channels = 512
        self.projection = nn.Conv2d(in_channels, output_dim, kernel_size=1) if in_channels != output_dim else nn.Identity()
        self.feature_dim = output_dim

        # ★ 変更点: AP用とLAT用に別々の位置埋め込み層を定義
        self.pos_embedding_ap = FlexibleSinusoidalPositionalEmbedding2D(self.feature_dim, dim_mapping=('-d', 'w'))
        self.pos_embedding_lat = FlexibleSinusoidalPositionalEmbedding2D(self.feature_dim, dim_mapping=('-d', 'h'))

    def forward(self, drr1: torch.Tensor, drr2: torch.Tensor) -> torch.Tensor:
        if drr1.shape[1] == 1: drr1 = drr1.repeat(1, 3, 1, 1)
        if drr2.shape[1] == 1: drr2 = drr2.repeat(1, 3, 1, 1)

        batch_size = drr1.shape[0]
        combined_batch = torch.cat([drr1, drr2], dim=0)
        feat_maps = self.feature_extractor(combined_batch)
        projected_maps = self.projection(feat_maps)
        
        # ★ 変更点: APとLATを分離し、それぞれの位置埋め込みを適用
        feat_ap, feat_lat = torch.chunk(projected_maps, 2, dim=0)
        feat_ap_pos = self.pos_embedding_ap(feat_ap)
        feat_lat_pos = self.pos_embedding_lat(feat_lat)
        
        # シーケンスに変換して結合
        seq_ap = feat_ap_pos.flatten(2).permute(0, 2, 1)
        seq_lat = feat_lat_pos.flatten(2).permute(0, 2, 1)
        context_sequence = torch.cat([seq_ap, seq_lat], dim=1)
        return context_sequence

# ConvNeXtとEfficientNetV2も同様に修正
class ConditioningEncoderConvNeXt(nn.Module):
    def __init__(self, output_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = models.convnext_tiny(weights=weights)
        self.feature_extractor = convnext.features
        
        in_channels = 768
        self.projection = nn.Conv2d(in_channels, output_dim, kernel_size=1)
        self.feature_dim = output_dim

        # ★ 変更点: AP用とLAT用に別々の位置埋め込み層を定義
        self.pos_embedding_ap = FlexibleSinusoidalPositionalEmbedding2D(self.feature_dim, dim_mapping=('-d', 'w'))
        self.pos_embedding_lat = FlexibleSinusoidalPositionalEmbedding2D(self.feature_dim, dim_mapping=('-d', 'h'))

    def forward(self, drr1: torch.Tensor, drr2: torch.Tensor) -> torch.Tensor:
        if drr1.shape[1] == 1: drr1 = drr1.repeat(1, 3, 1, 1)
        if drr2.shape[1] == 1: drr2 = drr2.repeat(1, 3, 1, 1)

        batch_size = drr1.shape[0]
        combined_batch = torch.cat([drr1, drr2], dim=0)
        feat_maps = self.feature_extractor(combined_batch)
        projected_maps = self.projection(feat_maps)

        # ★ 変更点: APとLATを分離し、それぞれの位置埋め込みを適用
        feat_ap, feat_lat = torch.chunk(projected_maps, 2, dim=0)
        feat_ap_pos = self.pos_embedding_ap(feat_ap)
        feat_lat_pos = self.pos_embedding_lat(feat_lat)

        seq_ap = feat_ap_pos.flatten(2).permute(0, 2, 1)
        seq_lat = feat_lat_pos.flatten(2).permute(0, 2, 1)
        context_sequence = torch.cat([seq_ap, seq_lat], dim=1)
        return context_sequence

class ConditioningEncoderEfficientNetV2(nn.Module):
    def __init__(self, output_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        efficientnet = models.efficientnet_v2_s(weights=weights)
        self.feature_extractor = efficientnet.features
        
        in_channels = 1280
        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=output_dim, kernel_size=1)
        self.feature_dim = output_dim

        # ★ 変更点: AP用とLAT用に別々の位置埋め込み層を定義
        self.pos_embedding_ap = FlexibleSinusoidalPositionalEmbedding2D(self.feature_dim, dim_mapping=('-d', 'w'))
        self.pos_embedding_lat = FlexibleSinusoidalPositionalEmbedding2D(self.feature_dim, dim_mapping=('-d', 'h'))

    def forward(self, drr1: torch.Tensor, drr2: torch.Tensor) -> torch.Tensor:
        if drr1.shape[1] == 1: drr1 = drr1.repeat(1, 3, 1, 1)
        if drr2.shape[1] == 1: drr2 = drr2.repeat(1, 3, 1, 1)

        batch_size = drr1.shape[0]
        combined_batch = torch.cat([drr1, drr2], dim=0)
        feat_maps = self.feature_extractor(combined_batch)
        projected_maps = self.projection(feat_maps)

        # ★ 変更点: APとLATを分離し、それぞれの位置埋め込みを適用
        feat_ap, feat_lat = torch.chunk(projected_maps, 2, dim=0)
        feat_ap_pos = self.pos_embedding_ap(feat_ap)
        feat_lat_pos = self.pos_embedding_lat(feat_lat)

        seq_ap = feat_ap_pos.flatten(2).permute(0, 2, 1)
        seq_lat = feat_lat_pos.flatten(2).permute(0, 2, 1)
        context_sequence = torch.cat([seq_ap, seq_lat], dim=1)
        return context_sequence