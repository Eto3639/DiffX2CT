# ファイル名: custom_models/unet.py (完全実装版)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange

# -----------------------------------------------------------------------------
# ビルディングブロック (基本的な構成要素)
# -----------------------------------------------------------------------------

class SinusoidalPositionEmbeddings3D(nn.Module):
    """
    入力された3つの1D座標ベクトルから、3Dの正弦波位置埋め込みマップを生成する。
    出力は (B, C, D, H, W) の形状を持つ。
    """
    def __init__(self, dim):
        super().__init__()
        assert dim % 6 == 0, "Embedding dimension must be divisible by 6"
        self.dim_per_axis = dim // 3

    def forward(self, pos_3d_vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_3d_vectors (torch.Tensor): 形状 (B, 3, N) の座標ベクトル (d, h, w)
        """
        device = pos_3d_vectors.device
        b, _, n = pos_3d_vectors.shape
        half_dim = self.dim_per_axis // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        all_embs = []
        for i in range(3): # d, h, w
            axis_coords = pos_3d_vectors[:, i, :] # (B, N)
            axis_embs = axis_coords.unsqueeze(-1) * embeddings.unsqueeze(0) # (B, N, half_dim)
            axis_embs = torch.cat([axis_embs.sin(), axis_embs.cos()], dim=-1) # (B, N, dim_per_axis)
            all_embs.append(axis_embs)

        emb_d, emb_h, emb_w = all_embs
        # (B, N, C) -> (B, C, N)
        emb_d = emb_d.permute(0, 2, 1).view(b, self.dim_per_axis, n, 1, 1)
        emb_h = emb_h.permute(0, 2, 1).view(b, self.dim_per_axis, 1, n, 1)
        emb_w = emb_w.permute(0, 2, 1).view(b, self.dim_per_axis, 1, 1, n)

        # ブロードキャストを利用して (B, C, D, H, W) のマップを生成
        # ★ 修正: 各軸の埋め込みを足し算ではなく、チャンネル方向に結合する
        emb_d_map = emb_d.repeat(1, 1, 1, n, n)
        emb_h_map = emb_h.repeat(1, 1, n, 1, n)
        emb_w_map = emb_w.repeat(1, 1, n, n, 1)
        return torch.cat([emb_d_map, emb_h_map, emb_w_map], dim=1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2))
            if time_emb_dim is not None
            else None
        )
        self.block1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act1 = nn.SiLU()
        self.block2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.res_conv = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.act1(self.norm1(self.block1(x)))
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            scale, shift = time_emb.chunk(2, dim=1)
            h = h * (scale + 1) + shift
        h = self.act2(self.norm2(self.block2(h)))
        return h + self.res_conv(x)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.scale = dim_head**-0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)

class AttentionBlock3D(nn.Module):
    def __init__(self, channels, num_heads=8, num_head_channels=32, context_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.attention = CrossAttention(channels, context_dim=context_dim)
        self.proj_out = nn.Conv3d(channels, channels, 1)

    def forward(self, x, context=None):
        b, c, d, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = rearrange(x, "b c d h w -> b (d h w) c")
        x = self.attention(x, context)
        x = rearrange(x, "b (d h w) c -> b c d h w", d=d, h=h, w=w)
        x = self.proj_out(x) + x_in
        return x

class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_res_blocks, use_attention, attention_dim):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.use_attention = use_attention
        
        current_channels = in_channels
        for _ in range(num_res_blocks):
            self.resnets.append(ResnetBlock3D(current_channels, out_channels, time_emb_dim=time_emb_dim))
            current_channels = out_channels
            if use_attention:
                self.attentions.append(AttentionBlock3D(out_channels, context_dim=attention_dim))
        
        self.downsampler = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t_emb, context=None): # contextをオプショナルに
        skip_outputs = []
        for resnet, attn in zip(self.resnets, self.attentions if self.use_attention else [None]*len(self.resnets)):
            x = resnet(x, t_emb)
            if self.use_attention and attn is not None:
                x = attn(x, context)
            skip_outputs.append(x) # 各ResNetブロックの出力をスキップ接続候補として保存
        
        # ★ 変更点: スキップ接続として渡すのは、ダウンサンプル前の最後の特徴マップ
        skip_x = skip_outputs[-1]
        x = self.downsampler(x)
        return skip_x, x

class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_emb_dim: int, num_res_blocks: int, use_attention: bool, attention_dim: int | None):
        super().__init__()
        self.upsampler = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.use_attention = use_attention
        
        resnet_in_channels = in_channels + skip_channels
        for _ in range(num_res_blocks):
            self.resnets.append(ResnetBlock3D(resnet_in_channels, out_channels, time_emb_dim=time_emb_dim))
            resnet_in_channels = out_channels
            if use_attention:
                self.attentions.append(AttentionBlock3D(out_channels, context_dim=attention_dim))

    def forward(self, x, skip_x, t_emb, context=None): # contextをオプショナルに
        x = self.upsampler(x)
        x = torch.cat([x, skip_x], dim=1)
        
        # forループ内のロジックを修正
        if self.use_attention:
            for resnet, attn in zip(self.resnets, self.attentions):
                x = resnet(x, t_emb)
                x = attn(x, context)
        else:
            for resnet in self.resnets:
                x = resnet(x, t_emb)
        return x

# -----------------------------------------------------------------------------
# 完全なU-Netモデル
# -----------------------------------------------------------------------------

class DiffusionModelUNet(nn.Module):
    def __init__(
        self,
        spatial_dims=3, in_channels=1, out_channels=1,
        num_channels=(32, 64, 128, 256),
        attention_levels=(False, False, True, True),
        num_res_blocks=2,
        num_head_channels=32,
        with_conditioning=True,
        cross_attention_dim=512,
    ):
        super().__init__()

        # 時刻埋め込み
        time_embed_dim = num_channels[0] * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(num_channels[0]),
            nn.Linear(num_channels[0], time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # ★ 追加: 3D位置埋め込み
        # チャンネル数を入力チャンネル数に合わせる
        # ★ 修正: 6で割り切れるように次元を調整
        base_pos_embed_dim = num_channels[0]
        pos_embed_dim = (base_pos_embed_dim + 5) // 6 * 6

        self.pos_emb_3d = SinusoidalPositionEmbeddings3D(dim=pos_embed_dim)

        # 入力層
        # ★ 変更: 入力チャンネル数を位置埋め込み分だけ増やす
        self.init_conv = nn.Conv3d(in_channels + pos_embed_dim, num_channels[0], kernel_size=3, padding=1)
        
        # Encoder (Down blocks)
        self.down_blocks = nn.ModuleList()
        ch_in = num_channels[0]
        for i, (ch_out, use_attn) in enumerate(zip(num_channels, attention_levels)):
            self.down_blocks.append(
                DownBlock3D(ch_in, ch_out, time_embed_dim, num_res_blocks, use_attn, cross_attention_dim if with_conditioning and use_attn else None)
            )
            ch_in = ch_out

        # Bottleneck
        self.mid_block1 = ResnetBlock3D(num_channels[-1], num_channels[-1], time_emb_dim=time_embed_dim)
        self.mid_attn = AttentionBlock3D(num_channels[-1], context_dim=cross_attention_dim if with_conditioning else None)
        self.mid_block2 = ResnetBlock3D(num_channels[-1], num_channels[-1], time_emb_dim=time_embed_dim)

        # --- ★ 変更点: Decoder (Up blocks)のチャンネル数計算を完全に修正 ---
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(num_channels)) # [256, 128, 64, 32]
        reversed_attention_levels = list(reversed(attention_levels))

        # 最初のUpBlockの入力チャンネル数はBottleneckの出力チャンネル数
        ch_in = reversed_channels[0] # 256
        
        for i in range(len(reversed_channels)):
            # スキップ接続で結合されるチャンネル数を取得
            # Encoderの最後のブロックから順に対応する
            skip_ch = num_channels[len(num_channels) - 1 - i]
            
            # このブロックの出力チャンネル数を取得
            ch_out = reversed_channels[i+1] if i < len(reversed_channels) - 1 else num_channels[0]
            
            use_attn = reversed_attention_levels[i]
            
            self.up_blocks.append(
                UpBlock3D(
                    in_channels=ch_in,
                    skip_channels=skip_ch,
                    out_channels=ch_out,
                    time_emb_dim=time_embed_dim,
                    num_res_blocks=num_res_blocks, # ここをオリジナルの値に戻す
                    use_attention=use_attn,
                    attention_dim=cross_attention_dim if with_conditioning and use_attn else None
                )
            )
            # 次のUpBlockの入力チャンネル数は、このブロックの出力チャンネル数
            ch_in = ch_out
        # -----------------------------------------------------------------

        # 出力層
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=num_channels[0]),
            nn.SiLU(),
            nn.Conv3d(num_channels[0], out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps, context=None, pos_3d=None):
        # 時間埋め込み
        t = self.time_mlp(timesteps)

        # ★ 追加: 3D位置埋め込みを計算し、時間埋め込みに加算
        # ★ 変更: 3D位置埋め込みを計算し、入力xに結合する
        if pos_3d is not None:
            p = self.pos_emb_3d(pos_3d)
            x = torch.cat([x, p], dim=1)

        x = self.init_conv(x)
        
        skip_connections = []
        for down_block in self.down_blocks:
            skip_x, x = down_block(x, t, context=context)
            skip_connections.append(skip_x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x, context)
        x = self.mid_block2(x, t)

        for up_block in self.up_blocks:
            skip_x = skip_connections.pop()
            x = up_block(x, skip_x, t, context=context)

        return self.out_conv(x)