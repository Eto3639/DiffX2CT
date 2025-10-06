# ファイル名: models.py

import torch
import torch.nn as nn

class DistributedUNet(nn.Module):
    """
    U-NetとConditioning Encoderを複数のGPUに分散配置し、
    フォワードパスでのデータ転送を管理するラッパークラス。
    """
    def __init__(self, unet, conditioning_encoder):
        super().__init__()
        # デバイスを定義
        self.device0 = torch.device("cuda:0")
        self.device1 = torch.device("cuda:1")
        self.device2 = torch.device("cuda:2")

        # --- 新しい分散戦略に基づいてモデルを再配置 ---
        # GPU 0: 入出力に近い層
        self.conditioning_encoder = conditioning_encoder.to(self.device0)
        self.pos_mlp_3d = unet.pos_mlp_3d.to(self.device0)
        self.time_mlp = unet.time_mlp.to(self.device0)
        self.init_conv = unet.init_conv.to(self.device0)
        self.down_block_0 = unet.down_blocks[0].to(self.device0)
        self.up_block_3 = unet.up_blocks[3].to(self.device0)
        self.out_conv = unet.out_conv.to(self.device0)

        # GPU 1: 中間層
        self.down_block_1 = unet.down_blocks[1].to(self.device1)
        self.down_block_2 = unet.down_blocks[2].to(self.device1)
        self.up_block_1 = unet.up_blocks[1].to(self.device1)
        self.up_block_2 = unet.up_blocks[2].to(self.device1)

        # GPU 2: ボトルネックに近い層
        self.down_block_3 = unet.down_blocks[3].to(self.device2)
        self.mid_block1 = unet.mid_block1.to(self.device2)
        self.mid_attn = unet.mid_attn.to(self.device2)
        self.mid_block2 = unet.mid_block2.to(self.device2)
        self.up_block_0 = unet.up_blocks[0].to(self.device2)

    def forward(self, x, timesteps, context, pos_3d):
        # --- GPU 0 ---
        t = self.time_mlp(timesteps) + self.pos_mlp_3d(pos_3d)
        x = self.init_conv(x)
        
        skip_connections = []
        
        # Down Block 0 on GPU 0
        skip_x0, x = self.down_block_0(x, t, context)
        skip_connections.append(skip_x0)

        # --- Data to GPU 1 ---
        x = x.to(self.device1)
        t = t.to(self.device1)
        context = context.to(self.device1)

        # Down Blocks 1, 2 on GPU 1
        skip_x1, x = self.down_block_1(x, t, context)
        skip_connections.append(skip_x1)
        skip_x2, x = self.down_block_2(x, t, context)
        skip_connections.append(skip_x2)

        # --- Data to GPU 2 ---
        x = x.to(self.device2)
        t = t.to(self.device2)
        context = context.to(self.device2)

        # Down Block 3 and Mid Blocks on GPU 2
        skip_x3, x = self.down_block_3(x, t, context)
        skip_connections.append(skip_x3)
        
        x = self.mid_block1(x, t)
        x = self.mid_attn(x, context)
        x = self.mid_block2(x, t)
        
        # Up Block 0 on GPU 2
        x = self.up_block_0(x, skip_connections.pop().to(self.device2), t, context)

        # --- Data to GPU 1 ---
        x = x.to(self.device1)
        t = t.to(self.device1)
        context = context.to(self.device1)

        # Up Blocks 1, 2 on GPU 1
        # ★ 修正: 正しい呼び出し順序
        x = self.up_block_1(x, skip_connections.pop().to(self.device1), t, context)
        x = self.up_block_2(x, skip_connections.pop().to(self.device1), t, context)

        # --- Data to GPU 0 ---
        x = x.to(self.device0)
        t = t.to(self.device0)
        context = context.to(self.device0)

        # Up Block 3 and Out Conv on GPU 0
        x = self.up_block_3(x, skip_connections.pop().to(self.device0), t, context)
        output = self.out_conv(x)
        return output