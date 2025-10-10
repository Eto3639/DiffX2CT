# ファイル名: train.py

import os
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np
import optuna
import json
import argparse
import matplotlib.pyplot as plt
import wandb
import gc # ★ 追加: ガベージコレクション
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from torchmetrics.image import StructuralSimilarityIndexMeasure
from functools import partial
import torch

# --- リファクタリングによるインポートの変更 ---
from utils import load_config, set_seed
from data_utils import Preprocessed_CT_DRR_Dataset, GridPatchDatasetWithCond
from models import DistributedUNet
from custom_models.conditioning_encoder import (
    ConditioningEncoderResNet,
    ConditioningEncoderConvNeXt,
    ConditioningEncoderEfficientNetV2
)
from custom_models.unet import DiffusionModelUNet
from torch.utils.checkpoint import checkpoint

# --- ★ 追加: 分離したユーティリティ関数をインポート ---
from evaluation_utils import (
    EMA,
    calculate_mae,
    generate_and_evaluate,
    create_evaluation_report
)

CONFIG = load_config()



def evaluate_epoch(device, distributed_model, scheduler, val_dataloader, trial=None, epoch=0):
    # 評価時はEMAの重みを使用する
    distributed_model.ema.apply_shadow()
    distributed_model.eval()
    val_loss_epoch = []
    with torch.no_grad():
        for ct_patch, drr1, drr2, pos_3d in val_dataloader:
            ct_patch, drr1, drr2, pos_3d = ct_patch.to(device), drr1.to(device), drr2.to(device), pos_3d.to(device)
            noise = torch.randn_like(ct_patch)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (ct_patch.shape[0],), device=ct_patch.device).long()
            noisy_ct = scheduler.add_noise(original_samples=ct_patch, noise=noise, timesteps=timesteps)

            # ★ 変更点: 分散モデルで推論
            context = distributed_model.conditioning_encoder(drr1, drr2)
            predicted_noise = distributed_model(x=noisy_ct, timesteps=timesteps, context=context, pos_3d=pos_3d)
            loss = F.l1_loss(predicted_noise, noise)
            val_loss_epoch.append(loss.item())
    # モデルの重みを元に戻す
    distributed_model.ema.restore()
    avg_val_loss = np.mean(val_loss_epoch)

    # --- ★ 追加: Optunaの枝刈り(Pruning)機能 ---
    if trial:
        trial.report(avg_val_loss, epoch)

    return avg_val_loss


# --- 学習・評価関数 ---
def train_and_evaluate(params, trial, train_paths, val_paths, loss_phase_epochs, data_config, resume_from_checkpoint=None): # noqa: E501
    start_epoch = 0 # ★ 変更: 開始エポックを定義
    wandb_run_id = None # ★ 追加: WandB再開用のID
    
    # --- ★ 変更: モデルサイズに応じてバッチサイズを動的に調整 ---
    base_batch_size = CONFIG["BATCH_SIZE"]
    model_scale = params["model_scale"]
    if model_scale == "medium":
        # パラメータ数が約2倍になると仮定し、バッチサイズを半分に
        BATCH_SIZE = max(1, base_batch_size // 2)
    elif model_scale == "large":
        # パラメータ数が約4倍になると仮定し、バッチサイズを1/4に
        BATCH_SIZE = max(1, base_batch_size // 4)
    else: # small
        BATCH_SIZE = base_batch_size
    
    print(f"ℹ️ Model scale: '{model_scale}', Base BATCH_SIZE: {base_batch_size}, Adjusted BATCH_SIZE: {BATCH_SIZE}")
    # ---------------------------------------------------------

    # --- ★ 変更: configからパッチサイズとボリュームサイズを直接読み込む ---
    # これにより、サイズ関連の設定がconfig.ymlに集約されていることが明確になります。
    patch_size_val = CONFIG["TRAINING"]["PATCH_SIZE"]
    target_volume_size = CONFIG["DATA"]["TARGET_VOLUME_SIZE"]

    PATCH_SIZE = (patch_size_val, patch_size_val, patch_size_val)
    lr = params["learning_rate"]
    weight_decay = params["weight_decay"]
    gradient_accumulation_steps = params["gradient_accumulation_steps"]

    SAVE_PATH = f"./checkpoints/trial_{trial.number}"
    os.makedirs(SAVE_PATH, exist_ok=True)

    # --- ★ 追加: パッチサイズとボリュームサイズの検証 ---
    for i in range(3):
        if PATCH_SIZE[i] > target_volume_size[i]:
            raise ValueError(f"設定エラー: PATCH_SIZE[{i}] ({PATCH_SIZE[i]}) が TARGET_VOLUME_SIZE[{i}] ({target_volume_size[i]}) を超えています。config.ymlを確認してください。")

    # --- ★ 変更点: モデル並列用のデバイス設定 ---
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        raise RuntimeError("This script requires at least 3 GPUs for model parallelism.")
    # メインのデバイスをcuda:0とする
    device = torch.device("cuda:0")
    print(f"Using device: {device}")

    drr_dir = Path(data_config["DRR_DIR"])
    # --- ★ 変更点: GridPatchDatasetを使用 ---
    patch_overlap_ratio = params.get('patch_overlap') or 0.0 # configにキーがないか、値がNoneの場合に0.0にフォールバック
    patch_overlap = tuple(int(p * patch_overlap_ratio) for p in PATCH_SIZE)

    train_dataset = GridPatchDatasetWithCond(
        data=train_paths, drr_dir=drr_dir, patch_size=PATCH_SIZE, patch_overlap=patch_overlap
    )
    val_dataset = GridPatchDatasetWithCond(
        data=val_paths, drr_dir=drr_dir, patch_size=PATCH_SIZE, patch_overlap=patch_overlap
    )
    vis_dataset = Preprocessed_CT_DRR_Dataset(val_paths, drr_dir, patch_size=None)  # フルボリューム用
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=CONFIG["NUM_WORKERS"], pin_memory=True) # ★ 高速化: pin_memory=True を追加
    vis_dataloader = DataLoader(vis_dataset, batch_size=1, shuffle=False, num_workers=CONFIG["NUM_WORKERS"], pin_memory=True) # ★ 高速化: pin_memory=True を追加
    
    # --- ★ 変更: configからモデル設定を読み込む ---
    model_config = CONFIG.get("MODEL", {})
    unet_config = model_config.get("UNET", {})
    cond_enc_config = model_config.get("CONDITIONING_ENCODER", {})

    # --- ★ 変更: Optunaのパラメータに基づいてU-Netの構成を決定 ---
    if model_scale == "small":
        num_channels = unet_config.get("NUM_CHANNELS", (32, 64, 128, 256))
        attention_levels = unet_config.get("ATTENTION_LEVELS", (False, True, True, True))
    elif model_scale == "medium":
        num_channels = (48, 96, 192, 384) 
        attention_levels = (False, True, True, True)
    else: # large
        num_channels = (64, 128, 256, 512)
        attention_levels = (False, True, True, True)
    
    num_res_blocks = unet_config.get("NUM_RES_BLOCKS", 2)
    encoder_name = params["encoder"]
    # ---------------------------------------------------------

    # --- ★ 変更点: モデルをCPU上でインスタンス化 ---
    if encoder_name == 'resnet':
        conditioning_encoder = ConditioningEncoderResNet(output_dim=cond_enc_config.get("OUTPUT_DIM", 256))
    elif encoder_name == 'convnext':
        conditioning_encoder = ConditioningEncoderConvNeXt(output_dim=cond_enc_config.get("OUTPUT_DIM", 256))
    elif encoder_name == 'efficientnet':
        conditioning_encoder = ConditioningEncoderEfficientNetV2(output_dim=cond_enc_config.get("OUTPUT_DIM", 256))

    # --- ★ 変更: 動的なモデル構成パラメータを渡す ---
    unet = DiffusionModelUNet(
        spatial_dims=3, in_channels=1, out_channels=1, with_conditioning=True,
        num_channels=tuple(num_channels),
        attention_levels=tuple(attention_levels),
        num_res_blocks=num_res_blocks,
        cross_attention_dim=conditioning_encoder.feature_dim
    )
    
    # --- ★ 変更点: 分散モデルラッパーを作成し、パラメータを収集 ---
    distributed_model = DistributedUNet(unet, conditioning_encoder)
    # --- ★ 追加: EMAモデルの初期化 ---
    ema_model = EMA(distributed_model, decay=model_config.get("EMA_DECAY", 0.9999))
    distributed_model.ema = ema_model # distributed_modelからアクセスできるようにする
    # --- ★ パフォーマンス改善案: torch.compile を適用 ---
    # PyTorch 2.0以降のJITコンパイラでモデルを最適化し、高速化します。
    # print("🚀 Applying torch.compile() for performance optimization...")
    # distributed_model = torch.compile(distributed_model, mode="reduce-overhead") # ★ 修正: inductor backendとの互換性問題のため一時的に無効化
    model_params = list(distributed_model.parameters())
    # --- ★ 変更: AMP安定化のため、epsを調整 ---
    optimizer = torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95), eps=1e-6)

    # --- ★ 変更点: スケジューラをconfigに基づいて設定 ---
    scheduler_name = params.get('scheduler_name', 'linear')
    if scheduler_name == 'cosine':
        beta_schedule = "squaredcos_cap_v2"
    else: # 'linear' or default
        beta_schedule = "linear"
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule=beta_schedule)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    # --- ★ 変更: 混合精度(AMP)を再度有効化 ---
    scaler = torch.cuda.amp.GradScaler()

    if resume_from_checkpoint:
        checkpoint_path = Path(resume_from_checkpoint)
        # ★ 変更点: 分散モデルのチェックポイントをロード
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        distributed_model.conditioning_encoder.load_state_dict(torch.load(Path(resume_from_checkpoint) / "conditioning_encoder.pth", map_location=distributed_model.device0))
        distributed_model.time_mlp.load_state_dict(torch.load(Path(resume_from_checkpoint) / "unet_time_mlp.pth", map_location=distributed_model.device0))
        distributed_model.init_conv.load_state_dict(torch.load(Path(resume_from_checkpoint) / "unet_init_conv.pth", map_location=distributed_model.device0))
        
        # 分割されたdown_blocksをロード
        down_blocks_state_dict = torch.load(checkpoint_path / "unet_down_blocks.pth")
        distributed_model.down_block_0.load_state_dict({k.replace('0.', ''): v for k, v in down_blocks_state_dict.items() if k.startswith('0.')}, strict=False)
        distributed_model.down_block_1.load_state_dict({k.replace('1.', ''): v for k, v in down_blocks_state_dict.items() if k.startswith('1.')}, strict=False)
        distributed_model.down_block_2.load_state_dict({k.replace('2.', ''): v for k, v in down_blocks_state_dict.items() if k.startswith('2.')}, strict=False)
        distributed_model.down_block_3.load_state_dict({k.replace('3.', ''): v for k, v in down_blocks_state_dict.items() if k.startswith('3.')}, strict=False)

        distributed_model.mid_block1.load_state_dict(torch.load(Path(resume_from_checkpoint) / "unet_mid_block1.pth", map_location=distributed_model.device2))
        distributed_model.mid_attn.load_state_dict(torch.load(Path(resume_from_checkpoint) / "unet_mid_attn.pth", map_location=distributed_model.device2))
        distributed_model.mid_block2.load_state_dict(torch.load(Path(resume_from_checkpoint) / "unet_mid_block2.pth", map_location=distributed_model.device2))
        
        # 分割されたup_blocksをロード
        up_blocks_state_dict = torch.load(checkpoint_path / "unet_up_blocks.pth")
        distributed_model.up_block_0.load_state_dict({k.replace('0.', ''): v for k, v in up_blocks_state_dict.items() if k.startswith('0.')}, strict=False)
        distributed_model.up_block_1.load_state_dict({k.replace('1.', ''): v for k, v in up_blocks_state_dict.items() if k.startswith('1.')}, strict=False)
        distributed_model.up_block_2.load_state_dict({k.replace('2.', ''): v for k, v in up_blocks_state_dict.items() if k.startswith('2.')}, strict=False)
        distributed_model.up_block_3.load_state_dict({k.replace('3.', ''): v for k, v in up_blocks_state_dict.items() if k.startswith('3.')}, strict=False)
        distributed_model.out_conv.load_state_dict(torch.load(Path(resume_from_checkpoint) / "unet_out_conv.pth", map_location=distributed_model.device0))
        optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pth", map_location=device))
        # EMAの重みもロード
        ema_checkpoint_path = checkpoint_path / "ema_model.pth"
        if ema_checkpoint_path.exists():
            ema_model.load_state_dict(torch.load(ema_checkpoint_path, map_location=device))
        # --- ★ 変更: scalerの状態もロード ---
        scaler.load_state_dict(torch.load(Path(resume_from_checkpoint) / "scaler.pth"))

        # ★ 変更: エポック数とbest_val_lossをロード
        info_path = checkpoint_path / "checkpoint_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                checkpoint_info = json.load(f)
            start_epoch = checkpoint_info.get('epoch', 0)
            best_val_loss = checkpoint_info.get('best_val_loss', float('inf'))
            wandb_run_id = checkpoint_info.get('wandb_run_id', None) # ★ 追加: WandB IDをロード
            print(f"  -> Resuming from epoch {start_epoch} with best_val_loss {best_val_loss:.4f}")
            if wandb_run_id:
                print(f"  -> Resuming WandB run with ID: {wandb_run_id}")
        else:
            print("  -> Warning: checkpoint_info.json not found. Resuming from epoch 0.")

    best_val_loss = float('inf')
    best_epoch = -1

    # --- WandB 初期化 (チェックポイント読み込み後) ---
    l2_end_epoch, l1_end_epoch = loss_phase_epochs
    config = {
        "encoder": encoder_name, "trial_number": trial.number, 
        "loss_schedule": f"L2(->{l2_end_epoch})_L1(->{l1_end_epoch})_SSIM",
        **params
    }
    run_name = f"trial-{trial.number}_scale-{model_scale}_enc-{encoder_name}_p-{params['patch_size']}"
    
    # ★ 変更点: 再開時はIDを指定し、新規時はIDを生成
    wandb.init(
        project=CONFIG["PROJECT_NAME"], 
        config=config, 
        name=run_name, 
        id=wandb_run_id, # 再開時はIDを指定、新規ならNone
        resume="must" if wandb_run_id else None # 再開時は"must"を指定
    )
    if wandb_run_id is None:
        wandb_run_id = wandb.run.id # 新規実行時に生成されたIDを保存

    # --- ★ 追加: WandBでモデルの勾配とパラメータを監視 ---
    wandb.watch(distributed_model, log="all", log_freq=CONFIG["TRAINING"].get("VISUALIZATION_FREQ", 10) * len(train_dataloader))

    # --- ★ 変更: 勾配爆発のデバッグのため、Anomaly Detectionを有効化 ---
    # これによりNaN/Infを生成した操作のスタックトレースが出力されますが、学習は遅くなります。
    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(start_epoch, CONFIG["EPOCHS"]):
        distributed_model.train()
        # --- ★ 修正: 元の損失関数スケジュールに戻す ---
        # 現在のエポックに基づいて損失タイプを決定
        if epoch < l2_end_epoch:
            current_loss_type = 'l2'
        elif epoch < l1_end_epoch:
            current_loss_type = 'l1'
        else:
            current_loss_type = 'l1_ssim'

        train_loss_epoch = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} Training")
        for step, (ct_patch, drr1, drr2, pos_3d) in enumerate(progress_bar):
            ct_patch, drr1, drr2, pos_3d = ct_patch.to(device), drr1.to(device), drr2.to(device), pos_3d.to(device)
            global_step = epoch * len(train_dataloader) + step
            noise = torch.randn_like(ct_patch)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (ct_patch.shape[0],), device=ct_patch.device).long()
            noisy_ct = scheduler.add_noise(original_samples=ct_patch, noise=noise, timesteps=timesteps)

            # --- ★ 変更: autocastを有効化 ---
            with torch.cuda.amp.autocast():
                context = distributed_model.conditioning_encoder(drr1, drr2)
                predicted_noise = checkpoint(distributed_model, noisy_ct, timesteps, context, pos_3d, use_reentrant=False)
                loss_details = {}
                
                # --- ★ 修正: 損失関数のスケジュールロジックを正しく反映 ---
                if current_loss_type == 'l1':
                    loss = F.l1_loss(predicted_noise, noise)
                elif current_loss_type == 'l2':
                    loss = F.mse_loss(predicted_noise, noise)
                else: # 'l1_ssim'
                    # --- ★ 変更: L1, SSIM, フーリエ損失を組み合わせる ---
                    weights = CONFIG["TRAINING"].get("LOSS_WEIGHTS", {"L1": 0.5, "SSIM": 0.5, "FOURIER": 0.0})

                    # 1. ピクセル空間でのL1損失
                    loss_details["l1_loss"] = F.l1_loss(predicted_noise, noise)

                    # 2. 知覚的損失 (SSIM)
                    denoised_ct = scheduler.step(predicted_noise, timesteps, noisy_ct).pred_original_sample
                    loss_details["ssim_loss"] = 1.0 - ssim_metric(denoised_ct, ct_patch)

                    # 3. 周波数空間でのL1損失 (フーリエ損失)
                    fft_predicted = torch.fft.fftn(predicted_noise, dim=[-3, -2, -1])
                    fft_target = torch.fft.fftn(noise, dim=[-3, -2, -1])
                    loss_details["fourier_loss"] = F.l1_loss(torch.abs(fft_predicted), torch.abs(fft_target))

                    # 4. 各損失を重み付けして合計
                    loss = (weights["L1"] * loss_details["l1_loss"] + 
                            weights["SSIM"] * loss_details["ssim_loss"] + 
                            weights["FOURIER"] * loss_details["fourier_loss"])

            # 勾配蓄積のために損失をスケーリング
            scaled_loss = loss / gradient_accumulation_steps

            if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                # --- ★ 変更: エラーをWandBに記録し、無限大の損失を返してOptunaに失敗を伝える ---
                print(f"\n🔥 NaN or Inf loss detected at epoch {epoch+1}, step {step}. Pruning trial.")
                # --- ★ 変更: 発散時の詳細ログ ---
                log_data = {
                    "train_loss": float('inf'), 
                    "val_loss": float('inf'), 
                    "epoch": epoch, 
                    "status": "pruned_nan_loss",
                    **{f"nan_loss_detail/{k}": v.item() if torch.is_tensor(v) else v for k, v in loss_details.items()}
                }
                wandb.log(log_data, step=global_step)
                wandb.summary["status"] = "pruned_nan_loss"
                wandb.summary.update({f"final_{k}": v for k, v in params.items()})
                wandb.summary.update({f"final_loss_weight_{k}": v for k, v in CONFIG["TRAINING"]["LOSS_WEIGHTS"].items()})
                # Optunaにこのトライアルが失敗したことを伝えるために大きな値を返す
                wandb.finish(exit_code=1) # 終了コード1でWandBを終了
                raise optuna.exceptions.TrialPruned()
            
            # --- ★ 変更: scalerを使って勾配を計算 ---
            scaler.scale(scaled_loss).backward()

            train_loss_epoch += loss.item() * gradient_accumulation_steps # スケールを元に戻して加算

            # 勾配蓄積ステップに達したらパラメータを更新
            if (step + 1) % gradient_accumulation_steps == 0:
                # --- ★ 変更: 勾配クリッピングをunscale後に行う ---
                # 1. 勾配をunscaleする
                scaler.unscale_(optimizer)
                
                # 2. 勾配をクリッピングする (inf/nanチェックも兼ねる)
                grad_norm = torch.nn.utils.clip_grad_norm_(model_params, CONFIG["MAX_GRAD_NORM"])
                wandb.log({"details/grad_norm": grad_norm.item()}, step=global_step)

                # 3. optimizerを更新 (scalerがinf/nanを検知したらスキップする)
                scaler.step(optimizer)
                
                # 4. scalerを更新
                scaler.update()
                ema_model.update()
                optimizer.zero_grad()
            
            # --- ★ 追加: ステップごとの詳細な損失をログ ---
            if (step + 1) % gradient_accumulation_steps == 0:
                log_step_data = {"details/step_loss": loss.item()}
                if loss_details:
                    log_step_data.update({f"details/loss_{k}": v.item() for k, v in loss_details.items()})
                wandb.log(log_step_data, step=global_step)
            
            # --- ★ 追加: メモリ解放 ---
            # ステップごとに不要になったテンソルを明示的に削除
            del ct_patch, drr1, drr2, pos_3d, noise, noisy_ct, context, predicted_noise, loss, scaled_loss
            if 'denoised_ct' in locals(): del denoised_ct
            if 'fft_predicted' in locals(): del fft_predicted, fft_target
        
        avg_val_loss = evaluate_epoch(device, distributed_model, scheduler, val_dataloader, trial, epoch)
        
        avg_train_loss = train_loss_epoch / len(train_dataloader)
        print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        # --- ★ 変更: エポックごとのログに学習率を追加 ---
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch,
                   "learning_rate": optimizer.param_groups[0]['lr']}, step=global_step)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            print(f"  ✨ New best val loss: {best_val_loss:.4f} at epoch {best_epoch}. Saving model to {SAVE_PATH}")
            # ★ 変更点: 分散モデルの各パーツを保存
            torch.save(distributed_model.conditioning_encoder.state_dict(), Path(SAVE_PATH) / "conditioning_encoder.pth")
            torch.save(distributed_model.time_mlp.state_dict(), Path(SAVE_PATH) / "unet_time_mlp.pth")
            torch.save(distributed_model.init_conv.state_dict(), Path(SAVE_PATH) / "unet_init_conv.pth")
            # 分割されたdown_blocksを結合して保存
            down_blocks_state_dict = {**{'0.'+k: v for k,v in distributed_model.down_block_0.state_dict().items()}, **{'1.'+k: v for k,v in distributed_model.down_block_1.state_dict().items()}, **{'2.'+k: v for k,v in distributed_model.down_block_2.state_dict().items()}, **{'3.'+k: v for k,v in distributed_model.down_block_3.state_dict().items()}}
            torch.save(down_blocks_state_dict, Path(SAVE_PATH) / "unet_down_blocks.pth")
            torch.save(distributed_model.mid_block1.state_dict(), Path(SAVE_PATH) / "unet_mid_block1.pth")
            torch.save(distributed_model.mid_attn.state_dict(), Path(SAVE_PATH) / "unet_mid_attn.pth")
            torch.save(distributed_model.mid_block2.state_dict(), Path(SAVE_PATH) / "unet_mid_block2.pth")
            up_blocks_state_dict = {**{'0.'+k: v for k,v in distributed_model.up_block_0.state_dict().items()}, **{'1.'+k: v for k,v in distributed_model.up_block_1.state_dict().items()}, **{'2.'+k: v for k,v in distributed_model.up_block_2.state_dict().items()}, **{'3.'+k: v for k,v in distributed_model.up_block_3.state_dict().items()}} # 分割されたup_blocksを結合して保存
            torch.save(up_blocks_state_dict, Path(SAVE_PATH) / "unet_up_blocks.pth") # 修正済み
            torch.save(distributed_model.out_conv.state_dict(), Path(SAVE_PATH) / "unet_out_conv.pth")
            torch.save(ema_model.state_dict(), Path(SAVE_PATH) / "ema_model.pth") # ★ 追加: EMAの重みを保存
            torch.save(optimizer.state_dict(), Path(SAVE_PATH) / "optimizer.pth")
            torch.save(scaler.state_dict(), Path(SAVE_PATH) / "scaler.pth") # ★ 変更: 混合精度を有効化

            # ★ 変更: エポック数と検証ロスを保存
            checkpoint_info = {
                'epoch': best_epoch,
                'best_val_loss': best_val_loss,
                'wandb_run_id': wandb_run_id # ★ 追加: WandBのRun IDを保存
            }
            with open(Path(SAVE_PATH) / "checkpoint_info.json", 'w') as f:
                json.dump(checkpoint_info, f, indent=4)

        # --- ★ 追加: 定期的な可視化プロセス ---
        visualization_freq = CONFIG["TRAINING"].get("VISUALIZATION_FREQ", 0)
        # visualization_freqが0より大きい場合、その頻度で実行
        # または、最終エポックの場合に実行
        if (visualization_freq > 0 and (epoch + 1) % visualization_freq == 0) or ((epoch + 1) == CONFIG["EPOCHS"]):
            print(f"--- 🖼️ Running visualization for epoch {epoch + 1} ---")
            # --- ★ 変更: 可視化直前にデータをロード ---
            try:
                fixed_vis_data = next(iter(vis_dataloader))
                vis_ct_full, vis_drr1, vis_drr2, vis_pos_3d = fixed_vis_data
                generate_and_evaluate(
                    device,
                    {"encoder": encoder_name, **params},
                    "dpm_solver", # 可視化には高速なスケジューラを使用
                    vis_ct_full,
                    vis_drr1,
                    vis_drr2,
                    vis_pos_3d,
                    epoch + 1, # 現在のエポック番号を渡す
                    trial.number, SAVE_PATH,
                    model_for_inference=distributed_model
                )
            except Exception as e:
                print(f"❌ An error occurred during periodic visualization at epoch {epoch + 1}: {e}")
            finally:
                # --- ★ 追加: 可視化データのメモリ解放 ---
                if 'fixed_vis_data' in locals(): del fixed_vis_data
                if 'vis_ct_full' in locals(): del vis_ct_full, vis_drr1, vis_drr2, vis_pos_3d
    
        # --- ★ 追加: エポック終了時にメモリをクリーンアップ ---
        gc.collect()
        torch.cuda.empty_cache()
    
    if best_epoch != -1:
        print(f"✨ Generating final visualization for Trial {trial.number} with best weights (from epoch {best_epoch})...")
        try:
            fixed_vis_data = next(iter(vis_dataloader))
            vis_ct_full, vis_drr1, vis_drr2, vis_pos_3d = fixed_vis_data

            generate_and_evaluate(
                device,
                {"encoder": encoder_name, **params},
                "", # defaultのDDPMスケジューラを使用
                vis_ct_full,
                vis_drr1,
                vis_drr2,
                vis_pos_3d,
                best_epoch, 
                trial.number, SAVE_PATH,
                model_for_inference=distributed_model, # ★ 修正: 辞書ではなく、DistributedUNetインスタンスを直接渡す
                inference_steps=200 # 最終評価ではステップ数を増やして高品質化
           )
        except Exception as e:
            print(f"An error occurred during final visualization: {e}")
    else:
        print("No best model found to visualize for this trial.")

    wandb.finish()
    
    return best_val_loss

# --- main関数 (ファイル検索とデータ同期を修正) ---
def main(args):
    # --- cuDNNエラー回避のための設定 ---
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    print("ℹ️ Set torch.backends.cudnn.benchmark = False to avoid potential cuDNN errors.")

    # --- ★ 変更点: configからデータセット設定を読み込む ---
    data_config = CONFIG["DATA"]
    pt_dir = Path(data_config["PT_DATA_DIR"])
    drr_dir = Path(data_config["DRR_DIR"])

    # --- ★ 変更点: ファイル検証をキャッシュする ---
    cache_file = Path("./verified_paths.json")

    if cache_file.exists():
        print(f"✅ Loading verified data paths from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        train_paths = [Path(p) for p in cached_data['train']]
        val_paths = [Path(p) for p in cached_data['val']]
        print(f"  -> Found {len(train_paths)} training paths and {len(val_paths)} validation paths.")
    else:
        print(f"🔍 No cache file found. Verifying all data pairs (this will run only once)...")
        print(f"   Searching for all preprocessed tensor files in: {pt_dir}")
        all_pt_files = sorted(list(pt_dir.glob("*.pt")))
        
        verified_file_paths = []
        for ct_path in tqdm(all_pt_files, desc="Verifying data pairs"):
            drr_subdir_name = ct_path.stem
            drr_ap_path = drr_dir / drr_subdir_name / "AP.pt"
            drr_lat_path = drr_dir / drr_subdir_name / "LAT.pt"
            if drr_ap_path.exists() and drr_lat_path.exists():
                verified_file_paths.append(ct_path)
        
        if not verified_file_paths:
            print(f"エラー: 有効なCTとDRRのペアが見つかりません。処理を中止します。")
            return
        
        print(f"🔍 Found {len(verified_file_paths)} verified data pairs.")
        train_paths, val_paths = train_test_split(
            verified_file_paths, test_size=CONFIG["VALIDATION_SPLIT"], random_state=CONFIG["SEED"]
        )
        with open(cache_file, 'w') as f:
            json.dump({'train': [str(p) for p in train_paths], 'val': [str(p) for p in val_paths]}, f, indent=4)
        print(f"💾 Saved verified paths to cache file: {cache_file}")

    # --- ★ 追加: config.ymlのDATA_RATIOに基づいてデータセットを削減 ---
    data_ratio = CONFIG["DATA"]["DATA_RATIO"]
    if data_ratio < 1.0:
        print(f"⚠️ Using only {data_ratio * 100:.1f}% of the dataset for training and validation based on DATA_RATIO in config.")
        
        num_train = int(len(train_paths) * data_ratio)
        num_val = int(len(val_paths) * data_ratio)
        
        train_paths = train_paths[:num_train]
        val_paths = val_paths[:num_val]
        
        print(f"  -> Reduced to {len(train_paths)} training samples and {len(val_paths)} validation samples.")

    # Optuna学習ループ
    if args.evaluate_only:
        # --- 評価モード ---
        if not args.checkpoint_dir:
            raise ValueError("--evaluate_only mode requires --checkpoint_dir to be specified.")
        
        print(f"\n--- Running in EVALUATION-ONLY mode for checkpoint: {args.checkpoint_dir} ---")
        set_seed(CONFIG["SEED"])
        
        encoder_name = CONFIG["TRAINING"]["ENCODER"]
        # 1. デバイスとデータローダーの準備
        device = torch.device("cuda:0")
        vis_dataset = Preprocessed_CT_DRR_Dataset(val_paths, drr_dir, patch_size=None)
        vis_dataloader = DataLoader(vis_dataset, batch_size=1, shuffle=False, num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
        fixed_vis_data = next(iter(vis_dataloader))
        vis_ct_full, vis_drr1, vis_drr2, vis_pos_3d = fixed_vis_data

        # 評価時はconfig.ymlのTRAININGセクションのパラメータを使用
        params = {
            "patch_size": CONFIG["TRAINING"]["PATCH_SIZE"],
            "patch_overlap": CONFIG["TRAINING"].get("patch_overlap", 0.5),
            "blend_mode": CONFIG["TRAINING"].get("blend_mode", "cosine"),
            # 以下はgenerate_and_evaluateで直接は使われないが、念のため設定
            "learning_rate": CONFIG["TRAINING"]["LEARNING_RATE"],
            "weight_decay": CONFIG["TRAINING"]["WEIGHT_DECAY"],
        }
        # 2. モデルのインスタンス化
        if encoder_name == 'resnet':
            conditioning_encoder = ConditioningEncoderResNet(output_dim=256)
        elif encoder_name == 'convnext':
            conditioning_encoder = ConditioningEncoderConvNeXt(output_dim=256)
        else: # efficientnet
            conditioning_encoder = ConditioningEncoderEfficientNetV2(output_dim=256)

        unet_full = DiffusionModelUNet(
            spatial_dims=3, in_channels=1, out_channels=1, with_conditioning=True,
            num_channels=(32, 64, 128, 256), attention_levels=(False, True, True, True),
            num_res_blocks=2, cross_attention_dim=conditioning_encoder.feature_dim
        )
        distributed_model = DistributedUNet(unet_full, conditioning_encoder)

        # 3. チェックポイントから重みをロード
        print(f"Loading model weights from {args.checkpoint_dir}...")
        checkpoint_path = Path(args.checkpoint_dir)
        distributed_model.conditioning_encoder.load_state_dict(torch.load(checkpoint_path / "conditioning_encoder.pth", map_location=distributed_model.device0))
        distributed_model.time_mlp.load_state_dict(torch.load(checkpoint_path / "unet_time_mlp.pth", map_location=distributed_model.device0))
        distributed_model.init_conv.load_state_dict(torch.load(checkpoint_path / "unet_init_conv.pth", map_location=distributed_model.device0))
        down_blocks_state_dict = torch.load(checkpoint_path / "unet_down_blocks.pth")
        distributed_model.down_block_0.load_state_dict({k.replace('0.', ''): v for k, v in down_blocks_state_dict.items() if k.startswith('0.')}, strict=False)
        distributed_model.down_block_1.load_state_dict({k.replace('1.', ''): v for k, v in down_blocks_state_dict.items() if k.startswith('1.')}, strict=False)
        distributed_model.down_block_2.load_state_dict({k.replace('2.', ''): v for k, v in down_blocks_state_dict.items() if k.startswith('2.')}, strict=False)
        distributed_model.down_block_3.load_state_dict({k.replace('3.', ''): v for k, v in down_blocks_state_dict.items() if k.startswith('3.')}, strict=False)
        distributed_model.mid_block1.load_state_dict(torch.load(checkpoint_path / "unet_mid_block1.pth", map_location=distributed_model.device2))
        distributed_model.mid_attn.load_state_dict(torch.load(checkpoint_path / "unet_mid_attn.pth", map_location=distributed_model.device2))
        distributed_model.mid_block2.load_state_dict(torch.load(checkpoint_path / "unet_mid_block2.pth", map_location=distributed_model.device2))
        up_blocks_state_dict = torch.load(checkpoint_path / "unet_up_blocks.pth")
        distributed_model.up_block_0.load_state_dict({k.replace('0.', ''): v for k, v in up_blocks_state_dict.items() if k.startswith('0.')}, strict=False)
        distributed_model.up_block_1.load_state_dict({k.replace('1.', ''): v for k, v in up_blocks_state_dict.items() if k.startswith('1.')}, strict=False)
        distributed_model.up_block_2.load_state_dict({k.replace('2.', ''): v for k, v in up_blocks_state_dict.items() if k.startswith('2.')}, strict=False)
        distributed_model.up_block_3.load_state_dict({k.replace('3.', ''): v for k, v in up_blocks_state_dict.items() if k.startswith('3.')}, strict=False)
        distributed_model.out_conv.load_state_dict(torch.load(checkpoint_path / "unet_out_conv.pth", map_location=distributed_model.device0))
        # EMAモデルのロード
        ema_model = EMA(distributed_model, decay=0.9999)
        ema_checkpoint_path = checkpoint_path / "ema_model.pth"
        if ema_checkpoint_path.exists():
            ema_model.load_state_dict(torch.load(ema_checkpoint_path, map_location=device))
        
        with open(checkpoint_path / "checkpoint_info.json", 'r') as f:
            best_epoch = json.load(f).get('epoch', 'N/A')

        # 4. 評価関数を実行 (trial_numberは0で固定)
        generate_and_evaluate(device, {"encoder": encoder_name, **params}, "", vis_ct_full, vis_drr1, vis_drr2, vis_pos_3d, best_epoch, 0, str(checkpoint_path), model_for_inference=distributed_model)
        print("\n--- Evaluation Finished ---")
    else:
        # --- ★ 変更: Optunaによる学習モード ---
        set_seed(CONFIG["SEED"])
        print(f"--- Running Optuna hyperparameter search (Seed: {CONFIG['SEED']}) ---")

        # 固定パラメータをconfigから読み込む
        loss_phase_epochs = CONFIG["TRAINING"]["LOSS_PHASE_EPOCHS"]
        optuna_params = CONFIG["OPTUNA"]["PARAMS"]

        def objective(trial):
            # Optunaからハイパーパラメータを提案
            params = {
                "encoder": trial.suggest_categorical("encoder", optuna_params["encoder"]),
                "model_scale": trial.suggest_categorical("model_scale", optuna_params["model_scale"]),
                "patch_size": trial.suggest_categorical("patch_size", optuna_params["patch_size"]),
                # --- ★ 修正: Optunaの非推奨APIを新しいAPIに変更 ---
                "learning_rate": trial.suggest_float("learning_rate", *optuna_params["learning_rate"], log=True),
                "weight_decay": trial.suggest_float("weight_decay", *optuna_params["weight_decay"], log=True),
                "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", optuna_params["gradient_accumulation_steps"]),
                "patch_overlap": CONFIG["TRAINING"]["patch_overlap"], # ★ 変更: configから固定値を読み込む
                "blend_mode": trial.suggest_categorical("blend_mode", optuna_params["blend_mode"]),
                "scheduler_name": trial.suggest_categorical("scheduler_name", optuna_params["scheduler_name"]),
            }
            
            # 損失の重みを提案し、合計が1になるように正規化
            l1 = trial.suggest_float("loss_weight_l1", *optuna_params["loss_weight_l1"])
            ssim = trial.suggest_float("loss_weight_ssim", *optuna_params["loss_weight_ssim"])
            fourier = trial.suggest_float("loss_weight_fourier", *optuna_params["loss_weight_fourier"])
            total_weight = l1 + ssim + fourier
            CONFIG["TRAINING"]["LOSS_WEIGHTS"] = {
                "L1": l1 / total_weight,
                "SSIM": ssim / total_weight,
                "FOURIER": fourier / total_weight,
            }

            # --- ★ 変更: エラーハンドリングを追加 ---
            try:
                # train_and_evaluateはチェックポイントからの再開をサポート
                # Optunaは通常、各トライアルを独立して実行するため、ここではresume_from_checkpoint=Noneとする
                # もしトライアル自体の再開を実装したい場合は、より高度な状態管理が必要
                result = train_and_evaluate(params, trial, train_paths, val_paths, loss_phase_epochs, data_config, resume_from_checkpoint=None)
                
                # --- ★ 追加: 枝刈りのチェック ---
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                return result
            except optuna.exceptions.TrialPruned:
                print(f"🍃 Trial {trial.number} pruned. 🍃")
                return float('inf')
            except Exception as e:
                print(f"🔥🔥🔥 Trial {trial.number} failed with an exception: {e} 🔥🔥🔥")
                # エラーが発生したトライアルをFAILとして記録し、次のトライアルに進む
                # WandBにも失敗を記録
                # --- ★ 変更: OOMエラーを明示的にログ ---
                error_msg = str(e)
                is_oom = "out of memory" in error_msg.lower()
                run = wandb.init(project=CONFIG["PROJECT_NAME"], name=f"trial-{trial.number}-{'OOM' if is_oom else 'FAILED'}", config=params, reinit=True)
                wandb.log({"status": "oom" if is_oom else "failed", "error_message": error_msg})
                # --- ★ 追加: 失敗時のハイパーパラメータをサマリーに記録 ---
                wandb.summary["status"] = "oom" if is_oom else "failed"
                wandb.summary.update({f"failed_param_{k}": v for k, v in trial.params.items()})
                run.finish(exit_code=1)
                return float('inf') # Optunaに大きな損失値を返して失敗を伝える

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=CONFIG["OPTUNA"]["N_TRIALS"])
        print("\n--- Optuna Search Finished ---")
        print(f"Best trial: {study.best_trial.value}")
        print(f"Best params: {study.best_trial.params}")

if __name__ == '__main__':
    set_seed(CONFIG["SEED"])
    # コマンドライン引数からresume_from_checkpointのみを取得するように変更
    parser = argparse.ArgumentParser(description="Train or evaluate a conditioned diffusion model.")
    parser.add_argument("--trial_number", type=int, default=0,
                        help="The trial number for this run, used for naming checkpoints and outputs.")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to a checkpoint directory to resume training from or for evaluation.")
    parser.add_argument("--evaluate_only", action="store_true",
                        help="If specified, skips training and runs evaluation on the model in --checkpoint_dir.")
    cli_args = parser.parse_args()
    main(cli_args)