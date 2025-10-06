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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler # noqa: E402
from custom_monai.inferer import SlidingWindowInferer # ★ 変更点: カスタムInfererをインポート
from torchmetrics.image import StructuralSimilarityIndexMeasure
from safetensors.torch import load_file
from functools import partial
import torch

# --- リファクタリングによるインポートの変更 ---
from utils import load_config, set_seed
from data_utils import Preprocessed_CT_DRR_Dataset
from models import DistributedUNet
from custom_models.conditioning_encoder import (
    ConditioningEncoderResNet,
    ConditioningEncoderConvNeXt,
    ConditioningEncoderEfficientNetV2
)
from custom_models.unet import DiffusionModelUNet
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import GradScaler, autocast # ★ 高速化: 混合精度学習のためにインポート

CONFIG = load_config()


# --- ★ 変更点: accelerator を使わない可視化関数 ---
def visualize_and_save_mpr(device, params, scheduler_name, ct_full, drr1, drr2, pos_3d, best_epoch, trial_number, save_dir, model_for_inference):
    print(f"--- Starting visualization on device: {device} (Central Coronal Slice Only) ---")

    vis_dir = Path(save_dir) / "visualizations"
    vis_dir.mkdir(exist_ok=True, parents=True)

    # 1. モデルを評価モードに設定
    is_distributed = isinstance(model_for_inference, DistributedUNet)
    if is_distributed:
        model_for_inference.eval()
        print("  [Visualization] Using provided DistributedUNet model.")
    else: # single GPU mode
        model_for_inference['unet'].eval()
        model_for_inference['conditioning_encoder'].eval()
        print("  [Visualization] Using provided single-GPU models.")

    # 3. スケジューラを選択
    if scheduler_name == "dpm_solver":
        scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000)
    elif scheduler_name == "euler":
        scheduler = EulerDiscreteScheduler(num_train_timesteps=1000)
    else: # ddpm
        scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 4. 推論の準備
    ct_full, drr1, drr2, pos_3d = ct_full.to(device), drr1.to(device), drr2.to(device), pos_3d.to(device)
    
    # ★ 変更点: SlidingWindowInfererを使用してフルボリュームを推論
    patch_size = (params['patch_size'], params['patch_size'], params['patch_size'])
    inferer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=1, overlap=0.5)

    # 5. 推論を実行
    with torch.no_grad():
        initial_noise = torch.randn_like(ct_full)
        scheduler.set_timesteps(num_inference_steps=50)
        image = initial_noise

        for t in tqdm(scheduler.timesteps, desc=f"🖼️ Visualizing Trial {trial_number} (Full Volume)"):
            timesteps_tensor = torch.tensor((t,), device=image.device).long().repeat(image.shape[0])
            
            if is_distributed:
                # 分散モデル用の推論関数
                context = model_for_inference.conditioning_encoder(drr1, drr2)
                model_func = lambda x: model_for_inference(x=x, timesteps=timesteps_tensor, context=context, pos_3d=pos_3d)
                model_output = inferer(inputs=image, network=model_func)
            else:
                # 単一GPUモデル用の推論関数
                context = model_for_inference['conditioning_encoder'](drr1, drr2)
                model_func = lambda x: model_for_inference['unet'](x, timesteps=timesteps_tensor, context=context, pos_3d=pos_3d)
                model_output = inferer(inputs=image, network=model_func)
            
            image = scheduler.step(model_output, t, image).prev_sample

    # 6. 結果をHU値に逆正規化し、描画・保存する
    print("  De-normalizing images to HU range for visualization...")
    
    # ターゲットのHU範囲
    min_hu = -1024
    max_hu = 1500

    # [0, 1] の範囲から [min_hu, max_hu] の範囲にスケール変換
    generated_hu_np = image.squeeze().cpu().numpy() * (max_hu - min_hu) + min_hu
    ground_truth_hu_np = ct_full.squeeze().cpu().numpy() * (max_hu - min_hu) + min_hu

    # 生成されたCTボリュームをNumPy配列として保存
    save_npy_path = vis_dir / f"generated_ct_trial_{trial_number}_epoch_{best_epoch}.npy"
    np.save(save_npy_path, generated_hu_np)
    print(f"  💾 Generated CT volume saved as numpy array: {save_npy_path}")

    z, y, x = ground_truth_hu_np.shape
    slice_ax, slice_cor, slice_sag = z // 2, y // 2, x // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='black')
    plt.suptitle(f'Trial {trial_number} - MPR Visualization (Epoch {best_epoch})', color='white', fontsize=16)

    views = {
        'Axial': (ground_truth_hu_np[slice_ax, :, :], generated_hu_np[slice_ax, :, :]),
        'Coronal': (ground_truth_hu_np[:, slice_cor, :], generated_hu_np[:, slice_cor, :]),
        'Sagittal': (ground_truth_hu_np[:, :, slice_sag], generated_hu_np[:, :, slice_sag])
    }

    # 表示範囲を一般的なCTウィンドウ（肺野）に設定
    vmin, vmax = -1024, 300

    for i, (title, (gt_img, gen_img)) in enumerate(views.items()):
        # --- Ground Truth ---
        if title == 'Axial':
            gt_view = gt_img
        elif title == 'Coronal':
            gt_view = np.flipud(gt_img)
        else: # Sagittal
            gt_view = np.fliplr(np.flipud(gt_img))
        axes[0, i].imshow(gt_view, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'Ground Truth {title}', color='cyan')
        axes[0, i].axis('off')
        
        # --- Generated ---
        if title == 'Axial':
            gen_view = gen_img
        elif title == 'Coronal':
            gen_view = np.flipud(gen_img)
        else: # Sagittal
            gen_view = np.fliplr(np.flipud(gen_img))
        axes[1, i].imshow(gen_view, cmap='gray', vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'Generated {title}', color='magenta')
        axes[1, i].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = vis_dir / f"best_model_vis_trial_{trial_number}_epoch_{best_epoch}_mpr.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, facecolor='black')
    
    wandb.log({"MPR_Visualization": wandb.Image(fig)}, step=CONFIG["EPOCHS"])
    plt.close(fig)
    print(f"  🖼️ MPR visualization saved.")

    # 2. PNGファイルを「アーティファクト」としてアップロード
    # #    これにより、後からファイルをダウンロードできるようになります。
    # if accelerator.is_main_process:
    #     # アーティファクトオブジェクトを作成（名前とタイプを指定）
    #     artifact = wandb.Artifact(
    #         name=f"visualization_trial_{trial_number}", 
    #         type="visualization"
    #     )
    #     # 保存したPNGファイルをアーティファクトに追加
    #     artifact.add_file(str(save_path))
        
    #     # アーティファクトをWandBにログとして記録（アップロード）
    #     accelerator.get_tracker("wandb").log_artifact(artifact)
    #     print(f"  📤 Artifact '{save_path.name}' uploaded to WandB.")


# --- ★ 変更点: accelerator を使わない評価関数 ---
def evaluate_epoch(device, distributed_model, scheduler, val_dataloader):
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
    return np.mean(val_loss_epoch)


# --- 学習・評価関数 ---
def train_and_evaluate(params, trial_number, train_paths, val_paths, encoder_name, loss_phase_epochs, data_config, resume_from_checkpoint=None): # noqa: E501
    BATCH_SIZE = CONFIG["BATCH_SIZE"]
    patch_size_val = params["patch_size"] # configから直接渡される
    PATCH_SIZE = (patch_size_val, patch_size_val, patch_size_val)
    lr = params["learning_rate"]
    weight_decay = params["weight_decay"]
    gradient_accumulation_steps = params["gradient_accumulation_steps"]

    SAVE_PATH = f"./checkpoints/trial_{trial_number}"
    os.makedirs(SAVE_PATH, exist_ok=True)

    l2_end_epoch, l1_end_epoch = loss_phase_epochs

    # --- WandB 初期化 ---
    config = {
        "encoder": encoder_name, "trial_number": trial_number, 
        "loss_schedule": f"L2(->{l2_end_epoch})_L1(->{l1_end_epoch})_SSIM",
        **params
    }
    run_name = f"trial-{trial_number}_enc-{encoder_name}_p-{params['patch_size']}"
    wandb.init(project=CONFIG["PROJECT_NAME"], config=config, name=run_name, reinit=True)
    print(f"--- Starting Run (Trial {trial_number}) | Encoder: {encoder_name} ---")
    print(f"Loss Schedule: L2 until epoch {l2_end_epoch}, then L1 until {l1_end_epoch}, then L1+SSIM")

    # --- ★ 変更点: モデル並列用のデバイス設定 ---
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        raise RuntimeError("This script requires at least 3 GPUs for model parallelism.")
    # メインのデバイスをcuda:0とする
    device = torch.device("cuda:0")
    print(f"Using device: {device}")

    drr_dir = Path(data_config["DRR_DIR"])
    train_dataset = Preprocessed_CT_DRR_Dataset(train_paths, drr_dir, PATCH_SIZE)
    val_dataset = Preprocessed_CT_DRR_Dataset(val_paths, drr_dir, PATCH_SIZE)
    vis_dataset = Preprocessed_CT_DRR_Dataset(val_paths, drr_dir, patch_size=None)  # フルボリューム用
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=CONFIG["NUM_WORKERS"], pin_memory=True) # ★ 高速化: pin_memory=True を追加
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=CONFIG["NUM_WORKERS"], pin_memory=True) # ★ 高速化: pin_memory=True を追加
    vis_dataloader = DataLoader(vis_dataset, batch_size=1, shuffle=False, num_workers=CONFIG["NUM_WORKERS"], pin_memory=True) # ★ 高速化: pin_memory=True を追加
    
    
    # --- ★ 変更点: モデルをCPU上でインスタンス化 ---
    if encoder_name == 'resnet':
        conditioning_encoder = ConditioningEncoderResNet(output_dim=256)
    elif encoder_name == 'convnext':
        conditioning_encoder = ConditioningEncoderConvNeXt(output_dim=256)
    elif encoder_name == 'efficientnet':
        conditioning_encoder = ConditioningEncoderEfficientNetV2(output_dim=256)

    unet = DiffusionModelUNet(
        spatial_dims=3, in_channels=1, out_channels=1, with_conditioning=True,
        num_channels=(32, 64, 128, 256),
        attention_levels=(False, True, True, True),
        num_res_blocks=2,
        cross_attention_dim=conditioning_encoder.feature_dim
    )
    
    # --- ★ 変更点: 分散モデルラッパーを作成し、パラメータを収集 ---
    distributed_model = DistributedUNet(unet, conditioning_encoder)
    # ★ 高速化: torch.compile を適用
    # distributed_model = torch.compile(distributed_model, mode="reduce-overhead")
    # 注意: torch.compileはカスタムのforwardを持つモデルやcheckpointと競合する可能性があるため、まずはコメントアウト。
    # 動作しない場合は、この行を無効にしてください。
    model_params = list(distributed_model.parameters())
    optimizer = torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=400).to(device)
    scaler = GradScaler() # ★ 高速化: GradScalerを初期化

    if resume_from_checkpoint:
        # ★ 変更点: 分散モデルのチェックポイントをロード
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        distributed_model.conditioning_encoder.load_state_dict(torch.load(Path(resume_from_checkpoint) / "conditioning_encoder.pth", map_location=distributed_model.device0))
        distributed_model.time_mlp.load_state_dict(torch.load(Path(resume_from_checkpoint) / "unet_time_mlp.pth", map_location=distributed_model.device0))
        distributed_model.init_conv.load_state_dict(torch.load(Path(resume_from_checkpoint) / "unet_init_conv.pth", map_location=distributed_model.device0))
        
        # 分割されたdown_blocksをロード
        down_blocks_state_dict = torch.load(Path(resume_from_checkpoint) / "unet_down_blocks.pth")
        distributed_model.down_block_0.load_state_dict({k.replace('0.', ''): v for k, v in down_blocks_state_dict.items() if k.startswith('0.')}, strict=False)
        distributed_model.down_block_1.load_state_dict({k.replace('1.', ''): v for k, v in down_blocks_state_dict.items() if k.startswith('1.')}, strict=False)
        distributed_model.down_block_2.load_state_dict({k.replace('2.', ''): v for k, v in down_blocks_state_dict.items() if k.startswith('2.')}, strict=False)
        distributed_model.down_block_3.load_state_dict({k.replace('3.', ''): v for k, v in down_blocks_state_dict.items() if k.startswith('3.')}, strict=False)

        distributed_model.mid_block1.load_state_dict(torch.load(Path(resume_from_checkpoint) / "unet_mid_block1.pth", map_location=distributed_model.device2))
        distributed_model.mid_attn.load_state_dict(torch.load(Path(resume_from_checkpoint) / "unet_mid_attn.pth", map_location=distributed_model.device2))
        distributed_model.mid_block2.load_state_dict(torch.load(Path(resume_from_checkpoint) / "unet_mid_block2.pth", map_location=distributed_model.device2))
        
        # 分割されたup_blocksをロード
        up_blocks_state_dict = torch.load(Path(resume_from_checkpoint) / "unet_up_blocks.pth")
        distributed_model.up_block_0.load_state_dict({k.replace('0.', ''): v for k, v in up_blocks_state_dict.items() if k.startswith('0.')}, strict=False)
        distributed_model.up_block_1.load_state_dict({k.replace('1.', ''): v for k, v in up_blocks_state_dict.items() if k.startswith('1.')}, strict=False)
        distributed_model.up_block_2.load_state_dict({k.replace('2.', ''): v for k, v in up_blocks_state_dict.items() if k.startswith('2.')}, strict=False)
        distributed_model.up_block_3.load_state_dict({k.replace('3.', ''): v for k, v in up_blocks_state_dict.items() if k.startswith('3.')}, strict=False)
        distributed_model.out_conv.load_state_dict(torch.load(Path(resume_from_checkpoint) / "unet_out_conv.pth", map_location=distributed_model.device0))
        distributed_model.pos_mlp_3d.load_state_dict(torch.load(Path(resume_from_checkpoint) / "unet_pos_mlp_3d.pth", map_location=distributed_model.device0)) # 修正済み
        optimizer.load_state_dict(torch.load(Path(resume_from_checkpoint) / "optimizer.pth", map_location=device))
        scaler.load_state_dict(torch.load(Path(resume_from_checkpoint) / "scaler.pth")) # ★ 高速化: scalerの状態もロード

    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(CONFIG["EPOCHS"]):
        distributed_model.train()
        # --- ★ 変更: L2損失に一時的に固定 ---
        # # 元のコード: 現在のエポックに基づいて損失タイプを決定
        # if epoch < l2_end_epoch:
        #     current_loss_type = 'l2'
        # elif epoch < l1_end_epoch:
        #     current_loss_type = 'l1'
        # else:
        #     current_loss_type = 'l1_ssim'
        current_loss_type = 'l2' # L2損失に固定

        train_loss_epoch = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} Training")
        for step, (ct_patch, drr1, drr2, pos_3d) in enumerate(progress_bar):
            ct_patch, drr1, drr2, pos_3d = ct_patch.to(device), drr1.to(device), drr2.to(device), pos_3d.to(device)
            noise = torch.randn_like(ct_patch)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (ct_patch.shape[0],), device=ct_patch.device).long()
            noisy_ct = scheduler.add_noise(original_samples=ct_patch, noise=noise, timesteps=timesteps)

            # ★ 高速化: autocastコンテキスト内でフォワードパスを実行
            with autocast():
                context = distributed_model.conditioning_encoder(drr1, drr2)
                predicted_noise = checkpoint(distributed_model, noisy_ct, timesteps, context, pos_3d, use_reentrant=False)

                # --- ★ 変更: L2損失に一時的に固定 ---
                loss = F.mse_loss(predicted_noise, noise)
                # # 元のコード:
                # if current_loss_type == 'l1':
                #     loss = F.l1_loss(predicted_noise, noise)
                # elif current_loss_type == 'l2':
                #     loss = F.mse_loss(predicted_noise, noise)
                # else: # 'l1_ssim'
                #     l1_loss = F.l1_loss(predicted_noise, noise)
                #     denoised_ct = scheduler.step(predicted_noise, timesteps, noisy_ct).pred_original_sample
                #     ssim_loss = 1.0 - ssim_metric(denoised_ct, ct_patch)
                #     loss = CONFIG["TRAINING"]["L1_SSIM_RATIO"] * l1_loss + (1 - CONFIG["TRAINING"]["L1_SSIM_RATIO"]) * ssim_loss
                
                # 勾配蓄積のために損失をスケーリング
                loss = loss / gradient_accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n🔥 NaN or Inf loss detected at epoch {epoch+1}, step {step}. Aborting trial.")
                return float('inf')

            # ★ 高速化: scalerを使って勾配を計算
            scaler.scale(loss).backward()

            train_loss_epoch += loss.item() * gradient_accumulation_steps # スケールを元に戻して加算

            # 勾配蓄積ステップに達したらパラメータを更新
            if (step + 1) % gradient_accumulation_steps == 0:
                # ★ 高速化: scalerを使って勾配をクリップし、オプティマイザをステップ
                scaler.unscale_(optimizer) # unscaleしてクリッピング
                torch.nn.utils.clip_grad_norm_(model_params, CONFIG["MAX_GRAD_NORM"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        avg_val_loss = evaluate_epoch(device, distributed_model, scheduler, val_dataloader)
        
        avg_train_loss = train_loss_epoch / len(train_dataloader)
        print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch}, step=epoch)

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
            # 分割されたup_blocksを結合して保存
            up_blocks_state_dict = {**{'0.'+k: v for k,v in distributed_model.up_block_0.state_dict().items()}, **{'1.'+k: v for k,v in distributed_model.up_block_1.state_dict().items()}, **{'2.'+k: v for k,v in distributed_model.up_block_2.state_dict().items()}, **{'3.'+k: v for k,v in distributed_model.up_block_3.state_dict().items()}}
            torch.save(up_blocks_state_dict, Path(SAVE_PATH) / "unet_up_blocks.pth")
            torch.save(distributed_model.out_conv.state_dict(), Path(SAVE_PATH) / "unet_out_conv.pth")
            torch.save(distributed_model.pos_mlp_3d.state_dict(), Path(SAVE_PATH) / "unet_pos_mlp_3d.pth") # 修正済み
            torch.save(optimizer.state_dict(), Path(SAVE_PATH) / "optimizer.pth")
            torch.save(scaler.state_dict(), Path(SAVE_PATH) / "scaler.pth") # ★ 高速化: scalerの状態も保存
    
    
    if best_epoch != -1:
        print(f"✨ Generating final visualization for Trial {trial_number} with best weights (from epoch {best_epoch})...")
        try:
            fixed_vis_data = next(iter(vis_dataloader))
            vis_ct_full, vis_drr1, vis_drr2, vis_pos_3d = fixed_vis_data
            
            visualize_and_save_mpr(
                device,
                {"encoder": encoder_name, **params},
                "",
                vis_ct_full,
                vis_drr1,
                vis_drr2,
                vis_pos_3d,
                best_epoch, 
                trial_number, SAVE_PATH,
                model_for_inference=distributed_model # 学習済み分散モデルを渡す
            )
        except Exception as e:
            print(f"An error occurred during final visualization: {e}")
    else:
        print("No best model found to visualize for this trial.")

    wandb.finish()
    
    return best_val_loss

# --- main関数 (ファイル検索とデータ同期を修正) ---
def main(args):
    # コマンドライン引数からresume_from_checkpointのみを取得するように変更
    parser = argparse.ArgumentParser(description="Train a conditioned diffusion model.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint directory to resume training from.")
    args = parser.parse_args()

    # --- cuDNNエラー回避のための設定 ---
    torch.backends.cudnn.benchmark = False
    print("ℹ️ Set torch.backends.cudnn.benchmark = False to avoid potential cuDNN errors.")

    # --- ★ 変更点: configからデータセット設定を読み込む ---
    data_config = CONFIG["DATA"]
    pt_dir = Path(data_config["PT_DATA_DIR"])
    drr_dir = Path(data_config["DRR_DIR"])

    # --- ファイル検索とデータ分割 ---
    print(f"🔍 Searching for all preprocessed tensor files in: {pt_dir}")
    all_pt_files = sorted(list(pt_dir.glob("*.pt")))
    
    verified_file_paths = []
    for ct_path in tqdm(all_pt_files, desc="Verifying data pairs"):
        drr_subdir_name = ct_path.stem
        drr_ap_path = drr_dir / drr_subdir_name / "AP.pt"
        drr_lat_path = drr_dir / drr_subdir_name / "LAT.pt"
        if drr_ap_path.exists() and drr_lat_path.exists():
            verified_file_paths.append(ct_path)
    
    if not verified_file_paths:
        print(f"エラー: 有効なCTとDRRのペアが見つかりません。")
        return
    
    print(f"🔍 Found {len(verified_file_paths)} verified data pairs.")
    train_paths, val_paths = train_test_split(
        verified_file_paths, test_size=CONFIG["VALIDATION_SPLIT"], random_state=CONFIG["SEED"]
    )

    # Optuna学習ループ
    # config.ymlから設定を読み込む
    encoder_name = CONFIG["TRAINING"]["ENCODER"]
    loss_phase_epochs = CONFIG["TRAINING"]["LOSS_PHASE_EPOCHS"]
    
    # --- ★ 変更点: Optunaを無効化し、config.ymlから直接パラメータを読み込む ---
    print("--- Running a single training session (Optuna is disabled) ---")
    
    # パラメータをconfig.ymlから取得
    params = {
        "patch_size": CONFIG["TRAINING"]["PATCH_SIZE"],
        "learning_rate": CONFIG["TRAINING"]["LEARNING_RATE"],
        "weight_decay": CONFIG["TRAINING"]["WEIGHT_DECAY"],
        "gradient_accumulation_steps": CONFIG["TRAINING"]["GRADIENT_ACCUMULATION_STEPS"],
    }
    
    # トライアル番号は0に固定（またはタイムスタンプなど）
    trial_number = 0 
    
    # シードを設定
    set_seed(CONFIG["SEED"])
    print(f"Setting seed to {CONFIG['SEED']}")
    
    # 学習と評価を実行
    result = train_and_evaluate(params, trial_number, train_paths, val_paths, encoder_name, loss_phase_epochs, data_config, args.resume_from_checkpoint)

    print("\n--- Training Finished ---")
    print(f"Final Validation Loss: {result:.4f}")

if __name__ == '__main__':
    set_seed(CONFIG["SEED"])
    # コマンドライン引数からresume_from_checkpointのみを取得するように変更
    parser = argparse.ArgumentParser(description="Train a conditioned diffusion model.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint directory to resume training from.")
    cli_args = parser.parse_args()
    main(cli_args)