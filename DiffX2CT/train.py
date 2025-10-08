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
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from custom_monai.inferer import SlidingWindowInferer # ★ 変更点: カスタムInfererをインポート
from torchmetrics.image import StructuralSimilarityIndexMeasure
from safetensors.torch import load_file
from functools import partial
import torch

# --- リファクタリングによるインポートの変更 ---
from skimage.metrics import peak_signal_noise_ratio as psnr # ★ 追加: PSNR計算のため
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
from torch.cuda.amp import GradScaler, autocast # ★ 高速化: 混合精度学習のためにインポート

CONFIG = load_config()

class EMA:
    """
    モデルパラメータの指数移動平均を管理するクラス。
    """
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        self.register()

    def register(self):
        """モデルのパラメータをシャドウパラメータとして登録する。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """現在のモデルの重みを使ってシャドウパラメータを更新する。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """推論のために、現在のモデルの重みをシャドウパラメータに置き換える。"""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """バックアップしておいた元のモデルの重みに戻す。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def calculate_mae(image_true, image_test):
    """平均絶対誤差 (MAE) を計算する"""
    return np.mean(np.abs(image_true - image_test))


# --- ★ 変更点: accelerator を使わない可視化関数 ---
def generate_and_evaluate(device, params, scheduler_name, ct_full, drr1, drr2, pos_3d, best_epoch, trial_number, save_dir, model_for_inference):
    print(f"--- Starting Generation & Evaluation on device: {device} ---")

    vis_dir = Path(save_dir) / "evaluation" # 保存先ディレクトリ名を変更
    vis_dir.mkdir(exist_ok=True, parents=True)

    # 1. モデルを評価モードに設定
    is_distributed = isinstance(model_for_inference, DistributedUNet)
    if is_distributed:
        model_for_inference.eval()
        print("  [Visualization] Using provided DistributedUNet model.")
    else: # single GPU mode (visualization.py用)
        model_for_inference['unet'].eval()
        model_for_inference['conditioning_encoder'].eval()
        print("  [Visualization] Using provided single-GPU models.")

    # EMAモデルの重みを適用
    if 'ema' in model_for_inference:
        model_for_inference['ema'].apply_shadow()
    
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
    inferer = SlidingWindowInferer(
        roi_size=patch_size, 
        sw_batch_size=1, 
        overlap=params.get('patch_overlap', 0.5), # configからoverlapを取得
        mode=params.get('blend_mode', 'cosine')) # configからblend_modeを取得

    # 5. 推論を実行
    with torch.no_grad(), autocast(enabled=False): # 推論時は混合精度をオフ
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

    # EMAモデルの重みを元に戻す
    if 'ema' in model_for_inference:
        model_for_inference['ema'].restore()

    # 6. 結果をHU値に逆正規化
    # 6a. 推論結果を正規化範囲 [0, 1] にクリッピング
    #     拡散モデルの出力は範囲外の値を取りうるため、クリッピングが不可欠
    image = torch.clamp(image, 0, 1)
    ct_full = torch.clamp(ct_full, 0, 1)

    print("  De-normalizing images to HU range...")
    # ターゲットのHU範囲
    min_hu = -1024
    max_hu = 1500

    # 6b. [0, 1] の範囲から [min_hu, max_hu] の範囲にスケール変換
    generated_hu_np = image.squeeze().cpu().numpy() * (max_hu - min_hu) + min_hu
    ground_truth_hu_np = ct_full.squeeze().cpu().numpy() * (max_hu - min_hu) + min_hu

    # 生成されたCTボリュームをNumPy配列として保存
    save_npy_path = vis_dir / f"generated_ct_trial_{trial_number}_epoch_{best_epoch}_HU.npy"
    np.save(save_npy_path, generated_hu_np)
    print(f"  💾 Generated CT volume saved as numpy array: {save_npy_path}")

    # 7. 品質評価指標を計算
    print("  📊 Calculating quality metrics...")
    data_range = max_hu - min_hu # 評価範囲を固定
    ssim_score = StructuralSimilarityIndexMeasure(data_range=data_range)(torch.from_numpy(generated_hu_np).unsqueeze(0).unsqueeze(0), torch.from_numpy(ground_truth_hu_np).unsqueeze(0).unsqueeze(0)).item()
    psnr_score = psnr(ground_truth_hu_np, generated_hu_np, data_range=data_range)
    mae_score = calculate_mae(ground_truth_hu_np, generated_hu_np)
    print(f"  -> SSIM={ssim_score:.4f}, PSNR={psnr_score:.2f} dB, MAE={mae_score:.2f} HU")

    # 9. 評価レポートを生成
    print("  🖼️ Creating evaluation report...")
    fig = create_evaluation_report(
        generated_hu_np=generated_hu_np,
        ground_truth_hu_np=ground_truth_hu_np,
        ssim_score=ssim_score,
        psnr_score=psnr_score,
        mae_score=mae_score,
        title_prefix=f'Evaluation Report for Trial {trial_number} (Epoch {best_epoch})'
    )
    
    save_path = vis_dir / f"evaluation_report_trial_{trial_number}_epoch_{best_epoch}.png"
    plt.savefig(save_path, facecolor='black')
    wandb.log({"Evaluation_Report": wandb.Image(fig)}, step=CONFIG["EPOCHS"])
    plt.close(fig)
    print(f"  ✅ Evaluation report saved to: {save_path}")


def create_evaluation_report(generated_hu_np, ground_truth_hu_np, ssim_score, psnr_score, mae_score, title_prefix):
    """
    生成されたCTと正解CTを比較する詳細な評価レポート（画像）を生成します。

    Args:
        generated_hu_np (np.ndarray): 生成されたCTボリューム（HU値）
        ground_truth_hu_np (np.ndarray): 正解のCTボリューム（HU値）
        ssim_score (float): SSIMスコア
        psnr_score (float): PSNRスコア
        mae_score (float): MAEスコア
        title_prefix (str): 図のタイトルのプレフィックス

    Returns:
        matplotlib.figure.Figure: 生成されたレポートのFigureオブジェクト
    """
    # 統計情報を計算 (空気以外)
    non_air_voxels_gen = generated_hu_np[generated_hu_np > -1000]
    stats_gen = {
        'mean': np.mean(non_air_voxels_gen), 'std': np.std(non_air_voxels_gen),
        'min': np.min(non_air_voxels_gen), 'max': np.max(non_air_voxels_gen)
    } if non_air_voxels_gen.size > 0 else {}

    non_air_voxels_gt = ground_truth_hu_np[ground_truth_hu_np > -1000]
    stats_gt = {
        'mean': np.mean(non_air_voxels_gt), 'std': np.std(non_air_voxels_gt),
        'min': np.min(non_air_voxels_gt), 'max': np.max(non_air_voxels_gt)
    } if non_air_voxels_gt.size > 0 else {}

    fig = plt.figure(figsize=(20, 14), facecolor='black')
    gs = plt.GridSpec(3, 4, figure=fig)
    
    z, y, x = ground_truth_hu_np.shape
    slice_ax, slice_cor, slice_sag = z // 2, y // 2, x // 2
    vmin, vmax = -1024, 300 # 表示ウィンドウ

    views = {
        'Axial': (ground_truth_hu_np[slice_ax, :, :], generated_hu_np[slice_ax, :, :], gs[0, 0], gs[0, 1]),
        'Coronal': (np.flipud(ground_truth_hu_np[:, slice_cor, :]), np.flipud(generated_hu_np[:, slice_cor, :]), gs[1, 0], gs[1, 1]),
        'Sagittal': (np.fliplr(np.flipud(ground_truth_hu_np[:, :, slice_sag])), np.fliplr(np.flipud(generated_hu_np[:, :, slice_sag])), gs[2, 0], gs[2, 1])
    }

    for title, (gt_img, gen_img, gs_gt, gs_gen) in views.items():
        ax_gt = fig.add_subplot(gs_gt)
        ax_gt.imshow(gt_img, cmap='gray', vmin=vmin, vmax=vmax)
        ax_gt.set_title(f'Ground Truth {title}', color='cyan')
        ax_gt.axis('off')

        ax_gen = fig.add_subplot(gs_gen)
        ax_gen.imshow(gen_img, cmap='gray', vmin=vmin, vmax=vmax)
        ax_gen.set_title(f'Generated {title}', color='magenta')
        ax_gen.axis('off')

    # 8. 統計情報を計算 (空気以外)
    print("  ⏳ Calculating statistics (excluding air)...")
    non_air_voxels_gen = generated_hu_np[generated_hu_np > -1000]
    stats_gen = {
        'mean': np.mean(non_air_voxels_gen), 'std': np.std(non_air_voxels_gen),
        'min': np.min(non_air_voxels_gen), 'max': np.max(non_air_voxels_gen)
    } if non_air_voxels_gen.size > 0 else {}

    non_air_voxels_gt = ground_truth_hu_np[ground_truth_hu_np > -1000]
    stats_gt = {
        'mean': np.mean(non_air_voxels_gt), 'std': np.std(non_air_voxels_gt),
        'min': np.min(non_air_voxels_gt), 'max': np.max(non_air_voxels_gt)
    } if non_air_voxels_gt.size > 0 else {}

    # ヒストグラム
    ax_hist_gt = fig.add_subplot(gs[0, 2]); ax_hist_gen = fig.add_subplot(gs[0, 3])
    if stats_gt: ax_hist_gt.hist(non_air_voxels_gt.flatten(), bins=100, color='deepskyblue')
    ax_hist_gt.set_title("Ground Truth - HU Histogram", color='cyan'); ax_hist_gt.set_facecolor('darkgray'); ax_hist_gt.tick_params(colors='white')
    if stats_gen: ax_hist_gen.hist(non_air_voxels_gen.flatten(), bins=100, color='orchid')
    ax_hist_gen.set_title("Generated - HU Histogram", color='magenta'); ax_hist_gen.set_facecolor('darkgray'); ax_hist_gen.tick_params(colors='white')

    # 統計情報とスコア
    ax_text = fig.add_subplot(gs[1:, 2:]); ax_text.axis('off')
    report_text = (
        f"--- Quality Metrics ---\n"
        f"  SSIM: {ssim_score:.4f}\n"
        f"  PSNR: {psnr_score:.2f} dB\n"
        f"  MAE:  {mae_score:.2f} HU\n\n"
        f"--- Statistics (HU, > -1000) ---\n"
        f"  [Ground Truth]\n"
        f"    Mean / Std: {stats_gt.get('mean', 0):.2f} / {stats_gt.get('std', 0):.2f}\n"
        f"    Min / Max:  {stats_gt.get('min', 0):.0f} / {stats_gt.get('max', 0):.0f}\n\n"
        f"  [Generated]\n"
        f"    Mean / Std: {stats_gen.get('mean', 0):.2f} / {stats_gen.get('std', 0):.2f}\n"
        f"    Min / Max:  {stats_gen.get('min', 0):.0f} / {stats_gen.get('max', 0):.0f}\n"
    )
    ax_text.text(0.05, 0.7, report_text, color='white', fontfamily='monospace', fontsize=14, va='top')

    fig.suptitle(title_prefix, color='white', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# --- ★ 変更点: accelerator を使わない評価関数 ---
def evaluate_epoch(device, distributed_model, scheduler, val_dataloader):
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
    return np.mean(val_loss_epoch)


# --- 学習・評価関数 ---
def train_and_evaluate(params, trial_number, train_paths, val_paths, encoder_name, loss_phase_epochs, data_config, resume_from_checkpoint=None): # noqa: E501
    start_epoch = 0 # ★ 変更: 開始エポックを定義
    wandb_run_id = None # ★ 追加: WandB再開用のID
    BATCH_SIZE = CONFIG["BATCH_SIZE"]
    patch_size_val = params["patch_size"] # configから直接渡される
    PATCH_SIZE = (patch_size_val, patch_size_val, patch_size_val)
    lr = params["learning_rate"]
    weight_decay = params["weight_decay"]
    gradient_accumulation_steps = params["gradient_accumulation_steps"]

    SAVE_PATH = f"./checkpoints/trial_{trial_number}"
    os.makedirs(SAVE_PATH, exist_ok=True)

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
    # --- ★ 追加: EMAモデルの初期化 ---
    ema_model = EMA(distributed_model, decay=0.9999)
    distributed_model.ema = ema_model # distributed_modelからアクセスできるようにする
    # ★ 高速化: torch.compile を適用
    # distributed_model = torch.compile(distributed_model, mode="reduce-overhead")
    # 注意: torch.compileはカスタムのforwardを持つモデルやcheckpointと競合する可能性があるため、まずはコメントアウト。
    # 動作しない場合は、この行を無効にしてください。
    model_params = list(distributed_model.parameters())
    optimizer = torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    # --- ★ 変更点: スケジューラをconfigに基づいて設定 ---
    scheduler_name = params.get('scheduler_name', 'linear')
    if scheduler_name == 'cosine':
        beta_schedule = "squaredcos_cap_v2"
    else: # 'linear' or default
        beta_schedule = "linear"
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule=beta_schedule)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    # scaler = GradScaler() # ★ 変更: 混合精度を無効化

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
        # scaler.load_state_dict(torch.load(Path(resume_from_checkpoint) / "scaler.pth")) # ★ 変更: 混合精度を無効化

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
        "encoder": encoder_name, "trial_number": trial_number, 
        "loss_schedule": f"L2(->{l2_end_epoch})_L1(->{l1_end_epoch})_SSIM",
        **params
    }
    run_name = f"trial-{trial_number}_enc-{encoder_name}_p-{params['patch_size']}"
    
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

    # --- ★ 変更: 勾配爆発のデバッグのため、Anomaly Detectionを有効化 ---
    # これによりNaN/Infを生成した操作のスタックトレースが出力されますが、学習は遅くなります。
    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(start_epoch, CONFIG["EPOCHS"]):
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

            # ★ 変更: autocastを無効化し、float32で計算
            # with autocast():
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

            # ★ 変更: scalerを使わずに勾配を計算
            loss.backward()
            # scaler.scale(loss).backward()

            train_loss_epoch += loss.item() * gradient_accumulation_steps # スケールを元に戻して加算

            # 勾配蓄積ステップに達したらパラメータを更新
            if (step + 1) % gradient_accumulation_steps == 0:
                # ★ 変更: scaler関連の処理をコメントアウト
                # scaler.unscale_(optimizer)
                
                # --- ★ 変更: 勾配クリッピングをより強力にする ---
                # ノルム(L2ノルム)でのクリッピングに加えて、値自体もクリッピングする
                torch.nn.utils.clip_grad_value_(model_params, clip_value=1.0)
                torch.nn.utils.clip_grad_norm_(model_params, CONFIG["MAX_GRAD_NORM"])

                # ★ 変更: scalerを使わずにoptimizerを更新
                optimizer.step()
                # scaler.step(optimizer)
                # scaler.update()
                # --- ★ 追加: EMAの更新 ---
                ema_model.update()
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
            up_blocks_state_dict = {**{'0.'+k: v for k,v in distributed_model.up_block_0.state_dict().items()}, **{'1.'+k: v for k,v in distributed_model.up_block_1.state_dict().items()}, **{'2.'+k: v for k,v in distributed_model.up_block_2.state_dict().items()}, **{'3.'+k: v for k,v in distributed_model.up_block_3.state_dict().items()}} # 分割されたup_blocksを結合して保存
            torch.save(up_blocks_state_dict, Path(SAVE_PATH) / "unet_up_blocks.pth") # 修正済み
            torch.save(distributed_model.out_conv.state_dict(), Path(SAVE_PATH) / "unet_out_conv.pth")
            torch.save(ema_model.state_dict(), Path(SAVE_PATH) / "ema_model.pth") # ★ 追加: EMAの重みを保存
            torch.save(optimizer.state_dict(), Path(SAVE_PATH) / "optimizer.pth")
            # torch.save(scaler.state_dict(), Path(SAVE_PATH) / "scaler.pth") # ★ 変更: 混合精度を無効化

            # ★ 変更: エポック数と検証ロスを保存
            checkpoint_info = {
                'epoch': best_epoch,
                'best_val_loss': best_val_loss,
                'wandb_run_id': wandb_run_id # ★ 追加: WandBのRun IDを保存
            }
            with open(Path(SAVE_PATH) / "checkpoint_info.json", 'w') as f:
                json.dump(checkpoint_info, f, indent=4)
    
    
    if best_epoch != -1:
        print(f"✨ Generating final visualization for Trial {trial_number} with best weights (from epoch {best_epoch})...")
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
                trial_number, SAVE_PATH,
                model_for_inference={'unet': distributed_model, 'ema': ema_model} # 学習済み分散モデルとEMAを渡す
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

    # Optuna学習ループ
    # config.ymlから設定を読み込む
    encoder_name = CONFIG["TRAINING"]["ENCODER"]
    loss_phase_epochs = CONFIG["TRAINING"]["LOSS_PHASE_EPOCHS"]
    
    # --- ★ 変更点: Optunaを無効化し、config.ymlから直接パラメータを読み込む ---
    print("--- Running a single training session (Optuna is disabled) ---")
    
    # パラメータをconfig.ymlから取得 (Optuna無効化のため)
    params = {
        "patch_size": CONFIG["TRAINING"]["PATCH_SIZE"],
        "learning_rate": CONFIG["TRAINING"]["LEARNING_RATE"],
        "weight_decay": CONFIG["TRAINING"]["WEIGHT_DECAY"],
        "gradient_accumulation_steps": CONFIG["TRAINING"]["GRADIENT_ACCUMULATION_STEPS"],
        "patch_overlap": CONFIG["TRAINING"].get("patch_overlap", 0.5), # configから取得、なければ0.5
        "blend_mode": CONFIG["TRAINING"].get("blend_mode", "cosine"), # configから取得、なければcosine
    }
    
    # トライアル番号をコマンドライン引数から取得
    trial_number = args.trial_number
    
    if args.evaluate_only:
        # --- 評価モード ---
        if not args.checkpoint_dir:
            raise ValueError("--evaluate_only mode requires --checkpoint_dir to be specified.")
        
        print(f"\n--- Running in EVALUATION-ONLY mode for checkpoint: {args.checkpoint_dir} ---")
        set_seed(CONFIG["SEED"])
        
        # 1. デバイスとデータローダーの準備
        device = torch.device("cuda:0")
        vis_dataset = Preprocessed_CT_DRR_Dataset(val_paths, drr_dir, patch_size=None)
        vis_dataloader = DataLoader(vis_dataset, batch_size=1, shuffle=False, num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
        fixed_vis_data = next(iter(vis_dataloader))
        vis_ct_full, vis_drr1, vis_drr2, vis_pos_3d = fixed_vis_data

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

        # 4. 評価関数を実行
        generate_and_evaluate(device, {"encoder": encoder_name, **params}, "", vis_ct_full, vis_drr1, vis_drr2, vis_pos_3d, best_epoch, trial_number, str(checkpoint_path), model_for_inference=distributed_model)
        print("\n--- Evaluation Finished ---")
    else:
        # --- 学習モード ---
        set_seed(CONFIG["SEED"])
        print(f"--- Running in TRAINING mode (Seed: {CONFIG['SEED']}) ---")
        result = train_and_evaluate(params, trial_number, train_paths, val_paths, encoder_name, loss_phase_epochs, data_config, args.checkpoint_dir)
        print("\n--- Training Finished ---")
        print(f"Final Best Validation Loss: {result:.4f}")

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