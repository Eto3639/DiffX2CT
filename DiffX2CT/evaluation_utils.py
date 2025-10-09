# ファイル名: evaluation_utils.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import wandb

from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from torchmetrics.image import StructuralSimilarityIndexMeasure
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.cuda.amp import autocast

from inferers import SlidingWindowInferer
from models import DistributedUNet


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

    # --- ★ 変更点: ヒストグラムのX軸を統一 ---
    # 両方のデータセットから最小値と最大値を計算して、ヒストグラムの描画範囲を決定する
    hist_min = min(stats_gt.get('min', 0), stats_gen.get('min', 0))
    hist_max = max(stats_gt.get('max', 0), stats_gen.get('max', 0))
    # 安全マージンを追加
    hist_range = (hist_min - 50, hist_max + 50)

    # ヒストグラム (統一した範囲で描画)
    ax_hist_gt = fig.add_subplot(gs[0, 2]); ax_hist_gen = fig.add_subplot(gs[0, 3])
    if stats_gt: ax_hist_gt.hist(non_air_voxels_gt.flatten(), bins=100, color='deepskyblue', range=hist_range)
    ax_hist_gt.set_title("Ground Truth - HU Histogram", color='cyan'); ax_hist_gt.set_facecolor('darkgray'); ax_hist_gt.tick_params(colors='white')
    if stats_gen: ax_hist_gen.hist(non_air_voxels_gen.flatten(), bins=100, color='orchid', range=hist_range)
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
    if is_distributed and hasattr(model_for_inference, 'ema'):
        model_for_inference.ema.apply_shadow()
    elif not is_distributed and 'ema' in model_for_inference:
        model_for_inference['ema'].apply_shadow()

    
    # 3. スケジューラを選択
    if scheduler_name == "dpm_solver":
        scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000)
    elif scheduler_name == "euler":
        scheduler = EulerDiscreteScheduler(num_train_timesteps=1000)
    else: # ddpm
        scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 4. 推論の準備
    ct_full, drr1, drr2 = ct_full.to(device), drr1.to(device), drr2.to(device)

    # ★ 変更点: フルボリューム用の正しい位置エンコーディングを生成する
    # generate_and_evaluateはフルボリュームを対象とするため、ボリューム全体の座標ベクトルを生成する。
    # SlidingWindowInfererは、このフルボリューム用の座標ベクトルから、各パッチに対応する部分を切り出して使用する。
    _, _, d, h, w = ct_full.shape
    pos_d = torch.linspace(-1.0, 1.0, d)
    pos_h = torch.linspace(-1.0, 1.0, h)
    pos_w = torch.linspace(-1.0, 1.0, w)
    # (3, N) の形状でスタックし、バッチ次元を追加
    pos_3d = torch.stack([pos_d, pos_h, pos_w], dim=0).to(device)
    
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
                context = model_for_inference.conditioning_encoder(drr1, drr2) # contextは事前に計算
                model_func = lambda x, **kwargs: model_for_inference(x=x, timesteps=timesteps_tensor, context=context, **kwargs)
                model_output = inferer(inputs=image, network=model_func, pos_3d=pos_3d)
            else:
                # 単一GPUモデル用の推論関数
                context = model_for_inference['conditioning_encoder'](drr1, drr2) # contextは事前に計算
                model_func = lambda x, **kwargs: model_for_inference['unet'](x, timesteps=timesteps_tensor, context=context, **kwargs)
                model_output = inferer(inputs=image, network=model_func, pos_3d=pos_3d)
            
            image = scheduler.step(model_output, t, image).prev_sample

    # EMAモデルの重みを元に戻す
    if is_distributed and hasattr(model_for_inference, 'ema'):
        model_for_inference.ema.restore()
    elif not is_distributed and 'ema' in model_for_inference:
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
    wandb.log({"Evaluation_Report": wandb.Image(fig)}, step=wandb.run.step)
    plt.close(fig)
    print(f"  ✅ Evaluation report saved to: {save_path}")