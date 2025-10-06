# ファイル名: evaluate_generated_ct.py

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

# train.pyからデータセットと設定をインポート
try:
    from train import Preprocessed_CT_DRR_Dataset, CONFIG
except ImportError as e:
    print("エラー: DiffX2CT/train.pyからのインポートに失敗しました。")
    print("このスクリプトを 'AMED' ディレクトリから実行していることを確認してください。")
    print(f"詳細: {e}")
    exit()

def calculate_mae(image_true, image_test):
    """平均絶対誤差 (MAE) を計算する"""
    return np.mean(np.abs(image_true - image_test))

def evaluate_and_visualize(generated_npy_path: Path, trial_number: int, output_dir: Path):
    """
    生成されたCTボリュームを評価し、結果を可視化して保存する。
    """
    print(f"--- 評価開始: {generated_npy_path.name} ---")
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. 生成されたCTデータをロード (HU値)
    generated_hu_np = np.load(generated_npy_path)
    print(f"✅ 生成CTデータをロードしました。形状: {generated_hu_np.shape}")
    
    # 2. 対応するGround Truthデータをロード
    print("⏳ 対応するGround Truthデータを検索・ロード中...")
    data_config = CONFIG["DATA"]
    drr_dir = Path(data_config["DRR_DIR"])
    pt_dir = Path(data_config["PT_DATA_DIR"])

    all_pt_files = sorted(list(pt_dir.rglob("*.pt")))
    verified_file_paths = []
    for ct_path in tqdm(all_pt_files, desc="Verifying data pairs for test", leave=False):
        drr_subdir_name = ct_path.stem
        drr_ap_path = drr_dir / drr_subdir_name / "AP.pt"
        drr_lat_path = drr_dir / drr_subdir_name / "LAT.pt"
        if drr_ap_path.exists() and drr_lat_path.exists():
            verified_file_paths.append(ct_path)
            break # visualization.pyと同様に最初の1つを使用

    if not verified_file_paths:
        print("❌ エラー: Ground Truthデータが見つかりませんでした。")
        return

    vis_dataset = Preprocessed_CT_DRR_Dataset(verified_file_paths, drr_dir, patch_size=None)
    ct_full, _, _, _ = vis_dataset[0]
    
    # [0, 1] から HU値に逆正規化
    min_hu, max_hu = -1024, 1500
    ground_truth_hu_np = ct_full.squeeze().cpu().numpy() * (max_hu - min_hu) + min_hu
    print(f"✅ Ground Truthデータをロードし、HU値に変換しました。形状: {ground_truth_hu_np.shape}")

    # 3. 品質評価指標を計算
    data_range = max_hu - min_hu # 評価範囲を固定
    ssim_score = ssim(ground_truth_hu_np, generated_hu_np, data_range=data_range, channel_axis=None)
    psnr_score = psnr(ground_truth_hu_np, generated_hu_np, data_range=data_range)
    mae_score = calculate_mae(ground_truth_hu_np, generated_hu_np)
    print(f"📊 品質評価指標: SSIM={ssim_score:.4f}, PSNR={psnr_score:.2f} dB, MAE={mae_score:.2f} HU")

    # 4. 統計情報を計算 (空気以外)
    non_air_voxels = generated_hu_np[generated_hu_np > -1000]
    stats = {
        'mean': np.mean(non_air_voxels),
        'std': np.std(non_air_voxels),
        'min': np.min(non_air_voxels),
        'max': np.max(non_air_voxels)
    }

    # 5. 可視化
    fig = plt.figure(figsize=(20, 14), facecolor='black')
    gs = plt.GridSpec(3, 4, figure=fig)
    
    # --- MPR表示 ---
    z, y, x = generated_hu_np.shape
    slice_ax, slice_cor, slice_sag = z // 2, y // 2, x // 2
    vmin, vmax = -1024, 300 # 表示ウィンドウ

    views = {
        'Axial': (ground_truth_hu_np[slice_ax, :, :], generated_hu_np[slice_ax, :, :]),
        'Coronal': (np.flipud(ground_truth_hu_np[:, slice_cor, :]), np.flipud(generated_hu_np[:, slice_cor, :])),
        'Sagittal': (np.fliplr(np.flipud(ground_truth_hu_np[:, :, slice_sag])), np.fliplr(np.flipud(generated_hu_np[:, :, slice_sag])))
    }

    for i, (title, (gt_img, gen_img)) in enumerate(views.items()):
        ax_gt = fig.add_subplot(gs[i, 0])
        ax_gt.imshow(gt_img, cmap='gray', vmin=vmin, vmax=vmax)
        ax_gt.set_title(f'Ground Truth {title}', color='cyan')
        ax_gt.axis('off')

        ax_gen = fig.add_subplot(gs[i, 1])
        ax_gen.imshow(gen_img, cmap='gray', vmin=vmin, vmax=vmax)
        ax_gen.set_title(f'Generated {title}', color='magenta')
        ax_gen.axis('off')

    # --- ヒストグラム ---
    ax_hist = fig.add_subplot(gs[0, 2:])
    ax_hist.hist(non_air_voxels.flatten(), bins=100, color='skyblue', range=(-1000, 1000))
    ax_hist.set_title("Generated CT - HU Histogram (air excluded)", color='white')
    ax_hist.set_facecolor('darkgray')
    ax_hist.tick_params(colors='white')

    # --- 統計情報とスコア ---
    ax_text = fig.add_subplot(gs[1:, 2:])
    ax_text.axis('off')
    report_text = (
        f"--- Quality Metrics ---\n"
        f"  SSIM: {ssim_score:.4f}\n"
        f"  PSNR: {psnr_score:.2f} dB\n"
        f"  MAE:  {mae_score:.2f} HU\n\n"
        f"--- Statistics (Generated CT) ---\n"
        f"  Mean: {stats['mean']:.2f} HU\n"
        f"  Std Dev: {stats['std']:.2f} HU\n"
        f"  Min: {stats['min']:.0f} HU\n"
        f"  Max: {stats['max']:.0f} HU\n"
    )
    ax_text.text(0.05, 0.7, report_text, color='white', fontfamily='monospace', fontsize=14, va='top')

    fig.suptitle(f'Evaluation Report for Trial {trial_number}', color='white', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = output_dir / f"evaluation_report_trial_{trial_number}.png"
    plt.savefig(save_path, facecolor='black')
    plt.close(fig)
    print(f"✅ 評価レポートを保存しました: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a generated CT volume (.npy) and create a visual report.")
    parser.add_argument(
        "--generated_npy", 
        type=str, 
        required=True,
        help="Path to the generated CT numpy file (e.g., checkpoints/trial_0/visualizations/generated_ct_trial_0_epoch_999.npy)"
    )
    parser.add_argument(
        "--trial_number", 
        type=int, 
        required=True,
        help="The trial number corresponding to the generated CT, used to find the ground truth."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./evaluation_reports",
        help="Directory to save the output evaluation report PNG."
    )
    args = parser.parse_args()

    evaluate_and_visualize(Path(args.generated_npy), args.trial_number, Path(args.output_dir))
