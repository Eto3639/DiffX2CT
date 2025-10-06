# ファイル名: visualization.py

import torch
from pathlib import Path
import argparse
from tqdm import tqdm
import wandb
import os

# train.pyから必要なコンポーネントをインポートします
# このスクリプトはtrain.pyと同じディレクトリに配置してください
try:
    from train import (
        visualize_and_save_mpr,
        Preprocessed_CT_DRR_Dataset,
        CONFIG, set_seed,
        DistributedUNet,
        ConditioningEncoderResNet,
        ConditioningEncoderConvNeXt,
        ConditioningEncoderEfficientNetV2,
        DiffusionModelUNet
    )
except ImportError as e:
    print("エラー: train.pyからのインポートに失敗しました。")
    print("このスクリプト (visualization.py) が train.py と同じディレクトリにあることを確認してください。")
    print(f"詳細: {e}")
    exit()

def test_visualization(trial_number, checkpoint_dir, encoder_name, gpu_mode):
    """
    指定されたトライアルのモデルとデータを使ってvisualize_and_save_mprをテストします。
    """
    print("--- Visualization Test Script ---")

    # 1. デバイスの準備
    # ★ 変更点: 3GPUチェックを緩和し、1GPU以上あれば動作するようにする
    if gpu_mode == 'multi' and torch.cuda.device_count() < 3:
        raise RuntimeError("Multi GPU mode requires at least 3 GPUs.")
    elif not torch.cuda.is_available():
        raise RuntimeError("This script requires at least one GPU for single or multi mode.")

    device = torch.device("cuda:0")
    print(f"Using GPU mode: {gpu_mode.upper()}")
    
    # シードを設定して再現性を確保
    set_seed(CONFIG["SEED"])

    # --- WandB 初期化 ---
    run_name = f"vis-trial-{trial_number}_enc-{encoder_name}_mode-{gpu_mode}"
    wandb.init(project=CONFIG["PROJECT_NAME"], name=run_name, reinit=True)
    print(f"WandB run initialized for visualization: {run_name}")

    # 2. テスト用データの準備
    #    検証データセットからフルボリュームを1つ取得します。
    data_config = CONFIG["DATA"]
    drr_dir = Path(data_config["DRR_DIR"])
    pt_dir = Path(data_config["PT_DATA_DIR"])
    
    # train.pyと同様に、有効なデータペアを1つ見つけます
    all_pt_files = sorted(list(pt_dir.rglob("*.pt"))) # サブディレクトリも検索
    verified_file_paths = []
    for ct_path in tqdm(all_pt_files, desc="Verifying data pairs for test"):
        drr_subdir_name = ct_path.stem
        drr_ap_path = drr_dir / drr_subdir_name / "AP.pt"
        drr_lat_path = drr_dir / drr_subdir_name / "LAT.pt"
        if drr_ap_path.exists() and drr_lat_path.exists():
            verified_file_paths.append(ct_path)
            break # テストなので1つ見つかればOK

    if not verified_file_paths:
        print("エラー: テストに使用できる有効なCT/DRRデータペアが見つかりませんでした。")
        return

    # フルボリュームを取得するデータセットを作成
    vis_dataset = Preprocessed_CT_DRR_Dataset(verified_file_paths, drr_dir, patch_size=None)
    if len(vis_dataset) == 0:
        print("エラー: データセットが空です。データパスを確認してください。")
        return
        
    # 最初のデータを取得し、バッチ次元を追加します
    ct_full, drr1, drr2, pos_3d = vis_dataset[0]
    ct_full, drr1, drr2, pos_3d = ct_full.unsqueeze(0), drr1.unsqueeze(0), drr2.unsqueeze(0), pos_3d.unsqueeze(0)
    print(f"Loaded data shapes: CT={ct_full.shape}, DRR1={drr1.shape}, DRR2={drr2.shape}")

    # 3. パラメータの準備
    #    このトライアルで使われたであろうパラメータを模倣します。
    #    visualize_and_save_mprが必要とするのは主に'encoder'キーです。
    params = {
        "encoder": encoder_name,
        "patch_size": CONFIG["TRAINING"]["PATCH_SIZE"], # train.pyのSlidingWindowInfererで必要
        "learning_rate": 1e-4, # train.pyのparamsキーに合わせる
        "weight_decay": 1e-5, # 同上
        "gradient_accumulation_steps": 1, # 同上
    }
    
    # 4. GPUモードに応じてモデルを準備
    print(f"\nPreparing model for {gpu_mode.upper()} GPU mode...")
    model_for_inference = None

    if gpu_mode == 'multi':
        print("  Instantiating and loading DistributedUNet for 3 GPUs...")
        if encoder_name == 'resnet':
            conditioning_encoder = ConditioningEncoderResNet(output_dim=256)
        elif encoder_name == 'convnext':
            conditioning_encoder = ConditioningEncoderConvNeXt(output_dim=256)
        else: # efficientnet
            conditioning_encoder = ConditioningEncoderEfficientNetV2(output_dim=256)
        
        # ★ 修正: ダミーではなく、完全なU-Netモデルをインスタンス化する
        unet_full = DiffusionModelUNet(
            spatial_dims=3, in_channels=1, out_channels=1, with_conditioning=True,
            num_channels=(32, 64, 128, 256), attention_levels=(False, True, True, True),
            num_res_blocks=2, cross_attention_dim=conditioning_encoder.feature_dim
        )
        
        distributed_model = DistributedUNet(unet_full, conditioning_encoder)
        
        # 分散モデルの重みをロード
        distributed_model.conditioning_encoder.load_state_dict(torch.load(Path(checkpoint_dir) / "conditioning_encoder.pth", map_location=distributed_model.device0))
        distributed_model.time_mlp.load_state_dict(torch.load(Path(checkpoint_dir) / "unet_time_mlp.pth", map_location=distributed_model.device0))
        distributed_model.init_conv.load_state_dict(torch.load(Path(checkpoint_dir) / "unet_init_conv.pth", map_location=distributed_model.device0))
        down_blocks_state_dict = torch.load(Path(checkpoint_dir) / "unet_down_blocks.pth")
        distributed_model.down_block_0.load_state_dict({k.replace('0.', ''): v for k, v in down_blocks_state_dict.items() if k.startswith('0.')}, strict=False)
        distributed_model.down_block_1.load_state_dict({k.replace('1.', ''): v for k, v in down_blocks_state_dict.items() if k.startswith('1.')}, strict=False)
        distributed_model.down_block_2.load_state_dict({k.replace('2.', ''): v for k, v in down_blocks_state_dict.items() if k.startswith('2.')}, strict=False)
        distributed_model.down_block_3.load_state_dict({k.replace('3.', ''): v for k, v in down_blocks_state_dict.items() if k.startswith('3.')}, strict=False)
        distributed_model.mid_block1.load_state_dict(torch.load(Path(checkpoint_dir) / "unet_mid_block1.pth", map_location=distributed_model.device2))
        distributed_model.mid_attn.load_state_dict(torch.load(Path(checkpoint_dir) / "unet_mid_attn.pth", map_location=distributed_model.device2))
        distributed_model.mid_block2.load_state_dict(torch.load(Path(checkpoint_dir) / "unet_mid_block2.pth", map_location=distributed_model.device2))
        up_blocks_state_dict = torch.load(Path(checkpoint_dir) / "unet_up_blocks.pth")
        distributed_model.up_block_0.load_state_dict({k.replace('0.', ''): v for k, v in up_blocks_state_dict.items() if k.startswith('0.')}, strict=False)
        distributed_model.up_block_1.load_state_dict({k.replace('1.', ''): v for k, v in up_blocks_state_dict.items() if k.startswith('1.')}, strict=False)
        distributed_model.up_block_2.load_state_dict({k.replace('2.', ''): v for k, v in up_blocks_state_dict.items() if k.startswith('2.')}, strict=False)
        distributed_model.up_block_3.load_state_dict({k.replace('3.', ''): v for k, v in up_blocks_state_dict.items() if k.startswith('3.')}, strict=False)
        distributed_model.out_conv.load_state_dict(torch.load(Path(checkpoint_dir) / "unet_out_conv.pth", map_location=distributed_model.device0))
        distributed_model.pos_mlp_3d.load_state_dict(torch.load(Path(checkpoint_dir) / "unet_pos_mlp_3d.pth", map_location=distributed_model.device0))
        model_for_inference = distributed_model

    else: # gpu_mode == 'single'
        print(f"  Instantiating and loading models onto single device: {device}...")
        if encoder_name == 'resnet':
            conditioning_encoder = ConditioningEncoderResNet(output_dim=256)
        elif encoder_name == 'convnext':
            conditioning_encoder = ConditioningEncoderConvNeXt(output_dim=256)
        else: # efficientnet
            conditioning_encoder = ConditioningEncoderEfficientNetV2(output_dim=256)
        
        unet = DiffusionModelUNet(
            spatial_dims=3, in_channels=1, out_channels=1, with_conditioning=True,
            num_channels=(32, 64, 128, 256), attention_levels=(False, True, True, True),
            num_res_blocks=2, cross_attention_dim=conditioning_encoder.feature_dim
        )
        
        # 単一デバイスに全パーツをロード
        conditioning_encoder.load_state_dict(torch.load(Path(checkpoint_dir) / "conditioning_encoder.pth", map_location=device))
        unet.time_mlp.load_state_dict(torch.load(Path(checkpoint_dir) / "unet_time_mlp.pth", map_location=device))
        unet.init_conv.load_state_dict(torch.load(Path(checkpoint_dir) / "unet_init_conv.pth", map_location=device))
        unet.down_blocks.load_state_dict(torch.load(Path(checkpoint_dir) / "unet_down_blocks.pth", map_location=device))
        unet.mid_block1.load_state_dict(torch.load(Path(checkpoint_dir) / "unet_mid_block1.pth", map_location=device))
        unet.mid_attn.load_state_dict(torch.load(Path(checkpoint_dir) / "unet_mid_attn.pth", map_location=device))
        unet.mid_block2.load_state_dict(torch.load(Path(checkpoint_dir) / "unet_mid_block2.pth", map_location=device))
        unet.up_blocks.load_state_dict(torch.load(Path(checkpoint_dir) / "unet_up_blocks.pth", map_location=device))
        unet.pos_mlp_3d.load_state_dict(torch.load(Path(checkpoint_dir) / "unet_pos_mlp_3d.pth", map_location=device))
        
        # ラッパーオブジェクトに格納
        model_for_inference = {'unet': unet.to(device), 'conditioning_encoder': conditioning_encoder.to(device)}

    # 4. visualize_and_save_mpr 関数の呼び出し
    print(f"\nCalling visualize_and_save_mpr for trial {trial_number}...")    
    try:
        visualize_and_save_mpr(
            device=device,
            params=params,
            scheduler_name="dpm_solver", # テストしたいスケジューラ名 (dpm_solver, euler, ddpm)
            ct_full=ct_full,
            drr1=drr1,
            drr2=drr2,
            pos_3d=pos_3d,
            best_epoch=999, # テストなのでダミーの値でOK
            trial_number=trial_number,
            save_dir=checkpoint_dir,
            model_for_inference=model_for_inference
        )
        print("\n✅ Visualization function finished successfully.")
        print(f"   -> Check the output in: {checkpoint_dir}/visualizations/")
    except Exception as e:
        print(f"\n❌ An error occurred during visualization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run visualization using a trained model checkpoint.")
    parser.add_argument("--trial_number", type=int, required=True, 
                        help="The trial number whose checkpoint you want to test.")
    parser.add_argument("--encoder", type=str, required=True, choices=["resnet", "convnext", "efficientnet"], 
                        help="The encoder model used for that trial.")
    parser.add_argument("--gpu_mode", type=str, default="multi", choices=["single", "multi"],
                        help="GPU mode for inference: 'single' for one GPU, 'multi' for model parallelism across 3 GPUs.")
    args = parser.parse_args()

    # チェックポイントディレクトリのパスを構築 (train.pyの保存パスと一致させる)
    checkpoint_dir = f"./checkpoints/trial_{args.trial_number}"
    
    if not Path(checkpoint_dir).exists():
        print(f"エラー: チェックポイントディレクトリが見つかりません: {checkpoint_dir}")
        print("正しい trial_number を指定したか確認してください。")
    else:
        test_visualization(args.trial_number, checkpoint_dir, args.encoder, args.gpu_mode)
