import os
import sys
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.transform import resize
from tqdm import tqdm
import warnings
import thop
warnings.filterwarnings('ignore')

torch.serialization.add_safe_globals([np._core.multiarray.scalar])

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Dataset import CTDataset
from Model import MultiTaskSwinUNet

class TestConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = 192
    depths = [1, 1, 1, 1]
    num_heads = [2, 4, 8, 16]
    window_size = 8
    pos_dim = 128
    position = 0.5
    base_checkpoint_dir = r"checkpoints"
    base_result_dir = r"Heart_results"
    images_dir = r"E:\Interpolation\Article\sample_data\Task02_Heart\imagesTr"
    labels_dir = r"E:\Interpolation\Article\sample_data\Task02_Heart\labelsTr"
    
    vis_num_samples_per_fold = 200
    seg_threshold = 0.5
    overlay_alpha = 0.5
    
    gt_colors = {
        "class1": np.array([0.0, 1.0, 0.0])
    }
    
    fixed_crop_size = 64
    inset_size = 128
    
    batch_size = 2
    random_seed = 42
    n_folds = 5
    
    enable_efficiency_test = True
    warmup_steps = 5
    repeat_times = 3

os.makedirs(TestConfig.base_result_dir, exist_ok=True)

def get_fixed_square_bbox(mask, crop_size):
    coords = np.argwhere(mask > 0)
    h, w = mask.shape[:2]
    
    if len(coords) == 0:
        cy, cx = h//2, w//2
    else:
        cy = int(np.mean(coords[:, 0]))
        cx = int(np.mean(coords[:, 1]))
    
    half = crop_size // 2
    y_min = max(0, cy - half)
    y_max = min(h, cy + half)
    x_min = max(0, cx - half)
    x_max = min(w, cx + half)
    
    if y_max - y_min< crop_size:
        y_min = max(0, y_max - crop_size)
    if x_max - x_min < crop_size:
        x_min = max(0, x_max - crop_size)
        
    return (y_min, y_max, x_min, x_max)

def crop_and_zoom(img, bbox, target_size):
    y_min, y_max, x_min, x_max = bbox
    cropped = img[y_min:y_max, x_min:x_max]
    
    zoomed = resize(
        cropped, 
        (target_size, target_size, img.shape[-1]) if len(img.shape)==3 else (target_size, target_size),
        order=0,          
        preserve_range=True,
        anti_aliasing=False
    )
    return zoomed

def add_segmentation_zoom(orig_img, seg_mask):
    img = orig_img.copy()
    h, w = img.shape[:2]
    inset_size = TestConfig.inset_size
    
    bbox = get_fixed_square_bbox(seg_mask, TestConfig.fixed_crop_size)
    zoomed_img = crop_and_zoom(img, bbox, inset_size)
    
    y_start = h - inset_size - 5
    x_start = 5
    y_end = y_start + inset_size
    x_end = x_start + inset_size
    
    img[y_start-2:y_end+2, x_start-2:x_end+2] = 1.0
    img[y_start:y_end, x_start:x_end] = zoomed_img
    return img

class MetricCalculator:
    @staticmethod
    def calculate_interp_metrics(pred, target):
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        metrics = {'MAE': [], 'MSE': [], 'SSIM': [], 'PSNR': []}
        for i in range(pred_np.shape[0]):
            pred_img = pred_np[i, 0, :, :]
            target_img = target_np[i, 0, :, :]
            mae = np.mean(np.abs(pred_img - target_img))
            mse = np.mean((pred_img - target_img) ** 2)
            ssim_score = ssim(pred_img, target_img, data_range=1.0, win_size=11, channel_axis=None)
            psnr_score = psnr(pred_img, target_img, data_range=1.0)
            metrics['MAE'].append(mae)
            metrics['MSE'].append(mse)
            metrics['SSIM'].append(ssim_score)
            metrics['PSNR'].append(psnr_score)
        for key in metrics:
            metrics[key] = np.mean(metrics[key])
        return metrics
    
    @staticmethod
    def calculate_seg_metrics(pred, target, threshold=0.5):
        pred_np = (pred.detach().cpu().numpy() > threshold).astype(np.float32)
        target_np = (target.detach().cpu().numpy() > threshold).astype(np.float32)
        metrics = {'Dice': [], 'IoU': [], 'Precision': [], 'Recall': []}
        for i in range(pred_np.shape[0]):
            for j in range(3):
                pred_seg = pred_np[i, j, :, :]
                target_seg = target_np[i, j, :, :]
                tp = np.sum(pred_seg * target_seg)
                fp = np.sum(pred_seg * (1 - target_seg))
                fn = np.sum((1 - pred_seg) * target_seg)
                
                dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
                iou = tp / (tp + fp + fn + 1e-8)
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                
                metrics['Dice'].append(dice)
                metrics['IoU'].append(iou)
                metrics['Precision'].append(precision)
                metrics['Recall'].append(recall)
        for key in metrics:
            metrics[key] = np.mean(metrics[key]) if len(metrics[key])>0 else 0.0
        return metrics

class EfficiencyCalculator:
    @staticmethod
    def record_infer_time(model, inputs, repeat_times=3):
        for _ in range(TestConfig.warmup_steps):
            with torch.no_grad():
                model(inputs)
        
        infer_times = []
        with torch.no_grad():
            for _ in range(repeat_times):
                if TestConfig.device.type == 'cuda':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    model(inputs)
                    end_event.record()
                    torch.cuda.synchronize()
                    infer_time_ms = start_event.elapsed_time(end_event)
                else:
                    start_time = time.time()
                    model(inputs)
                    end_time = time.time()
                    infer_time_ms = (end_time - start_time) * 1000
                infer_times.append(infer_time_ms)
        
        avg_infer_time_ms = np.mean(infer_times)
        batch_size = inputs.shape[0]
        fps = batch_size / (avg_infer_time_ms / 1000)
        return avg_infer_time_ms, fps

def overlay_seg_gt_on_img(orig_img, seg_gt_raw, alpha=0.5):
    rgb_img = np.stack([orig_img, orig_img, orig_img], axis=-1)
    mask1 = seg_gt_raw == 1
    if np.any(mask1):
        rgb_img[mask1] = (1 - alpha) * rgb_img[mask1] + alpha * TestConfig.gt_colors['class1']
    return rgb_img

def overlay_raw_seg_pred(orig_img, seg_pred, gt_class1, alpha=0.5):
    rgb_img = np.stack([orig_img, orig_img, orig_img], axis=-1)
    pred_bin = seg_pred > 0.5
    gt_bin = gt_class1 > 0.5
    
    tp_mask = pred_bin & gt_bin
    error_mask = (pred_bin & ~gt_bin) | (~pred_bin & gt_bin)
    
    correct_color = np.array([0.0, 1.0, 0.0])
    error_color = np.array([1.0, 0.0, 0.0])
    
    if np.any(tp_mask):
        rgb_img[tp_mask] = (1-alpha)*rgb_img[tp_mask] + alpha * correct_color
    if np.any(error_mask):
        rgb_img[error_mask] = (1-alpha)*rgb_img[error_mask] + alpha * error_color
    return rgb_img

def visualize_results(sample_idx, position,
                     img1, img2, interp_gt, interp_pred,
                     seg_gt, seg_pred, save_dir):
    sample_folder = os.path.join(save_dir, f"sample_{sample_idx:04d}")
    os.makedirs(sample_folder, exist_ok=True)
    
    seg_gt_raw = seg_gt.astype(np.float32)
    seg_pred_bin = (seg_pred > TestConfig.seg_threshold).astype(np.float32)
    
    img_size = img1.shape[0]
    dpi = 300
    figsize = (img_size / dpi, img_size / dpi)
    prefix = f"sample{sample_idx:04d}"
    
    subplot_configs = [
        (f"{prefix}_interp_gt.png", interp_gt, seg_gt_raw[2]),
        (f"{prefix}_interp_pred.png", interp_pred, seg_pred_bin[2]),
    ]
    
    frame_names = ["frame1", "frame2", "interp"]
    orig_imgs = [img1, img2, interp_gt]
    
    for i in range(3):
        frame_name = frame_names[i]
        orig_img = orig_imgs[i]
        gt_frame_raw = seg_gt_raw[i]
        pred_frame = seg_pred_bin[i]
        gt1 = (gt_frame_raw == 1).astype(np.float32)
        
        gt_img = overlay_seg_gt_on_img(orig_img, gt_frame_raw, TestConfig.overlay_alpha)
        subplot_configs.append((f"{prefix}_{frame_name}_seg_gt.png", gt_img, gt_frame_raw))
        
        pred_img = overlay_raw_seg_pred(orig_img, pred_frame, gt1, TestConfig.overlay_alpha)
        subplot_configs.append((f"{prefix}_{frame_name}_seg_pred.png", pred_img, pred_frame))
    
    for filename, img_data, mask in subplot_configs:
        img_rot = img_data
        mask_rot = mask
        img_final = add_segmentation_zoom(img_rot, mask_rot)
        
        fig, ax = plt.subplots(1,1,figsize=figsize,dpi=dpi)
        plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
        fig.patch.set_facecolor('black')
        ax.imshow(img_final, cmap='gray' if len(img_final.shape)==2 else None, vmin=0,vmax=1)
        ax.axis('off')
        ax.margins(0,0)
        ax.spines[:].set_visible(False)
        save_path = os.path.join(sample_folder, filename)
        plt.savefig(save_path,dpi=dpi,bbox_inches=None,pad_inches=0,facecolor='black',edgecolor="none")
        plt.close(fig)

    def process_for_grid(img, mask=None):
        img_rot = img
        if len(img_rot.shape) == 2:
            img_rot = np.stack([img_rot, img_rot, img_rot], axis=-1)
        if mask is not None:
            mask_rot = mask
            img_rot = add_segmentation_zoom(img_rot, mask_rot)
        return img_rot

    alpha = TestConfig.overlay_alpha
    f1_gt, f2_gt, interp_gt_seg = seg_gt_raw[0], seg_gt_raw[1], seg_gt_raw[2]
    f1_pred, f2_pred, interp_pred_seg = seg_pred_bin[0], seg_pred_bin[1], seg_pred_bin[2]
    
    gt1_f1 = (f1_gt == 1).astype(np.float32)
    gt1_interp = (interp_gt_seg == 1).astype(np.float32)
    gt1_f2 = (f2_gt == 1).astype(np.float32)

    seg_row1 = [
        process_for_grid(overlay_seg_gt_on_img(img1, f1_gt, alpha), f1_gt),
        process_for_grid(overlay_seg_gt_on_img(interp_gt, interp_gt_seg, alpha), interp_gt_seg),
        process_for_grid(overlay_seg_gt_on_img(img2, f2_gt, alpha), f2_gt)
    ]
    seg_row2 = [
        process_for_grid(overlay_raw_seg_pred(img1, f1_pred, gt1_f1, alpha), f1_pred),
        process_for_grid(overlay_raw_seg_pred(interp_gt, interp_pred_seg, gt1_interp, alpha), interp_pred_seg),
        process_for_grid(overlay_raw_seg_pred(img2, f2_pred, gt1_f2, alpha), f2_pred)
    ]
    seg_grid = [seg_row1, seg_row2]
    seg_col_names = ["Input Frame1", "Interp Frame", "Input Frame2"]

    fig1, axes1 = plt.subplots(2, 3, dpi=dpi, figsize=(6, 4))
    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0, wspace=0, hspace=0)
    fig1.patch.set_facecolor("black")
    for row_idx in range(2):
        for col_idx in range(3):
            ax = axes1[row_idx][col_idx]
            img = seg_grid[row_idx][col_idx]
            ax.imshow(img, vmin=0, vmax=1)
            ax.axis("off")
            ax.margins(0,0)
            ax.spines[:].set_visible(False)
            if row_idx == 0:
                ax.set_title(seg_col_names[col_idx], color="white", fontsize=8, pad=1)
    seg_grid_path = os.path.join(sample_folder, "grid_3x2_seg_visualization.png")
    plt.savefig(seg_grid_path, dpi=dpi, bbox_inches="tight", pad_inches=0, 
                facecolor="black", edgecolor="none")
    plt.close(fig1)

    raw_grid = [
        process_for_grid(img1),                          
        process_for_grid(interp_gt, interp_gt_seg),      
        process_for_grid(interp_pred, interp_pred_seg),  
        process_for_grid(img2)                           
    ]
    raw_col_names = ["Input Frame1", "Interp GT", f"Interp Pred", "Input Frame2"]

    fig2, axes2 = plt.subplots(1, 4, dpi=dpi, figsize=(8, 2))
    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0, wspace=0, hspace=0)
    fig2.patch.set_facecolor("black")
    for col_idx in range(4):
        ax = axes2[col_idx]
        img = raw_grid[col_idx]
        ax.imshow(img, vmin=0, vmax=1)
        ax.axis("off")
        ax.margins(0,0)
        ax.spines[:].set_visible(False)
        ax.set_title(raw_col_names[col_idx], color="white", fontsize=8, pad=1)
    raw_grid_path = os.path.join(sample_folder, "grid_4x1_raw_visualization.png")
    plt.savefig(raw_grid_path, dpi=dpi, bbox_inches="tight", pad_inches=0, 
                facecolor="black", edgecolor="none")
    plt.close(fig2)

def load_model(fold_idx):
    checkpoint_path = os.path.join(
        TestConfig.base_checkpoint_dir, 
        f"fold_{fold_idx}", 
        "best_model.pth"
    )
    model = MultiTaskSwinUNet().to(TestConfig.device)
    
    checkpoint = torch.load(checkpoint_path, map_location=TestConfig.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Loaded Fold {fold_idx} Best Model: {checkpoint_path}")
    return model, checkpoint_path

def test_single_fold(fold_idx):
    fold_result_dir = os.path.join(TestConfig.base_result_dir, f"fold_{fold_idx}")
    vis_dir = os.path.join(fold_result_dir, "visualizations")
    csv_path = os.path.join(fold_result_dir, "evaluation_metrics.csv")
    efficiency_csv = os.path.join(fold_result_dir, "efficiency_metrics.csv")
    summary_csv = os.path.join(fold_result_dir, "fold_summary.csv")
    
    os.makedirs(fold_result_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    metric_calc = MetricCalculator()
    efficiency_calc = EfficiencyCalculator()
    model, checkpoint_path = load_model(fold_idx)

    test_dataset = CTDataset(
        images_dir=TestConfig.images_dir,
        labels_dir=TestConfig.labels_dir,
        position=TestConfig.position,
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=TestConfig.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    all_metrics = []
    all_efficiency_metrics = []
    total_interp_metrics = {'MAE':[],'MSE':[],'SSIM':[],'PSNR':[]}
    total_seg_metrics = {'Dice':[],'IoU':[],'Precision':[],'Recall':[]}
    total_efficiency = {'infer_time_ms': [], 'fps': []}
    vis_count = 0
    pbar = tqdm(test_loader, desc=f"Testing Fold {fold_idx}")

    params = sum(p.numel() for p in model.parameters())
    params_m = params / 1e6
    model_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    dummy_input = torch.randn(1, 2, 512, 512).to(TestConfig.device)
    flops, _ = thop.profile(model, inputs=(dummy_input,), verbose=False)
    flops_g = flops / 1e9

    if TestConfig.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
        memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        memory_mb = 0.0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(pbar):
            inputs = batch_data[0].to(TestConfig.device)
            interp_gt = batch_data[1].to(TestConfig.device)
            input_seg_gt = batch_data[2].to(TestConfig.device)
            interp_seg_gt = batch_data[3].to(TestConfig.device)
            
            seg_gt = torch.cat([input_seg_gt, interp_seg_gt], dim=1)
            
            avg_infer_time_ms = 0.0
            fps = 0.0
            if TestConfig.enable_efficiency_test:
                avg_infer_time_ms, fps = efficiency_calc.record_infer_time(
                    model, inputs, repeat_times=TestConfig.repeat_times
                )
                total_efficiency['infer_time_ms'].append(avg_infer_time_ms)
                total_efficiency['fps'].append(fps)
            
            interp_pred, seg_pred = model(inputs)
            
            interp_metrics = metric_calc.calculate_interp_metrics(interp_pred, interp_gt)
            seg_metrics = metric_calc.calculate_seg_metrics(seg_pred, seg_gt, TestConfig.seg_threshold)
            
            for k in total_interp_metrics:
                total_interp_metrics[k].append(interp_metrics[k])
            for k in total_seg_metrics:
                total_seg_metrics[k].append(seg_metrics[k])
            
            batch_metrics = {
                'batch_idx': batch_idx,
                'interp_MAE': interp_metrics['MAE'],
                'interp_MSE': interp_metrics['MSE'],
                'interp_SSIM': interp_metrics['SSIM'],
                'interp_PSNR': interp_metrics['PSNR'],
                'seg_Dice': seg_metrics['Dice'],
                'seg_IoU': seg_metrics['IoU'],
                'seg_Precision': seg_metrics['Precision'],
                'seg_Recall': seg_metrics['Recall'],
                'infer_time_ms': avg_infer_time_ms,
                'fps': fps,
            }
            all_metrics.append(batch_metrics)
            
            pbar.set_postfix({
                'SSIM':f"{interp_metrics['SSIM']:.4f}",
                'Dice':f"{seg_metrics['Dice']:.4f}",
            })
            
            if vis_count < TestConfig.vis_num_samples_per_fold:
                inputs_np = inputs.cpu().numpy()
                interp_gt_np = interp_gt.cpu().numpy()
                interp_pred_np = interp_pred.cpu().numpy()
                seg_gt_np = seg_gt.cpu().numpy()
                seg_pred_np = seg_pred.cpu().numpy()
                
                for s in range(min(TestConfig.batch_size, TestConfig.vis_num_samples_per_fold - vis_count)):
                    visualize_results(
                        sample_idx=batch_idx*TestConfig.batch_size+s,
                        position=TestConfig.position,
                        img1=inputs_np[s,0],
                        img2=inputs_np[s,1],
                        interp_gt=interp_gt_np[s,0],
                        interp_pred=interp_pred_np[s,0],
                        seg_gt=seg_gt_np[s],
                        seg_pred=seg_pred_np[s],
                        save_dir=vis_dir
                    )
                    vis_count +=1
    
    overall_interp = {f'interp_{k}': np.mean(v) for k, v in total_interp_metrics.items()}
    overall_seg = {f'seg_{k}': np.mean(v) for k, v in total_seg_metrics.items()}
    overall_efficiency = {}
    if TestConfig.enable_efficiency_test:
        overall_efficiency = {
            'avg_infer_time_ms': np.mean(total_efficiency['infer_time_ms']),
            'avg_fps': np.mean(total_efficiency['fps']),
        }
    
    overall = {**overall_interp, **overall_seg, **overall_efficiency}
    
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['MAE','MSE','SSIM','PSNR','Dice','IoU','Precision','Recall'])
        writer.writerow([
            overall['interp_MAE'], overall['interp_MSE'], overall['interp_SSIM'], overall['interp_PSNR'],
            overall['seg_Dice'], overall['seg_IoU'], overall['seg_Precision'], overall['seg_Recall']
        ])
    
    print(f"\n🎉 Fold {fold_idx} Test Completed!")
    print(f"SSIM: {overall['interp_SSIM']:.4f} | Dice: {overall['seg_Dice']:.4f}")
    return overall

def test_all_folds():
    torch.manual_seed(TestConfig.random_seed)
    np.random.seed(TestConfig.random_seed)
    
    print("=== Starting 5-Fold Cross Validation Test ===")
    fold_metrics_list = []
    
    for fold_idx in range(1, TestConfig.n_folds + 1):
        fold_metrics = test_single_fold(fold_idx)
        fold_metrics_list.append(fold_metrics)
    
    metric_keys = ['interp_MAE', 'interp_MSE', 'interp_SSIM', 'interp_PSNR',
                   'seg_Dice', 'seg_IoU', 'seg_Precision', 'seg_Recall']
    avg_metrics = {key: np.mean([m[key] for m in fold_metrics_list]) for key in metric_keys}
    
    total_summary_csv = os.path.join(TestConfig.base_result_dir, "5fold_average_summary.csv")
    with open(total_summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Fold'] + metric_keys)
        for i, m in enumerate(fold_metrics_list):
            writer.writerow([f"Fold {i+1}"] + [m[key] for key in metric_keys])
        writer.writerow(['Average'] + [avg_metrics[key] for key in metric_keys])
    
    print("\n" + "="*60)
    print("🎉 5-Fold Cross Validation Test Final Results")
    print("="*60)
    print(f"Average SSIM: {avg_metrics['interp_SSIM']:.4f}")
    print(f"Average Dice: {avg_metrics['seg_Dice']:.4f}")
    print(f"Average PSNR: {avg_metrics['interp_PSNR']:.4f}")
    print(f"Average IoU:  {avg_metrics['seg_IoU']:.4f}")
    print(f"\nFull average metrics saved to: {total_summary_csv}")

def main():
    test_all_folds()

if __name__ == "__main__":
    main()