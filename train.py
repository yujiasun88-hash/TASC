import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from sklearn.model_selection import KFold
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cosine
import matplotlib.colors as mcolors
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Dataset import CTDataset
from Model import MultiTaskSwinUNet
import torch.nn.functional as F

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.4):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.alpha = alpha
    def forward(self, pred, target):
        return self.alpha*self.l1(pred, target) + (1-self.alpha)*self.mse(pred, target)

class DynamicMultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.interpolation_loss = HybridLoss(alpha=0.4)
        self.segmentation_loss = nn.BCELoss()
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, interp_pred, interp_target, seg_pred, seg_target):
        interp_loss = self.interpolation_loss(interp_pred, interp_target)
        seg_loss = self.segmentation_loss(seg_pred, seg_target)
        
        w1 = torch.exp(-self.log_vars[0])
        w2 = torch.exp(-self.log_vars[1])
        
        total_loss = w1 * interp_loss + w2 * seg_loss + torch.sum(self.log_vars)
        
        return total_loss, interp_loss, seg_loss

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    epochs = 2
    lr = 1e-4
    weight_decay = 1e-5
    patience = 10
    embed_dim = 192
    depths = [1, 1, 1, 1]
    num_heads = [2, 4, 8, 16]
    window_size = 8
    pos_dim = 128
    position = 0.5
    checkpoint_dir = r"checkpoints"
    plot_path = r"plot/train_loss_curve.png"
    analysis_dir = r"analysis"
    log_interval = 10
    n_folds = 5
    random_seed = 42
    analysis_layers = ['layer4']
    grad_clip_norm = 1.0

os.makedirs(Config.checkpoint_dir, exist_ok=True)
os.makedirs(os.path.dirname(Config.plot_path), exist_ok=True)
os.makedirs(Config.analysis_dir, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

def load_full_dataset():
    print("="*50)
    print("Loading Full Dataset...")
    
    full_dataset = CTDataset(
        position=Config.position,
        split='train'
    )
    
    print(f"Total Dataset Samples: {len(full_dataset)}")
    return full_dataset

def get_fold_dataloaders(full_dataset, fold_idx, kf):
    train_indices, val_indices = list(kf.split(full_dataset))[fold_idx]
    
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    print(f"\nFold {fold_idx+1} - Train Samples: {len(train_subset)}, Val Samples: {len(val_subset)}")
    
    train_loader = DataLoader(
        train_subset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if Config.device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if Config.device.type == "cuda" else False
    )
    
    return train_loader, val_loader

def init_model():
    print("="*50)
    print("Initializing Multi-Task Model...")
    
    model = MultiTaskSwinUNet().to(Config.device)
    
    criterion = DynamicMultiTaskLoss().to(Config.device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.lr,
        weight_decay=Config.weight_decay
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=1e-6)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Params: {num_params / 1e6:.2f}M")
    print(f"Device: {Config.device}")
    print("Loss: Paper Dynamic Multi-Task Loss (learnable weights)")
    
    return model, criterion, optimizer, scheduler

def calculate_ssim(pred, target):
    ssim_scores = []
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    for i in range(pred_np.shape[0]):
        pred_img = pred_np[i, 0, :, :]
        target_img = target_np[i, 0, :, :]
        
        score = ssim(
            pred_img, target_img,
            data_range=1.0, win_size=11, channel_axis=None
        )
        ssim_scores.append(score)
    
    return np.mean(ssim_scores)

def calculate_dice(pred, target, threshold=0.5):
    dice_scores = []
    pred_np = (pred.detach().cpu().numpy() > threshold).astype(np.float32)
    target_np = (target.detach().cpu().numpy() > threshold).astype(np.float32)
    
    for i in range(pred_np.shape[0]):
        batch_dice = []
        for j in range(3):
            pred_seg = pred_np[i, j, :, :]
            target_seg = target_np[i, j, :, :]
            
            intersection = np.sum(pred_seg * target_seg)
            union = np.sum(pred_seg) + np.sum(target_seg)
            
            dice = 1.0 if union == 0 else (2.0 * intersection) / (union + 1e-8)
            batch_dice.append(dice)
        
        dice_scores.append(np.mean(batch_dice))
    
    return np.mean(dice_scores)

class TaskAnalyzer:
    def __init__(self, model):
        self.model = model
        self.feature_hooks = []
        self.attention_maps = defaultdict(list)
        self.features = defaultdict(list)
        
    def register_hooks(self):
        def feature_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.features['last_layer'].append(output.detach().cpu())
        
        matched = False
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.LayerNorm)):
                hook = module.register_forward_hook(feature_hook)
                self.feature_hooks.append(hook)
                matched = True
                break
        if not matched:
            print("⚠️ No feature layer matched, using fallback mode")

    def remove_hooks(self):
        for hook in self.feature_hooks:
            hook.remove()
        self.feature_hooks.clear()

    @staticmethod
    def compute_gradient_correlation(model, interp_loss, seg_loss):
        valid_params = [p for p in model.parameters() if p.requires_grad]
        
        model.zero_grad()
        interp_loss.backward(retain_graph=True)
        interp_grads = []
        for param in valid_params:
            if param.grad is not None:
                interp_grads.append(param.grad.detach().clone())
            else:
                interp_grads.append(torch.zeros_like(param))
        
        model.zero_grad()
        seg_loss.backward(retain_graph=True)
        seg_grads = []
        for param in valid_params:
            if param.grad is not None:
                seg_grads.append(param.grad.detach().clone())
            else:
                seg_grads.append(torch.zeros_like(param))
        
        model.zero_grad()

        correlations = []
        eps = 1e-8
        for g1, g2 in zip(interp_grads, seg_grads):
            g1_flat = g1.view(-1).float()
            g2_flat = g2.view(-1).float()
            
            norm1 = torch.norm(g1_flat)
            norm2 = torch.norm(g2_flat)
            if norm1 < eps or norm2 < eps:
                correlations.append(0.0)
                continue
            
            cos_sim = torch.sum(g1_flat * g2_flat) / (norm1 * norm2 + eps)
            correlations.append(cos_sim.item())
        
        correlation = float(np.mean(correlations))
        return round(correlation, 4)

    @staticmethod
    def compute_feature_contribution(features, interp_pred, seg_pred):
        if len(features) == 0:
            return 0.0, 0.0
        feat = features[-1].mean(dim=[0,2,3]).cpu().numpy()
        interp_contrib = np.mean(np.abs(feat * interp_pred.mean().item()))
        seg_contrib = np.mean(np.abs(feat * seg_pred.mean().item()))
        return round(interp_contrib, 4), round(seg_contrib, 4)

    @staticmethod
    def compute_attention_stats(attention_maps):
        if len(attention_maps) == 0 or len(attention_maps['last_layer']) == 0:
            return 0.0, 0.0
        attn = torch.cat(attention_maps['last_layer']).cpu().numpy()
        return round(np.mean(attn), 4), round(np.std(attn), 4)

def save_analysis_csv(fold_idx, analysis_data):
    csv_path = os.path.join(Config.analysis_dir, f"fold_{fold_idx+1}_task_analysis.csv")
    df = pd.DataFrame(analysis_data)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✅ Analysis CSV Saved: {csv_path}")

def visualize_task_analysis(fold_idx, analysis_data):
    fold_vis_dir = os.path.join(Config.analysis_dir, f"fold_{fold_idx+1}")
    os.makedirs(fold_vis_dir, exist_ok=True)
    epochs = [d['epoch'] for d in analysis_data]

    plt.figure(figsize=(10, 6))
    grad_corr = [d['gradient_correlation'] for d in analysis_data]
    plt.plot(epochs, grad_corr, 'g-', linewidth=2.5, label='Task Gradient Correlation')
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Gradient Cosine Correlation', fontsize=12)
    plt.title(f'Fold {fold_idx+1} - Multi-Task Gradient Correlation Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(fold_vis_dir, 'gradient_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()

    interp_contrib = [d['interp_feature_contrib'] for d in analysis_data]
    seg_contrib = [d['seg_feature_contrib'] for d in analysis_data]
    contrib_matrix = np.array([interp_contrib, seg_contrib])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(contrib_matrix, cmap='viridis', annot=True, fmt='.4f',
                xticklabels=[f'Epoch {i}' for i in epochs],
                yticklabels=['Interpolation Task', 'Segmentation Task'])
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Task Type', fontsize=12)
    plt.title(f'Fold {fold_idx+1} - Feature Contribution Heatmap', fontsize=14)
    plt.savefig(os.path.join(fold_vis_dir, 'feature_contribution_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ All Visualization Figures Saved to: {fold_vis_dir}")

@torch.no_grad()
def validate(model, val_loader, criterion, analyzer):
    model.eval()
    total_loss = 0.0
    interp_loss = 0.0
    seg_loss = 0.0
    val_ssim = 0.0
    val_dice = 0.0
    total_batches = 0
    
    pbar = tqdm(val_loader, desc="Validation", leave=False)
    for batch_idx, batch_data in enumerate(pbar):
        inputs = batch_data[0].to(Config.device)
        interp_target = batch_data[1].to(Config.device).clamp(0.0, 1.0)
        input_labels = batch_data[2].to(Config.device).clamp(0.0, 1.0)
        interp_label = batch_data[3].to(Config.device).clamp(0.0, 1.0)
        s = batch_data[4].squeeze(1).to(Config.device)

        seg_target = torch.cat([input_labels, interp_label], dim=1).clamp(0.0, 1.0)
        
        interp_pred, seg_pred = model(inputs)
        
        loss, interp_l, seg_l = criterion(interp_pred, interp_target, seg_pred, seg_target)
        
        total_loss += loss.item()
        interp_loss += interp_l.item()
        seg_loss += seg_l.item()
        batch_ssim = calculate_ssim(interp_pred, interp_target)
        batch_dice = calculate_dice(seg_pred, seg_target)
        val_ssim += batch_ssim
        val_dice += batch_dice
        total_batches += 1
        
        pbar.set_postfix({
            "total_loss": loss.item(), "interp_loss": interp_l.item(),
            "seg_loss": seg_l.item(), "ssim": batch_ssim, "dice": batch_dice
        })
    
    interp_contrib, seg_contrib = TaskAnalyzer.compute_feature_contribution(
        analyzer.features['last_layer'], interp_pred, seg_pred
    )
    attn_mean, attn_std = 0.0, 0.0
    
    analyzer.features.clear()
    analyzer.attention_maps.clear()

    avg_total_loss = total_loss / total_batches
    avg_interp_loss = interp_loss / total_batches
    avg_seg_loss = seg_loss / total_batches
    avg_val_ssim = val_ssim / total_batches
    avg_val_dice = val_dice / total_batches
    
    return avg_total_loss, avg_interp_loss, avg_seg_loss, avg_val_ssim, avg_val_dice, interp_contrib, seg_contrib, attn_mean, attn_std

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    interp_loss = 0.0
    seg_loss = 0.0
    total_batches = 0
    grad_correlation = 0.0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, batch_data in enumerate(pbar):
        inputs = batch_data[0].to(Config.device)
        interp_target = batch_data[1].to(Config.device).clamp(0.0, 1.0)
        input_labels = batch_data[2].to(Config.device).clamp(0.0, 1.0)
        interp_label = batch_data[3].to(Config.device).clamp(0.0, 1.0)
        s = batch_data[4].squeeze(1).to(Config.device)

        seg_target = torch.cat([input_labels, interp_label], dim=1).clamp(0.0, 1.0)
        
        optimizer.zero_grad()
        interp_pred, seg_pred = model(inputs)
        loss, interp_l, seg_l = criterion(interp_pred, interp_target, seg_pred, seg_target)
        
        grad_correlation = TaskAnalyzer.compute_gradient_correlation(model, interp_l, seg_l)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.grad_clip_norm)
        optimizer.step()
        
        total_loss += loss.item()
        interp_loss += interp_l.item()
        seg_loss += seg_l.item()
        total_batches += 1
        
        if (batch_idx + 1) % Config.log_interval == 0:
            pbar.set_postfix({
                "total_loss": loss.item(), "interp_loss": interp_l.item(),
                "seg_loss": seg_l.item(), "grad_corr": grad_correlation,
                "lr": optimizer.param_groups[0]['lr']
            })
    
    avg_total_loss = total_loss / total_batches
    avg_interp_loss = interp_loss / total_batches
    avg_seg_loss = seg_loss / total_batches
    
    return avg_total_loss, avg_interp_loss, avg_seg_loss, grad_correlation

def plot_loss_curve(fold_idx, train_losses, val_losses, train_interp_losses, val_interp_losses, train_seg_losses, val_seg_losses):
    fold_plot_path = Config.plot_path.replace(".png", f"_fold_{fold_idx+1}.png")
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Total Loss', linewidth=2)
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Total Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.title(f'Fold {fold_idx+1} - Total Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_interp_losses, 'b-', label='Training Interp Loss', linewidth=2)
    plt.plot(epochs_range, val_interp_losses, 'r-', label='Validation Interp Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Interpolation Loss')
    plt.title(f'Fold {fold_idx+1} - Interpolation Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_seg_losses, 'b-', label='Training Seg Loss', linewidth=2)
    plt.plot(epochs_range, val_seg_losses, 'r-', label='Validation Seg Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Segmentation Loss')
    plt.title(f'Fold {fold_idx+1} - Segmentation Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fold_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Loss Curve Saved: {fold_plot_path}")

def train_fold(fold_idx, full_dataset, kf):
    print("\n" + "="*60)
    print(f"Starting Training Fold {fold_idx+1}/{Config.n_folds}")
    print("="*60)
    
    fold_checkpoint_dir = os.path.join(Config.checkpoint_dir, f"fold_{fold_idx+1}")
    os.makedirs(fold_checkpoint_dir, exist_ok=True)
    
    train_loader, val_loader = get_fold_dataloaders(full_dataset, fold_idx, kf)
    model, criterion, optimizer, scheduler = init_model()
    
    analyzer = TaskAnalyzer(model)
    analyzer.register_hooks()
    
    best_model_path = os.path.join(fold_checkpoint_dir, "best_model.pth")
    start_epoch = 1
    best_val_loss = float('inf')
    best_epoch = 0
    early_stop_counter = 0
    
    train_total_losses, train_interp_losses, train_seg_losses = [], [], []
    val_total_losses, val_interp_losses, val_seg_losses = [], [], []
    val_ssims, val_dices = [], []
    analysis_data = []
    
    if os.path.exists(best_model_path):
        print(f"Loading Best Model for Fold {fold_idx+1}...")
        try:
            checkpoint = torch.load(best_model_path, map_location=Config.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            best_epoch = checkpoint['epoch']
            print(f"✅ Resume from Epoch {start_epoch}, Best Validation Total Loss: {best_val_loss:.6f}")
        except Exception as e:
            print(f"❌ Load Model Failed: {e}")
    
    start_time = time.time()
    for epoch in range(start_epoch, Config.epochs + 1):
        print(f"\nFold {fold_idx+1} - Epoch [{epoch}/{Config.epochs}]")
        
        train_total_loss, train_interp_loss, train_seg_loss, grad_corr = train_epoch(
            model, train_loader, criterion, optimizer
        )
        
        val_total_loss, val_interp_loss, val_seg_loss, val_ssim, val_dice, interp_contrib, seg_contrib, attn_mean, attn_std = validate(
            model, val_loader, criterion, analyzer
        )
        
        scheduler.step()
        
        train_total_losses.append(train_total_loss)
        train_interp_losses.append(train_interp_loss)
        train_seg_losses.append(train_seg_loss)
        val_total_losses.append(val_total_loss)
        val_interp_losses.append(val_interp_loss)
        val_seg_losses.append(val_seg_loss)
        val_ssims.append(val_ssim)
        val_dices.append(val_dice)
        
        analysis_data.append({
            'epoch': epoch,
            'gradient_correlation': grad_corr,
            'interp_feature_contrib': interp_contrib,
            'seg_feature_contrib': seg_contrib,
            'attention_mean': attn_mean,
            'attention_std': attn_std,
            'val_ssim': val_ssim,
            'val_dice': val_dice
        })
        
        print(f"Train - Total: {train_total_loss:.6f} | Interp: {train_interp_loss:.6f} | Seg: {train_seg_loss:.6f}")
        print(f"Val   - Total: {val_total_loss:.6f} | Interp: {val_interp_loss:.6f} | Seg: {val_seg_loss:.6f}")
        print(f"Metrics - SSIM: {val_ssim:.4f} | Dice: {val_dice:.4f} | Grad Corr: {grad_corr:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_epoch = epoch
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'current_ssim': val_ssim,
                'current_dice': val_dice,
            }, best_model_path)
            print(f"🎉 Best Model Saved (Validation Total Loss: {best_val_loss:.6f})")
        else:
            early_stop_counter += 1
            print(f"Early Stop Counter: {early_stop_counter}/{Config.patience}")
            if early_stop_counter >= Config.patience:
                print(f"🛑 Early Stop at Epoch {epoch}, Best Validation Total Loss: {best_val_loss:.6f}")
                break
    
    fold_total_time = time.time() - start_time
    print(f"\nFold {fold_idx+1} Training Finished! Time: {fold_total_time / 60:.2f}min")
    
    plot_loss_curve(fold_idx, train_total_losses, val_total_losses, train_interp_losses, val_interp_losses, train_seg_losses, val_seg_losses)
    
    save_analysis_csv(fold_idx, analysis_data)
    visualize_task_analysis(fold_idx, analysis_data)
    
    analyzer.remove_hooks()
    
    return {
        'fold_idx': fold_idx+1,
        'best_val_loss': best_val_loss,
        'best_ssim': val_ssims[best_epoch-1] if (val_ssims and 1 <= best_epoch <= len(val_ssims)) else 0,
        'best_dice': val_dices[best_epoch-1] if (val_dices and 1 <= best_epoch <= len(val_dices)) else 0,
        'best_epoch': best_epoch,
        'total_time': fold_total_time
    }

def main():
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.random_seed)
        torch.backends.cudnn.deterministic = True
    
    full_dataset = load_full_dataset()
    kf = KFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.random_seed)
    
    fold_results = []
    total_start_time = time.time()
    
    for fold_idx in range(Config.n_folds):
        fold_result = train_fold(fold_idx, full_dataset, kf)
        fold_results.append(fold_result)
    
    total_time = time.time() - total_start_time
    print("\n" + "="*60)
    print("5-Fold Cross Validation Completed!")
    print("="*60)
    print(f"Total Time: {total_time / 60:.2f} minutes")
    
    avg_val_loss = np.mean([r['best_val_loss'] for r in fold_results])
    avg_dice = np.mean([r['best_dice'] for r in fold_results])
    avg_ssim = np.mean([r['best_ssim'] for r in fold_results])
    print(f"\nAverage Best Validation Loss: {avg_val_loss:.6f} | Average Dice: {avg_dice:.4f} | Average SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    main()