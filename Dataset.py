import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
import torch.nn.functional as F
import glob
import pynvml
import random
pynvml.nvmlInit()

def downsample_keep_size(batch_data):
    downsampled = F.interpolate(batch_data, scale_factor=0.5, mode='bilinear', align_corners=False)
    upsampled = F.interpolate(downsampled, size=(512, 512), mode='bilinear', align_corners=False)
    return downsampled

class CTDataset(Dataset):
    def __init__(self, images_dir = "E:\\Interpolation\\Article\\sample_data\\Task02_Heart\\imagesTr", labels_dir = "E:\\Interpolation\\Article\\sample_data\\Task02_Heart\\labelsTr", seq_length=None, split='train', position=0.5):
        self.position = max(0.01, min(0.99, position))
            
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        
        self.file_list = glob.glob(os.path.join(self.images_dir, "la*.nii.gz"))
        if not self.file_list:
            raise ValueError("No .nii.gz files found, please check the path!")
        
        random.seed(42)
        random.shuffle(self.file_list)
        total_len = len(self.file_list)
        
        train_end = int(total_len * 0.7)
        valid_end = train_end + int(total_len * 0.1)
        
        if split == 'train':
            self.data_files = self.file_list[:train_end]
        elif split == 'valid':
            self.data_files = self.file_list[train_end:valid_end]
        elif split == 'test':
            self.data_files = self.file_list[valid_end:]
        else:
            raise ValueError(f"split must be train/valid/test, got: {split}")
        
        print(f"Total data: {total_len} | {split} set data: {len(self.data_files)}")

        self.valid_volumes = []
        for img_path in self.data_files:
            try:
                img_filename = os.path.basename(img_path)
                label_path = os.path.join(self.labels_dir, img_filename)
                
                if not os.path.exists(label_path):
                    print(f"Warning: Label file {label_path} not found, skipped")
                    continue
                
                img = nib.load(img_path)
                img_data = img.get_fdata()
                if len(img_data.shape) != 3:
                    print(f"Warning: File {os.path.basename(img_path)} is not 3D data, shape {img_data.shape}, skipped")
                    continue
                    
                depth = img_data.shape[2]
                if depth >= 3:
                    label_img = nib.load(label_path)
                    label_data = label_img.get_fdata()
                    if label_data.shape != img_data.shape:
                        print(f"Warning: Image and label shape mismatch for {os.path.basename(img_path)}, skipped")
                        continue
                        
                    valid_slice_indices = []
                    for slice_idx in range(depth):
                        slice_label = label_data[:, :, slice_idx]
                        if np.sum(slice_label) > 0:
                            valid_slice_indices.append(slice_idx)
                    
                    if len(valid_slice_indices) < 3:
                        print(f"Warning: Valid label slices ({len(valid_slice_indices)}) < 3 for {os.path.basename(img_path)}, skipped")
                        continue
                    
                    valid_depth = len(valid_slice_indices)
                    valid_samples = valid_depth - 2
                    self.valid_volumes.append((img_path, label_path, valid_depth, valid_slice_indices))
                    print(f"Loaded {os.path.basename(img_path)}, total slices: {depth}, valid slices: {valid_depth}, samples: {valid_samples}")
                else:
                    print(f"Warning: Slices ({depth}) < 3 for {os.path.basename(img_path)}, skipped")
                    
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        if not self.valid_volumes:
            raise ValueError("No valid data files available, check data path and integrity")
    
    def _get_frame_indices(self, base_idx, valid_slice_indices):
        max_available_len = len(valid_slice_indices) - base_idx - 1
        if max_available_len < 2:
            max_available_len = 2
        
        seq_len = int(np.ceil(1 / (1 - self.position)))
        seq_len = min(seq_len, max_available_len)
        
        start_pos = base_idx
        target_pos = base_idx + seq_len
        interp_pos = base_idx + int(seq_len * self.position)
        
        start_pos = min(start_pos, len(valid_slice_indices) - 1)
        target_pos = min(target_pos, len(valid_slice_indices) - 1)
        interp_pos = min(interp_pos, len(valid_slice_indices) - 1)
        
        if start_pos == interp_pos:
            interp_pos += 1
        if interp_pos == target_pos:
            interp_pos -= 1
        if start_pos == target_pos:
            target_pos += 1
        
        start_idx = valid_slice_indices[start_pos]
        target_idx = valid_slice_indices[target_pos]
        interp_idx = valid_slice_indices[interp_pos]
        
        return start_idx, target_idx, interp_idx
    
    def __len__(self):
        total_samples = 0
        for _, _, valid_depth, _ in self.valid_volumes:
            total_samples += max(0, valid_depth - 2)
        return total_samples
    
    def __getitem__(self, idx):
        vol_idx = 0
        remaining_idx = idx
        while vol_idx < len(self.valid_volumes):
            img_path, label_path, valid_depth, valid_slice_indices = self.valid_volumes[vol_idx]
            samples_in_vol = max(0, valid_depth - 2)
            
            if remaining_idx < samples_in_vol:
                base_idx = remaining_idx
                break
                
            remaining_idx -= samples_in_vol
            vol_idx += 1
        else:
            raise IndexError(f"Index {idx} out of range (total samples: {len(self)})")
        
        try:
            img = nib.load(img_path)
            img_volume = img.get_fdata()
            
            label_img = nib.load(label_path)
            label_volume = label_img.get_fdata()
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")
        
        start_idx, target_idx, interp_idx = self._get_frame_indices(base_idx, valid_slice_indices)
        
        depth = img_volume.shape[2]
        for idx_name, frame_idx in zip(['start', 'target', 'interp'], [start_idx, target_idx, interp_idx]):
            if frame_idx >= depth:
                raise IndexError(f"File {os.path.basename(img_path)} {idx_name} frame index {frame_idx} out of range (max: {depth-1})")
        
        img_frames_data = []
        for frame_idx in [start_idx, target_idx, interp_idx]:
            slice_data = img_volume[:, :, frame_idx]
            if slice_data.shape != (512, 512):
                slice_data = resize(slice_data, (512, 512), anti_aliasing=True)
            img_frames_data.append(slice_data)
        
        label_frames_data = []
        for frame_idx in [start_idx, target_idx, interp_idx]:
            slice_data = label_volume[:, :, frame_idx]
            if slice_data.shape != (512, 512):
                slice_data = resize(slice_data, (512, 512), anti_aliasing=True, preserve_range=True)
            label_frames_data.append(slice_data)
        
        img_frames_tensor = np.stack(img_frames_data, axis=0)
        img_frames_tensor = torch.FloatTensor(img_frames_tensor)
        img_frames_tensor = (img_frames_tensor - img_frames_tensor.min()) / (img_frames_tensor.max() - img_frames_tensor.min() + 1e-8)
        img_frames_tensor = img_frames_tensor.unsqueeze(1)
        
        label_frames_tensor = np.stack(label_frames_data, axis=0)
        label_frames_tensor = torch.FloatTensor(label_frames_tensor)
        label_frames_tensor = label_frames_tensor.unsqueeze(1)
        
        input_frames = torch.cat([img_frames_tensor[0], img_frames_tensor[1]], dim=0)
        label_frame = img_frames_tensor[2]
        
        input_labels = torch.cat([label_frames_tensor[0], label_frames_tensor[1]], dim=0)
        label_label = label_frames_tensor[2]
        
        return input_frames, label_frame, input_labels, label_label, torch.tensor([self.position], dtype=torch.float32)
        

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        
    def forward(self, pred, target):
        return self.alpha * self.l1(pred, target) + (1 - self.alpha) * self.l2(pred, target)

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.4, beta=0.6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.interpolation_loss = HybridLoss(alpha=0.7)
        self.segmentation_loss = nn.BCELoss()

    def forward(self, interpolation_pred, interpolation_target, segmentation_pred, segmentation_target):
        interp_loss = self.interpolation_loss(interpolation_pred, interpolation_target)
        seg_loss = self.segmentation_loss(segmentation_pred, segmentation_target)
        return self.alpha * interp_loss + self.beta * seg_loss

def test_dataset():
    print("="*60)
    print("Start testing CTDataset (empty label filtering + 7:1:2 split)")
    print("="*60)
    
    test_cases = [
        (0.5, "Middle position (frame 1,3 and 2)"),
        (0.33, "1/3 position (frame 1,4 and 2)"),
        (0.6, "0.6 position (frame 1,6 and 4)"),
        (0.25, "1/4 position (frame 1,5 and 2)")
    ]
    
    for split_name in ['train', 'valid', 'test']:
        print(f"\n{'='*40}")
        print(f"Testing {split_name} set")
        print(f"{'='*40}")
        
        for position, desc in test_cases:
            print(f"\n{'-'*50}")
            print(f"Test case: {desc} (position={position})")
            print(f"{'-'*50}")
            
            try:
                dataset = CTDataset(
                    images_dir="E:\\Interpolation\\Article\\Data\\Task02_Heart\\imagesTr",
                    labels_dir="E:\\Interpolation\\Article\\Data\\Task02_Heart\\labelsTr",
                    position=position,
                    split=split_name
                )
                print(f"Total samples: {len(dataset)}")
                
                if len(dataset) > 0:
                    input_frames, label_frame, input_labels, label_label, position_tensor = dataset[0]
                    
                    print(f"Sample 0 shape info:")
                    print(f"  Image input shape: {input_frames.shape} (expected: [2, 512, 512])")
                    print(f"  Image label shape: {label_frame.shape} (expected: [1, 512, 512])")
                    print(f"  Seg input shape: {input_labels.shape} (expected: [2, 512, 512])")
                    print(f"  Seg label shape: {label_label.shape} (expected: [1, 512, 512])")
                    print(f"  Position value: {position_tensor.item():.4f}")
                    
                    print(f"\nValue range check:")
                    print(f"  Image input range: [{input_frames.min():.4f}, {input_frames.max():.4f}]")
                    print(f"  Seg input range: [{input_labels.min():.4f}, {input_labels.max():.4f}]")
                    
                    if torch.sum(label_label) == 0:
                        print("Warning: Empty label frame!")
                    else:
                        print("✓ Valid segmentation label found")
                    
                    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
                    batch_data = next(iter(dataloader))
                    batch_input, batch_label, batch_input_labels, batch_label_labels, batch_positions = batch_data
                    print(f"\nDataLoader test:")
                    print(f"  Batch image input: {batch_input.shape}")
                    print(f"  Batch image label: {batch_label.shape}")
                    print(f"  Batch seg input: {batch_input_labels.shape}")
                    print(f"  Batch seg label: {batch_label_labels.shape}")
                    print(f"  Batch position: {batch_positions.shape}")
                    
                else:
                    print("Warning: Empty dataset, check data path")
                    
            except Exception as e:
                print(f"Test failed: {e}")
                import traceback
                traceback.print_exc()

def test_loss_functions():
    print(f"\n{'='*60}")
    print("Testing loss functions")
    print(f"{'='*60}")
    
    try:
        pred = torch.randn(4, 1, 512, 512)
        target = torch.randn(4, 1, 512, 512)
        seg_pred = torch.sigmoid(torch.randn(4, 1, 512, 512))
        seg_target = torch.randint(0, 2, (4, 1, 512, 512), dtype=torch.float32)
        
        hybrid_loss = HybridLoss(alpha=0.5)
        loss1 = hybrid_loss(pred, target)
        print(f"HybridLoss value: {loss1.item():.4f}")
        
        multi_loss = MultiTaskLoss(alpha=0.4, beta=0.6)
        loss2 = multi_loss(pred, target, seg_pred, seg_target)
        print(f"MultiTaskLoss value: {loss2.item():.4f}")
        
        print("Loss function test passed!")
        
    except Exception as e:
        print(f"Loss test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
    test_loss_functions()
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")