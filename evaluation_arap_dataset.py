print("|| Evaluation Script for HDR/Albedo/Shading Dataset ||")
import yaml
import argparse
import torch
import random
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from vae import VAE
from torch.utils.data.dataloader import DataLoader
from collections import OrderedDict
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
import torchvision.utils as vutils
from unet_img_my import Unet
from unet_img_my import Encoder
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel
import cv2
import logging
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Setup device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name(device)} (CUDA:{torch.cuda.current_device()})")
else:
    print("Using device: CPU")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EPSILON = 1e-6

# Author 2's LMSE implementation adapted for Author 1's data
def ssq_error(correct, estimate, mask):
    """Calculate sum-squared error with optimal scaling (Author 2's method)"""
    # Flatten arrays
    correct_flat = correct.flatten()
    estimate_flat = estimate.flatten()
    mask_flat = mask.flatten().astype(bool)
    
    if np.sum(mask_flat) == 0:
        return 0.0
    
    # Extract valid pixels
    correct_masked = correct_flat[mask_flat]
    estimate_masked = estimate_flat[mask_flat]
    
    # Calculate optimal scale factor (Author 2's method)
    if np.sum(estimate_masked**2) > 1e-5:
        alpha = np.sum(correct_masked * estimate_masked) / np.sum(estimate_masked**2)
    else:
        alpha = 0.0
    
    # Calculate scaled prediction and error
    scaled_estimate = alpha * estimate_masked
    error = np.sum((correct_masked - scaled_estimate) ** 2)
    
    return error

def lmse_rgb(correct, estimate, mask, window_size, window_shift):
    """Author 2's RGB LMSE implementation"""
    M, N = correct.shape[:2]
    ssq = total = 0.

    for i in range(0, M - window_size + 1, window_shift):
        for j in range(0, N - window_size + 1, window_shift):

            correct_curr = correct[i:i+window_size, j:j+window_size, :]
            estimate_curr = estimate[i:i+window_size, j:j+window_size, :]
            mask_curr = mask[i:i+window_size, j:j+window_size]

            # Concatenate channels as Author 2 does
            rep_mask = np.concatenate([mask_curr] * 3, 0)
            rep_cor = np.concatenate([correct_curr[:, :, 0], correct_curr[:, :, 1], correct_curr[:, :, 2]], 0)
            rep_est = np.concatenate([estimate_curr[:, :, 0], estimate_curr[:, :, 1], estimate_curr[:, :, 2]], 0)

            ssq += ssq_error(rep_cor, rep_est, rep_mask)
            total += np.sum(rep_mask * rep_cor**2)

    assert ~np.isnan(ssq/total)
    return ssq / total

import numpy as np

def lmse_gray(correct, estimate, mask, window_size, window_shift):
    """
    Local Mean Squared Error (LMSE) for grayscale images with masking
    using Author 2's optimal scaling method.
    """

    # Helper to squeeze and ensure 2D
    def prepare_image(img):
        img = np.squeeze(img)  # remove singleton dims like (1,H,W) or (H,W,1)
        if img.ndim != 2:
            raise ValueError(f"Image must be 2D after squeezing, got {img.shape}")
        return img

    correct = prepare_image(correct)
    estimate = prepare_image(estimate)
    mask = prepare_image(mask)

    # Shape check
    if correct.shape != estimate.shape or correct.shape != mask.shape:
        raise ValueError(f"Shape mismatch after squeezing: "
                         f"{correct.shape}, {estimate.shape}, {mask.shape}")

    M, N = correct.shape
    ssq = total = 0.0

    for i in range(0, M - window_size + 1, window_shift):
        for j in range(0, N - window_size + 1, window_shift):
            correct_curr = correct[i:i+window_size, j:j+window_size]
            estimate_curr = estimate[i:i+window_size, j:j+window_size]
            mask_curr = mask[i:i+window_size, j:j+window_size]

            if np.sum(mask_curr) == 0:
                continue  # no valid pixels in this window

            ssq += ssq_error(correct_curr, estimate_curr, mask_curr)
            total += np.sum((correct_curr**2)[mask_curr.astype(bool)])

    if total < 1e-8:
        return float('nan')
    return ssq / total

def lmse_gray__(correct, estimate, mask, window_size, window_shift):
    """
    Local Mean Squared Error (LMSE) for grayscale images with masking.

    Parameters
    ----------
    correct : np.ndarray
        Ground truth image (H, W) or (H, W, 1) or (1, H, W).
    estimate : np.ndarray
        Estimated image (H, W) or (H, W, 1) or (1, H, W).
    mask : np.ndarray
        Binary mask image (H, W), (H, W, 1), or (1, H, W).
    window_size : int
        Size of the local window.
    window_shift : int
        Step size for moving the window.

    Returns
    -------
    float
        The LMSE value.
    """
    
    # Helper to squeeze and ensure 2D
    def prepare_image(img):
        img = np.squeeze(img)  # remove singleton dims
        if img.ndim != 2:
            raise ValueError(f"Image must be 2D after squeezing, got shape {img.shape}")
        return img

    correct = prepare_image(correct)
    estimate = prepare_image(estimate)
    mask = prepare_image(mask)

    # Ensure all shapes match
    if correct.shape != estimate.shape or correct.shape != mask.shape:
        raise ValueError(f"Shape mismatch after squeezing: "
                         f"correct {correct.shape}, estimate {estimate.shape}, mask {mask.shape}")

    M, N = correct.shape
    ssq = total = 0.0

    for i in range(0, M - window_size + 1, window_shift):
        for j in range(0, N - window_size + 1, window_shift):
            correct_curr = correct[i:i+window_size, j:j+window_size]
            estimate_curr = estimate[i:i+window_size, j:j+window_size]
            mask_curr = mask[i:i+window_size, j:j+window_size]

            # Flatten
            correct_flat = correct_curr.flatten()
            estimate_flat = estimate_curr.flatten()
            mask_flat = mask_curr.flatten().astype(bool)

            if np.sum(mask_flat) == 0:
                continue  # skip windows with no valid pixels

            # Compute squared error for masked pixels
            ssq += np.sum(((correct_flat - estimate_flat) ** 2)[mask_flat])
            total += np.sum((correct_flat ** 2)[mask_flat])

    if total < 1e-8:
        return float('nan')
    return ssq / total

def lmse_gray_o(correct, estimate, mask, window_size, window_shift):
    """Author 2's grayscale LMSE implementation"""
    M, N = correct.shape[:2]
    ssq = total = 0.

    for i in range(0, M - window_size + 1, window_shift):
        for j in range(0, N - window_size + 1, window_shift):

            correct_curr = correct[i:i+window_size, j:j+window_size]
            estimate_curr = estimate[i:i+window_size, j:j+window_size]
            mask_curr = mask[i:i+window_size, j:j+window_size]
            
            ssq += ssq_error(correct_curr, estimate_curr, mask_curr)
            total += np.sum(mask_curr * correct_curr**2)

    assert ~np.isnan(ssq/total)
    return ssq / total

def lmse(correct, estimate, mask, window_size=20, window_shift=10):
    
    """Author 2's main LMSE function"""
    if len(correct.shape) == 2 or correct.shape[-1] == 1:
        return lmse_gray(correct, estimate, mask, window_size, window_shift)
    else:
        return lmse_rgb(correct, estimate, mask, window_size, window_shift)

def compute_grad(img):
    """Author 2's gradient computation"""
    dy = sobel(img, axis=0)
    dx = sobel(img, axis=1)
    return np.stack([dx, dy], axis=-1)

def ssq_grad_error(correct, estimate, mask):
    """Author 2's gradient error calculation"""
    assert correct.ndim == 2
    
    # the mask is (h, w, 2) to compare gradients, 
    # but sometimes we need the (h, w) version..
    single_mask = mask[:, :, 0]
    
    if np.sum(estimate**2 * single_mask) > 1e-5:
        alpha = np.sum(correct * estimate * single_mask) / np.sum(estimate**2 * single_mask)
    else:
        alpha = 0.
        
    scaled_est = alpha * estimate
    est_grad_mag = compute_grad(scaled_est)
    cor_grad_mag = compute_grad(correct)
    
    return np.sum(mask * (cor_grad_mag - est_grad_mag) ** 2), cor_grad_mag

def grad_lmse(correct, estimate, mask, window_size=20, window_shift=10):
    """Author 2's gradient-based LMSE"""
    M, N = correct.shape[:2]
    ssq = total = 0.

    for i in range(0, M - window_size + 1, window_shift):
        for j in range(0, N - window_size + 1, window_shift):
            
            correct_curr = correct[i:i+window_size, j:j+window_size]
            estimate_curr = estimate[i:i+window_size, j:j+window_size]
            mask_curr = mask[i:i+window_size, j:j+window_size]
            
            # repeat mask to create a two channel image
            mask_curr = np.stack([mask_curr] * 2, -1)

            error, corr_grad = ssq_grad_error(correct_curr, estimate_curr, mask_curr)
            ssq += error
            total += np.sum(mask_curr * corr_grad**2)
            
    assert ~np.isnan(ssq/total)
    return ssq / total

def rmse_error(pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
    mask = mask if mask is not None else np.ones_like(pred)
    # Broadcast mask if needed
    if pred.shape != mask.shape:
        mask_expanded = mask.astype(bool)
        if pred.ndim == 3 and mask_expanded.ndim == 2:
            mask_expanded = np.broadcast_to(mask_expanded, pred.shape)
        elif pred.ndim == 3 and mask_expanded.ndim == 3:
            pass  # Already matches
        else:
            # For unexpected shapes, try flattening
            mask_expanded = mask_expanded.flatten()
            pred = pred.flatten()
            target = target.flatten()
    else:
        mask_expanded = mask.astype(bool)

    pred_masked = pred[mask_expanded]
    target_masked = target[mask_expanded]
    diff = (pred_masked - target_masked) ** 2
    valid_pixels = np.sum(mask_expanded)
    error = np.sqrt(np.sum(diff) / (valid_pixels + EPSILON))
    return error

def rmse_error_o(pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
    """Author 2's RMSE calculation"""
    mask = mask if mask is not None else np.ones_like(pred)
    #pred = pred[mask.astype(bool)]
    mask_expanded = mask.astype(bool)[None, :, :]  # Shape: (1, H, W)
    mask_expanded = np.repeat(mask_expanded, pred.shape[0], axis=0)  # Shape: (C, H, W)

    pred = pred[mask_expanded]
    target = target[mask_expanded]

    #target = target[mask.astype(bool)]
    diff = (pred - target) ** 2
    valid_pixels = np.sum(mask)
    error = np.sqrt(np.sum(diff) / (valid_pixels + EPSILON))
    return error

def calculate_scale_invariant_mse_author2_style(pred, target, valid_pixels):
    """
    Calculate scale-invariant MSE using Author 2's approach
    """
    try:
        # Convert to numpy if tensors
        if torch.is_tensor(pred):
            pred_np = pred.cpu().numpy().astype(np.float64)
        else:
            pred_np = pred.astype(np.float64)
            
        if torch.is_tensor(target):
            target_np = target.cpu().numpy().astype(np.float64)
        else:
            target_np = target.astype(np.float64)
            
        if torch.is_tensor(valid_pixels):
            mask_np = valid_pixels.cpu().numpy().astype(bool)
        else:
            mask_np = valid_pixels.astype(bool)
        
        if mask_np.sum() == 0:
            return float('nan')
        
        # Extract valid pixels
        pred_masked = pred_np[mask_np]
        target_masked = target_np[mask_np]
        
        # Author 2's scale calculation method
        if np.sum(pred_masked**2) > 1e-5:
            alpha = np.sum(target_masked * pred_masked) / np.sum(pred_masked**2)
        else:
            alpha = 0.0
        
        # Calculate error
        scaled_pred = alpha * pred_masked
        si_mse = np.mean((scaled_pred - target_masked) ** 2)
        
        return float(si_mse)
        
    except Exception as e:
        print(f"Error in calculate_scale_invariant_mse_author2_style: {e}")
        return float('nan')

def calculate_local_mse_author2_style(pred, target, mask, window_size=20, window_shift=10):
    """
    Calculate Local MSE using Author 2's method
    """
    try:
        # Convert tensors to numpy
        if torch.is_tensor(pred):
            pred_np = pred.cpu().numpy().astype(np.float64)
        else:
            pred_np = pred.astype(np.float64)
            
        if torch.is_tensor(target):
            target_np = target.cpu().numpy().astype(np.float64)
        else:
            target_np = target.astype(np.float64)
            
        if torch.is_tensor(mask):
            mask_np = mask.cpu().numpy().astype(bool)
        else:
            mask_np = mask.astype(bool)
        
        # Handle tensor dimensions (B, C, H, W) -> take first batch and convert to HWC
        if pred_np.ndim == 4:
            pred_np = pred_np[0].transpose(1, 2, 0)  # CHW -> HWC
            target_np = target_np[0].transpose(1, 2, 0)
            mask_np = mask_np[0, 0] if mask_np.shape[1] == 1 else mask_np[0, 0]  # Take first channel
        elif pred_np.ndim == 3 and pred_np.shape[0] <= 3:  # CHW format
            pred_np = pred_np.transpose(1, 2, 0)  # CHW -> HWC
            target_np = target_np.transpose(1, 2, 0)
            mask_np = mask_np[0] if mask_np.shape[0] > 1 else mask_np
        
        # Use Author 2's LMSE function
        return lmse(target_np, pred_np, mask_np, window_size, window_shift)
        
    except Exception as e:
        print(f"Error in calculate_local_mse_author2_style: {e}")
        return float('nan')

def calculate_metrics_with_mask_author2_style(pred, target, mask, metric_name_prefix="", window_size=20):
    """
    Calculate metrics using Author 2's approach
    """
    # Ensure tensors are on CPU for metric calculation
    pred_cpu = pred.detach().cpu()
    target_cpu = target.detach().cpu()
    mask_cpu = mask.detach().cpu()
    
    # Expand mask to match prediction/target channels if needed
    if pred_cpu.shape[1] != mask_cpu.shape[1]:
        if mask_cpu.shape[1] == 1:
            mask_cpu = mask_cpu.expand_as(pred_cpu)
        else:
            mask_cpu = mask_cpu[:, :1].expand_as(pred_cpu)
    
    batch_size = pred_cpu.shape[0]
    si_mse_values = []
    si_rmse_values = []
    lmse_values = []
    ssim_values = []
    rmse_values = []
    valid_pixels_count = []
    
    # Fixed window shift (Author 2's approach - no overlap configuration)
    window_shift = window_size // 2  # 50% overlap like Author 2's examples
    
    for i in range(batch_size):
        pred_batch = pred_cpu[i]
        target_batch = target_cpu[i]
        mask_batch = mask_cpu[i]
        
        # Apply mask - only consider pixels where mask is True
        valid_pixels = mask_batch.bool()
        valid_count = valid_pixels.sum().item()
        
        if valid_count == 0:
            si_mse_values.append(float('nan'))
            si_rmse_values.append(float('nan'))
            lmse_values.append(float('nan'))
            ssim_values.append(float('nan'))
            rmse_values.append(float('nan'))
            valid_pixels_count.append(0)
            continue
        
        valid_pixels_count.append(valid_count)
        
        # Calculate scale-invariant MSE using Author 2's method
        si_mse_val = calculate_scale_invariant_mse_author2_style(pred_batch, target_batch, valid_pixels)
        si_rmse_val = np.sqrt(si_mse_val) if not np.isnan(si_mse_val) and si_mse_val >= 0 else float('nan')
        
        si_mse_values.append(si_mse_val)
        si_rmse_values.append(si_rmse_val)
        
        # Calculate LMSE using Author 2's method
        lmse_val = calculate_local_mse_author2_style(pred_batch, target_batch, mask_batch, window_size, window_shift)
        lmse_values.append(lmse_val)
        
        # Calculate RMSE using Author 2's method
        rmse_val = rmse_error(
            pred_batch.numpy() if torch.is_tensor(pred_batch) else pred_batch,
            target_batch.numpy() if torch.is_tensor(target_batch) else target_batch,
            mask_batch[0].numpy() if torch.is_tensor(mask_batch) else mask_batch[0]  # Take first channel
        )
        rmse_values.append(rmse_val)
        
        # Calculate SSIM (keep Author 1's implementation as Author 2 doesn't have SSIM)
        ssim_val = calculate_ssim_simple(pred_batch, target_batch, mask_batch)
        ssim_values.append(ssim_val)
    
    # Filter out NaN and inf values for averaging
    si_mse_finite = [x for x in si_mse_values if not np.isnan(x) and np.isfinite(x)]
    si_rmse_finite = [x for x in si_rmse_values if not np.isnan(x) and np.isfinite(x)]
    lmse_finite = [x for x in lmse_values if not np.isnan(x) and np.isfinite(x)]
    ssim_finite = [x for x in ssim_values if not np.isnan(x) and np.isfinite(x)]
    rmse_finite = [x for x in rmse_values if not np.isnan(x) and np.isfinite(x)]
    
    # Calculate averages with fallback values
    avg_si_mse = float(np.mean(si_mse_finite)) if si_mse_finite else float('inf')
    avg_si_rmse = float(np.mean(si_rmse_finite)) if si_rmse_finite else float('inf')
    avg_lmse = float(np.mean(lmse_finite)) if lmse_finite else float('inf')
    avg_ssim = float(np.mean(ssim_finite)) if ssim_finite else 0.0
    avg_rmse = float(np.mean(rmse_finite)) if rmse_finite else float('inf')
    
    return {
        f'{metric_name_prefix}SI_MSE': avg_si_mse,
        f'{metric_name_prefix}SI_RMSE': avg_si_rmse,
        f'{metric_name_prefix}LMSE': avg_lmse,
        f'{metric_name_prefix}SSIM': avg_ssim,
        f'{metric_name_prefix}RMSE': avg_rmse,
        f'{metric_name_prefix}Valid_Pixels': float(np.mean(valid_pixels_count))
    }

def calculate_ssim_simple(pred, target, mask, data_range=1.0):
    """
    Simplified SSIM calculation
    """
    try:
        pred_np = pred.cpu().numpy().astype(np.float64)
        target_np = target.cpu().numpy().astype(np.float64)
        mask_np = mask.cpu().numpy().astype(bool)
        
        # Handle multi-channel case
        if pred_np.ndim == 3:
            ssim_values = []
            for c in range(pred_np.shape[0]):
                pred_channel = pred_np[c]
                target_channel = target_np[c]
                mask_channel = mask_np[c] if mask_np.shape[0] > 1 else mask_np[0]
                
                if mask_channel.sum() > 0:
                    # Simple approach: set invalid pixels to 0
                    pred_masked = pred_channel * mask_channel
                    target_masked = target_channel * mask_channel
                    
                    ssim_val = ssim(target_masked, pred_masked, data_range=data_range)
                    if not np.isnan(ssim_val):
                        ssim_values.append(ssim_val)
            
            return float(np.mean(ssim_values)) if ssim_values else float('nan')
        else:
            if mask_np.sum() > 0:
                pred_masked = pred_np * mask_np
                target_masked = target_np * mask_np
                
                ssim_val = ssim(target_masked, pred_masked, data_range=data_range)
                return float(ssim_val) if not np.isnan(ssim_val) else float('nan')
            else:
                return float('nan')
        
    except Exception as e:
        print(f"Error in calculate_ssim_simple: {e}")
        return float('nan')

def calculate_reconstruction_loss_author2_style(predicted_albedo, calculated_shading, original_ldr, mask=None):
    """
    Calculate reconstruction loss using Author 2's approach
    """
    epsilon = 1e-6
    
    # Ensure inputs are in valid ranges
    predicted_albedo = torch.clamp(predicted_albedo, min=epsilon, max=1.0)
    calculated_shading = torch.clamp(calculated_shading, min=0.0, max=10.0)
    
    # Simple multiplication (Author 2's typical approach)
    reconstructed_ldr = predicted_albedo * calculated_shading
    reconstructed_ldr = torch.clamp(reconstructed_ldr, 0.0, 1.0)
    
    # Calculate metrics using Author 2's style
    if mask is not None:
        mask_rgb = mask.expand_as(reconstructed_ldr)
        reconstruction_metrics = calculate_metrics_with_mask_author2_style(
            reconstructed_ldr, original_ldr, mask_rgb, "Reconstruction_"
        )
    else:
        # Simple MSE calculation
        mse = torch.mean((reconstructed_ldr - original_ldr) ** 2).item()
        rmse = np.sqrt(mse)
        
        reconstruction_metrics = {
            'Reconstruction_MSE': mse,
            'Reconstruction_RMSE': rmse,
            'Reconstruction_Valid_Pixels': reconstructed_ldr.numel()
        }
    
    return reconstruction_metrics, reconstructed_ldr

# Keep the rest of Author 1's utility functions as they are needed for the pipeline
def debug_data_ranges(triplet_loader, num_batches=3):
    """Debug function to check data ranges and identify potential issues"""
    print("="*60)
    print("DATA RANGE DEBUGGING")
    print("="*60)
    
    for batch_idx, batch_data in enumerate(triplet_loader):
        if batch_idx >= num_batches:
            break
            
        hdr_im = batch_data['hdr']
        albedo_im = batch_data['albedo'] 
        shading_im = batch_data['shading']
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  HDR - Shape: {hdr_im.shape}, Range: [{hdr_im.min():.6f}, {hdr_im.max():.6f}], Mean: {hdr_im.mean():.6f}")
        print(f"  Albedo - Shape: {albedo_im.shape}, Range: [{albedo_im.min():.6f}, {albedo_im.max():.6f}], Mean: {albedo_im.mean():.6f}")
        print(f"  Shading - Shape: {shading_im.shape}, Range: [{shading_im.min():.6f}, {shading_im.max():.6f}], Mean: {shading_im.mean():.6f}")
        
        # Check for problematic values
        hdr_zeros = (hdr_im == 0).sum().item()
        albedo_zeros = (albedo_im == 0).sum().item()
        shading_zeros = (shading_im == 0).sum().item()
        
        print(f"  Zero pixels - HDR: {hdr_zeros}, Albedo: {albedo_zeros}, Shading: {shading_zeros}")
        
        # Check mask
        mask = create_albedo_mask_fixed(albedo_im, shading_im, threshold=0.004)
        mask_ratio = mask.float().mean().item()
        print(f"  Valid mask ratio: {mask_ratio:.3f}")
        
        if mask_ratio < 0.1:
            print(f"  WARNING: Low valid pixel ratio!")

def strip_orig_mod(state_dict):
    """Remove '_orig_mod.' prefix from state dict keys"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
        new_state_dict[new_key] = v
    return new_state_dict

def read_hdr_image(file_path):
    """Read HDR/EXR image file and convert to torch tensor in CHW format"""
    try:
        import cv2
        
        image = cv2.imread(str(file_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        
        if image is None:
            raise ValueError(f"Could not load image: {file_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return tensor
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def read_png_image(file_path):
    """Read PNG image file and convert to torch tensor in CHW format"""
    try:
        from PIL import Image
        import torchvision.transforms as transforms

        image = Image.open(file_path).convert('RGB')
        tensor = transforms.ToTensor()(image)

        return tensor
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def extract_scene_name(filename):
    """Extract scene name from filename by removing suffixes"""
    name = Path(filename).stem
    suffixes_to_remove = ['_albedo', '_shading', 'albedo', 'shading']
    
    for suffix in suffixes_to_remove:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    
    return name

class HDRAlbedoShadingTripletDataset(torch.utils.data.Dataset):
    """Dataset that loads HDR, albedo, and shading images as triplets from the same scene"""
    def __init__(self, hdr_folder, albedo_folder, shading_folder, transform=None):
        self.hdr_folder = Path(hdr_folder)
        self.albedo_folder = Path(albedo_folder)
        self.shading_folder = Path(shading_folder)
        self.transform = transform
        
        self.triplets = self._create_triplets()
        print(f"Found {len(self.triplets)} valid triplets")
        
        if len(self.triplets) == 0:
            print("WARNING: No valid triplets found!")
    
    def _create_triplets(self):
        """Create triplets by matching scene names across the three folders"""
        triplets = []
        hdr_files = list(self.hdr_folder.glob("*.hdr")) + list(self.hdr_folder.glob("*.png"))
        hdr_files = sorted(hdr_files)
        
        print(f"Found {len(hdr_files)} HDR files")
        
        for hdr_file in hdr_files:
            scene_name = extract_scene_name(hdr_file.name)
            
            albedo_file = self._find_matching_file(self.albedo_folder, scene_name, ['_albedo'])
            if albedo_file is None:
                print(f"Warning: No albedo file found for scene '{scene_name}'")
                continue
                
            shading_file = self._find_matching_file(self.shading_folder, scene_name, ['_shading'])
            if shading_file is None:
                print(f"Warning: No shading file found for scene '{scene_name}'")
                continue
            
            triplets.append({
                'scene_name': scene_name,
                'hdr_path': hdr_file,
                'albedo_path': albedo_file,
                'shading_path': shading_file
            })
            
        return triplets
    
    def _find_matching_file(self, folder, scene_name, suffixes):
        """Find a file in the folder that matches the scene name with given suffixes"""
        for suffix in suffixes + ['']:
            for ext in ['.hdr', '.png']:
                candidate = folder / f"{scene_name}{suffix}{ext}"
                if candidate.exists():
                    return candidate
                
                candidate = folder / f"{scene_name}{ext}"
                if candidate.exists():
                    return candidate
        
        return None
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        
        hdr_image = self._load_image(triplet['hdr_path'], 'hdr')
        if hdr_image is None:
            hdr_image = torch.zeros((3, 256, 256), dtype=torch.float32)
            print(f"Failed to load HDR image: {triplet['hdr_path']}")
        
        if hdr_image.max() > 1.0:
            print("hdr greater than 1")
        hdr_image = hdr_image * 2.0 - 1.0
        
        albedo_image = self._load_image(triplet['albedo_path'], 'albedo')
        if albedo_image is None:
            albedo_image = torch.zeros((3, 256, 256), dtype=torch.float32)
            print(f"Failed to load albedo image: {triplet['albedo_path']}")
        
        if albedo_image.max() > 1.0:
            print("albedo greater than 1")
        
        shading_image = self._load_image(triplet['shading_path'], 'shading')
        if shading_image is None:
            shading_image = torch.zeros((1, 256, 256), dtype=torch.float32)
            print(f"Failed to load shading image: {triplet['shading_path']}")
        
        if shading_image.shape[0] > 1:
            shading_image = shading_image.mean(dim=0, keepdim=True)
        
        if shading_image.max() > 1.0:
            print("shading greater than 1")
        
        if self.transform:
            hdr_image = self.transform(hdr_image)
            albedo_image = self.transform(albedo_image)
            shading_image = self.transform(shading_image)
        
        return {
            'hdr': hdr_image,
            'albedo': albedo_image,
            'shading': shading_image,
            'scene_name': triplet['scene_name']
        }
    
    def _load_image(self, file_path, image_type):
        """Load image based on file extension"""
        if file_path.suffix.lower() in ['.hdr', '.exr']:
            return read_hdr_image(file_path)
        elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            return read_png_image(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            return None

def calculate_shading_from_hdr_albedo(hdr_image, albedo_image, epsilon=1e-6):
    """Calculate shading from HDR image and albedo"""
    hdr_01 = (hdr_image + 1.0) / 2.0
    hdr_01 = torch.clamp(hdr_01, min=0.0, max=1.0)
    albedo_image = torch.clamp(albedo_image, min=0.0, max=1.0)
    
    numerator = torch.sum(hdr_01 * albedo_image, dim=1, keepdim=True)
    denominator = torch.sum(albedo_image ** 2, dim=1, keepdim=True)
    
    shading = numerator / denominator
    return shading

def create_albedo_mask_fixed(ground_truth_albedo, ground_truth_shading, threshold=0.004):
    """Create mask based on albedo and shading thresholds"""
    albedo_mean = torch.mean(ground_truth_albedo, dim=1, keepdim=True)
    
    if ground_truth_shading.shape[1] != 1:
        if ground_truth_shading.shape[1] == 3:
            ground_truth_shading = torch.mean(ground_truth_shading, dim=1, keepdim=True)
        else:
            ground_truth_shading = ground_truth_shading[:, :1]
    
    mask = (albedo_mean >= threshold) & (ground_truth_shading >= threshold)
    
    valid_ratio = mask.float().mean()
    if valid_ratio < 0.1:
        print(f"Warning: Only {valid_ratio:.1%} valid pixels with threshold {threshold}")
        lower_threshold = threshold * 0.5
        mask = (albedo_mean >= lower_threshold) & (ground_truth_shading >= lower_threshold)
        print(f"Using lower threshold {lower_threshold}, valid pixels: {mask.float().mean():.1%}")
    
    return mask

class WrappedModel(ModelWrapper):
    def __init__(self, model, encoder=None):
        super().__init__(model)
        self.encoder = encoder
        self.condition = None
    
    def set_condition(self, condition):
        self.condition = condition
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        if self.condition is None:
            raise ValueError("Condition not set. Call set_condition() first.")
        return self.model(x, t, self.condition)

def save_visualization_grid_with_reconstruction(ldr_im, gt_albedo, gt_shading, pred_albedo, calc_shading, 
                                               reconstructed_ldr, sample_names, save_path, sample_idx=0):
    """Create and save visualization grid showing original images, predictions, and reconstruction"""
    ldr = ldr_im[sample_idx:sample_idx+1]
    gt_alb = gt_albedo[sample_idx:sample_idx+1]
    gt_shd = gt_shading[sample_idx:sample_idx+1]
    pred_alb = pred_albedo[sample_idx:sample_idx+1]
    calc_shd = calc_shading[sample_idx:sample_idx+1]
    recon_ldr = reconstructed_ldr[sample_idx:sample_idx+1]
    
    gt_shd_rgb = gt_shd.repeat(1, 3, 1, 1)
    calc_shd_rgb = calc_shd.repeat(1, 3, 1, 1)
    
    row1 = torch.cat([ldr, gt_alb, gt_shd_rgb], dim=3)
    row2 = torch.cat([recon_ldr, pred_alb, calc_shd_rgb], dim=3)
    
    grid = torch.cat([row1, row2], dim=2)
    
    sample_name = sample_names[sample_idx] if sample_idx < len(sample_names) else f"sample_{sample_idx}"
    filename = f"{sample_name}_full_comparison.png"
    filepath = os.path.join(save_path, filename)
    
    vutils.save_image(grid, filepath, nrow=1, normalize=False)
    return filepath

def evaluate_model_optimized_author2_style(model, encoder, vae, triplet_loader, config, guidance_scales, methods):
    """Modified evaluation using Author 2's metric calculation methods"""
    model.eval()
    encoder.eval() 
    vae.eval()
    
    results = []
    
    train_config = config['train_params']
    base_save_path = os.path.join('/home/project/dataset/arap', train_config['task_name'], 'test_Flow_samples')
    os.makedirs(base_save_path, exist_ok=True)
    
    with torch.no_grad():
        for guidance_scale in tqdm(guidance_scales, desc="Guidance Scales"):
            for method in tqdm(methods, desc="Methods", leave=False):
                
                wrapped_model = WrappedModel(model, encoder)
                solver = ODESolver(velocity_model=wrapped_model)
                
                all_metrics = {
                    'Albedo_SI_MSE': [], 'Albedo_SI_RMSE': [], 'Albedo_LMSE': [], 'Albedo_SSIM': [], 'Albedo_RMSE': [], 'Albedo_Valid_Pixels': [],
                    'Shading_SI_MSE': [], 'Shading_SI_RMSE': [], 'Shading_LMSE': [], 'Shading_SSIM': [], 'Shading_RMSE': [], 'Shading_Valid_Pixels': [],
                    'Reconstruction_SI_MSE': [], 'Reconstruction_SI_RMSE': [], 'Reconstruction_LMSE': [], 'Reconstruction_SSIM': [], 'Reconstruction_RMSE': [], 'Reconstruction_Valid_Pixels': []
                }
                batch_count = 0
                visualization_count = 0
                
                latent_channels = config['autoencoder_params']['z_channels']
                latent_size = config['train_params']['im_size_lt']
                
                for batch_idx, batch_data in enumerate(tqdm(triplet_loader, desc=f"Batches", leave=False)):
                    try:
                        hdr_im = batch_data['hdr'].float().to(device, non_blocking=True)
                        albedo_im = batch_data['albedo'].float().to(device, non_blocking=True)
                        shading_im = batch_data['shading'].float().to(device, non_blocking=True)
                        scene_names = batch_data['scene_name']
                        
                        current_batch_size = hdr_im.size(0)
                        
                        ldr_im = (hdr_im + 1.0) / 2.0
                        ldr_im = torch.clamp(ldr_im, 0.0, 1.0)
                        
                        if batch_idx == 0:
                            print(f"Debug info for first batch:")
                            print(f"  HDR shape: {hdr_im.shape}, range: [{hdr_im.min():.3f}, {hdr_im.max():.3f}]")
                            print(f"  LDR shape: {ldr_im.shape}, range: [{ldr_im.min():.3f}, {ldr_im.max():.3f}]")
                            print(f"  Albedo shape: {albedo_im.shape}, range: [{albedo_im.min():.3f}, {albedo_im.max():.3f}]")
                            print(f"  Shading shape: {shading_im.shape}, range: [{shading_im.min():.3f}, {shading_im.max():.3f}]")
                        
                        albedo_mask = create_albedo_mask_fixed(albedo_im, shading_im, threshold=0.004)
                        
                        x_init = torch.randn(current_batch_size, latent_channels, 
                                           latent_size, latent_size, 
                                           device=device, dtype=torch.float32)
                        
                        with torch.amp.autocast(device_type='cuda'):
                            cond_encoded = encoder(hdr_im)
                            wrapped_model.set_condition(cond_encoded)
                            
                            samples = solver.sample(
                                x_init=x_init,
                                method=method,
                                step_size=0.5,
                                return_intermediates=False
                            )
                            
                            predicted_albedo = vae.decoder(samples.float())
                        
                        predicted_albedo_01 = (predicted_albedo + 1.0) / 2.0
                        predicted_albedo_01 = torch.clamp(predicted_albedo_01, 0.0, 1.0)
                        
                        calculated_shading = calculate_shading_from_hdr_albedo(hdr_im, predicted_albedo_01)
                        calculated_shading = torch.clamp(calculated_shading, 0.0, 1.0)
                        
                        # Use Author 2's style reconstruction loss calculation
                        reconstruction_metrics, reconstructed_ldr = calculate_reconstruction_loss_author2_style(
                            predicted_albedo_01, calculated_shading, ldr_im, albedo_mask
                        )
                        
                        if batch_idx == 0:
                            print(f"  Predicted albedo range: [{predicted_albedo_01.min():.3f}, {predicted_albedo_01.max():.3f}]")
                            print(f"  Calculated shading range: [{calculated_shading.min():.3f}, {calculated_shading.max():.3f}]")
                            print(f"  Reconstructed LDR range: [{reconstructed_ldr.min():.3f}, {reconstructed_ldr.max():.3f}]")
                            print(f"  Original LDR range: [{ldr_im.min():.3f}, {ldr_im.max():.3f}]")
                            print(f"  Mask ratio: {albedo_mask.float().mean():.3f}")
                        
                        # Calculate metrics using Author 2's methods
                        albedo_metrics = calculate_metrics_with_mask_author2_style(
                            predicted_albedo_01, albedo_im, albedo_mask, "Albedo_"
                        )
                        shading_metrics = calculate_metrics_with_mask_author2_style(
                            calculated_shading, shading_im, albedo_mask, "Shading_"
                        )
                        
                        # Store metrics
                        for metric_name, value in {**albedo_metrics, **shading_metrics, **reconstruction_metrics}.items():
                            if metric_name in all_metrics:
                                all_metrics[metric_name].append(value)
                        
                        # Save visualizations
                        if visualization_count < 5:
                            samples_to_save = min(current_batch_size, 5 - visualization_count)
                            for i in range(samples_to_save):
                                try:
                                    filepath = save_visualization_grid_with_reconstruction(
                                        ldr_im, albedo_im, shading_im, 
                                        predicted_albedo_01, calculated_shading, reconstructed_ldr,
                                        scene_names, base_save_path, sample_idx=i
                                    )
                                    logger.info(f"Saved visualization {visualization_count + 1}: {filepath}")
                                    visualization_count += 1
                                except Exception as e:
                                    logger.error(f"Error saving visualization: {e}")
                        
                        batch_count += 1
                        
                        if batch_count % 10 == 0:
                            torch.cuda.empty_cache()
                            
                        # Early exit for debugging
                        if batch_count >= 10:
                            print(f"Early exit after {batch_count} batches for debugging")
                            break
                            
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_count}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # Calculate averages
                avg_metrics = {}
                for metric_name, values in all_metrics.items():
                    if values:
                        finite_values = [v for v in values if np.isfinite(v) and not np.isnan(v)]
                        if finite_values:
                            avg_metrics[metric_name] = np.mean(finite_values)
                            avg_metrics[f'{metric_name}_std'] = np.std(finite_values)
                            avg_metrics[f'{metric_name}_valid_count'] = len(finite_values)
                        else:
                            avg_metrics[metric_name] = float('inf')
                            avg_metrics[f'{metric_name}_std'] = float('inf')
                            avg_metrics[f'{metric_name}_valid_count'] = 0
                    else:
                        avg_metrics[metric_name] = float('inf')
                        avg_metrics[f'{metric_name}_std'] = float('inf')
                        avg_metrics[f'{metric_name}_valid_count'] = 0
                
                result_row = {
                    'guidance_scale': guidance_scale,
                    'method': method,
                    'num_batches_processed': batch_count,
                    'visualizations_saved': visualization_count,
                    **avg_metrics
                }
                results.append(result_row)
                
                print(f"\nResults for guidance_scale={guidance_scale}, method={method} (using Author 2's methods):")
                for key, value in avg_metrics.items():
                    if not key.endswith('_std') and not key.endswith('_valid_count'):
                        if 'SSIM' in key:
                            print(f"  {key}: {value:.4f}")
                        elif np.isfinite(value):
                            print(f"  {key}: {value:.6f}")
                        else:
                            print(f"  {key}: {value}")
    
    return pd.DataFrame(results)

def main(args):
    """Main evaluation function"""
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            return
    
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    logger.info("Loading models...")
    
    vae = VAE(latent_dim=16).to(device)
    vae.eval()
    
    encoder = Encoder(im_channels=3).to(device)
    encoder.eval()
    
    model = Unet(im_channels=autoencoder_config['z_channels']).to(device)
    model.eval()
    
    # Load VAE checkpoint
    vae_path = "checkpoints/epoch_41_best_autoencoder_model_checkpoint.pth"
    if os.path.exists(vae_path):
        logger.info(f'Loading VAE checkpoint from {vae_path}')
        checkpoint_vae = torch.load(vae_path, weights_only=False, map_location=device)
        vae.load_state_dict(checkpoint_vae['model_state_dict'])
        logger.info('VAE loaded successfully')
    else:
        logger.error(f'VAE checkpoint not found at {vae_path}')
        return
    
    # Load flow model checkpoint
    checkpoint_path = "checkpoints/epoch_191_flow_model_ckpt.pth"
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(strip_orig_mod(checkpoint['model_state_dict']))
            encoder.load_state_dict(strip_orig_mod(checkpoint['encoder_state_dict']))
            logger.info("Model and encoder loaded successfully")
        else:
            model.load_state_dict(strip_orig_mod(checkpoint))
            logger.info("Model weights loaded successfully")
    else:
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return
    
    # Freeze all model parameters
    for param in vae.parameters():
        param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = False
    for param in encoder.parameters():
        param.requires_grad = False
    
    logger.info("Loading triplet dataset...")
    
    triplet_dataset = HDRAlbedoShadingTripletDataset(
        hdr_folder=args.hdr_folder,
        albedo_folder=args.albedo_folder,
        shading_folder=args.shading_folder
    )
    
    if len(triplet_dataset) == 0:
        logger.error("No valid triplets found. Please check your dataset structure and file naming.")
        return
    
    triplet_loader = DataLoader(
        triplet_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4
    )
    
    logger.info(f"Triplet dataset size: {len(triplet_dataset)}")
    logger.info(f"Number of batches: {len(triplet_loader)}")
    
    logger.info("Sample triplets:")
    for i in range(min(5, len(triplet_dataset.triplets))):
        triplet = triplet_dataset.triplets[i]
        logger.info(f"  {i+1}. Scene: {triplet['scene_name']}")
        logger.info(f"     HDR: {triplet['hdr_path'].name}")
        logger.info(f"     Albedo: {triplet['albedo_path'].name}")
        logger.info(f"     Shading: {triplet['shading_path'].name}")
        
    guidance_scales = [1.0]
    methods = ['euler']
    
    logger.info(f"Starting evaluation with Author 2's matrix calculation methods")
    debug_data_ranges(triplet_loader)
    
    # Use the modified evaluation function with Author 2's methods
    results_df = evaluate_model_optimized_author2_style(model, encoder, vae, triplet_loader, config, guidance_scales, methods)
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY (Using Author 2's Matrix Calculation Methods)")
    print("="*80)
    print(f"Total experiments completed: {len(results_df)}")
    
    if len(results_df) > 0:
        print("\nBEST RESULTS:")
        print("-"*50)
        
        metrics_to_show = ['Albedo_SI_MSE', 'Albedo_SI_RMSE', 'Albedo_LMSE', 'Albedo_SSIM', 'Albedo_RMSE',
                          'Shading_SI_MSE', 'Shading_SI_RMSE', 'Shading_LMSE', 'Shading_SSIM', 'Shading_RMSE']
        
        for metric in metrics_to_show:
            if metric in results_df.columns:
                if 'SSIM' in metric:
                    if results_df[metric].max() > 0:
                        best_row = results_df.loc[results_df[metric].idxmax()]
                        print(f"Best {metric}: {best_row[metric]:.6f} "
                              f"(guidance_scale={best_row['guidance_scale']}, method={best_row['method']})")
                    else:
                        print(f"Best {metric}: No valid values found")
                else:
                    finite_mask = np.isfinite(results_df[metric]) & (results_df[metric] != float('inf'))
                    if finite_mask.any():
                        best_row = results_df[finite_mask].loc[results_df[finite_mask][metric].idxmin()]
                        print(f"Best {metric}: {best_row[metric]:.6f} "
                              f"(guidance_scale={best_row['guidance_scale']}, method={best_row['method']})")
                    else:
                        print(f"Best {metric}: No finite values found")
        
        print("\nFull results table:")
        print("-"*50)
        key_columns = ['guidance_scale', 'method', 'num_batches_processed'] + metrics_to_show
        display_columns = [col for col in key_columns if col in results_df.columns]
        print(results_df[display_columns].round(6))
        base_save_path = os.path.join('/home/project/dataset/arap', train_config['task_name'], 'test_Flow_samples')
        # Save results to CSV
        results_path = os.path.join(base_save_path, 'evaluation_results_author2_style.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
    else:
        print("No results generated!")
    
    logger.info("Evaluation completed using Author 2's matrix calculation methods!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate flow matching model using Author 2 matrix calculation methods')
    parser.add_argument('--config', dest='config_path',
                       default='config/fine.yaml', type=str,
                       help='Path to configuration file')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation (default: 64)')
    parser.add_argument('--hdr_folder', type=str, default="/mnt/zone/B/mithlesh/dataset/arap/processed_arap_png_256x256/ldr_images_256x256",
                       help='Path to folder containing HDR images')
    parser.add_argument('--albedo_folder', type=str, default="/mnt/zone/B/mithlesh/dataset/arap/processed_arap_png_256x256/albedo_images_256x256",
                       help='Path to folder containing albedo images')
    parser.add_argument('--shading_folder', type=str, default="/mnt/zone/B/mithlesh/dataset/arap/processed_arap_png_256x256/shading_images_256x256",
                       help='Path to folder containing shading images')
    
    import sys
    args, unknown = parser.parse_known_args(sys.argv[1:])
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        raise