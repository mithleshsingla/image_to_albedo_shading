#!/usr/bin/env python3
import os
import re
import numpy as np
import h5py
import argparse
from tqdm import tqdm
import torch
import random

def extract_scene_info(file_path):
    """Extract scene, camera, and frame information from a file path."""
    # Extract ai_VVV_NNN
    scene_match = re.search(r'ai_\d+_\d+', file_path)
    scene = scene_match.group(0) if scene_match else None
    
    # Extract cam_XX
    cam_match = re.search(r'cam_\d+', file_path)
    cam = cam_match.group(0) if cam_match else None
    
    # Extract frame.XXXX
    frame_match = re.search(r'frame\.\d+', file_path)
    frame = frame_match.group(0) if frame_match else None
    
    return scene, cam, frame

def compute_tonemapped_image_gpu(rgb_color_tensor, render_entity_id_tensor=None, gamma=1.0/2.2, device='cuda'):
    """
    Compute tonemapped image using the provided gamma value on GPU.
    
    Args:
        rgb_color_tensor: RGB image tensor data (HDR) on GPU
        render_entity_id_tensor: Entity ID tensor for valid mask (optional) on GPU
        gamma: Gamma correction value (default: 1.0/2.2)
        device: Computing device ('cuda' or 'cpu')
        
    Returns:
        tuple: (tonemapped_image_tensor, scale_value)
    """
    inv_gamma = 1.0/gamma
    percentile = 90  # we want this percentile brightness value in the unmodified image...
    brightness_nth_percentile_desired = 0.8  # ...to be this bright after scaling
    
    # Create valid mask (all pixels are valid if no render_entity_id is provided)
    if render_entity_id_tensor is None:
        valid_mask = torch.ones(rgb_color_tensor.shape[:2], dtype=torch.bool, device=device)
    else:
        valid_mask = render_entity_id_tensor != -1
    
    if torch.count_nonzero(valid_mask) == 0:
        scale = 1.0  # if there are no valid pixels, then set scale to 1.0
    else:
        # "CCIR601 YIQ" method for computing brightness
        brightness = 0.3*rgb_color_tensor[:,:,0] + 0.59*rgb_color_tensor[:,:,1] + 0.11*rgb_color_tensor[:,:,2]
        brightness_valid = brightness[valid_mask]
        
        eps = 0.00001  # avoid divide-by-zero
        # Calculate percentile - move to CPU for this operation as torch doesn't have a direct percentile function
        brightness_valid_cpu = brightness_valid.cpu()
        brightness_nth_percentile_current = float(torch.quantile(brightness_valid_cpu, q=percentile/100.0))
        
        if brightness_nth_percentile_current < eps:
            scale = 0.0
        else:
            # Calculate scale factor
            scale = float(torch.pow(torch.tensor(brightness_nth_percentile_desired), inv_gamma) / brightness_nth_percentile_current)
    
    # Apply tone mapping - convert scale to tensor for GPU computation
    scale_tensor = torch.tensor(scale, device=device)
    rgb_color_scaled = torch.clamp(scale_tensor * rgb_color_tensor, min=0.0)
    rgb_color_tm = torch.pow(rgb_color_scaled, gamma)
    
    # Ensure values are in valid range
    rgb_color_tm = torch.clamp(rgb_color_tm, 0.0, 1.0)
    
    return rgb_color_tm, scale

def process_image_gpu(color_file, geometry_dir, output_dir, device, apply_random_exposure=True):
    """
    Process a color HDR image to generate two LDR versions using GPU acceleration:
    1. Linear (gamma=1)
    2. Standard gamma correction (gamma=1.0/2.2)
    
    Args:
        color_file: Path to color.hdf5 file
        geometry_dir: Directory containing geometry data (for render_entity_id)
        output_dir: Output directory for LDR images
        device: Computing device ('cuda' or 'cpu')
        apply_random_exposure: Whether to apply random exposure scaling (default: True)
        
    Returns:
        tuple: (success, message)
    """
    try:
        # Extract scene info
        scene, cam, frame = extract_scene_info(color_file)
        if not all([scene, cam, frame]):
            return False, f"Could not extract scene info from {color_file}"
        
        # Load color image
        with h5py.File(color_file, 'r') as f:
            rgb_color_np = f['dataset'][:].astype(np.float32)
        
        # Convert to PyTorch tensor and move to GPU
        rgb_color_tensor = torch.tensor(rgb_color_np, dtype=torch.float32, device=device)
        
        # Apply random exposure scaling
        exposure_info = ""
        if apply_random_exposure:
            t = np.random.uniform(-3, 3)  # t ∈ [-3, 3]
            exposure_scale = 2 ** t
            rgb_color_tensor = rgb_color_tensor * exposure_scale  # Apply the exposure before tonemapping
            exposure_info = f", exposure_scale={exposure_scale:.4f} (t={t:.4f})"
        
        # Try to load render_entity_id if available
        render_entity_id_tensor = None
        geometry_file = color_file.replace('final_hdf5', 'geometry_hdf5').replace('.color.hdf5', '.render_entity_id.hdf5')
        if os.path.exists(geometry_file):
            try:
                with h5py.File(geometry_file, 'r') as f:
                    render_entity_id_np = f['dataset'][:].astype(np.int32)
                    render_entity_id_tensor = torch.tensor(render_entity_id_np, dtype=torch.int32, device=device)
            except Exception as e:
                print(f"Warning: Could not load render_entity_id from {geometry_file}: {str(e)}")
                print("Proceeding without valid mask")
        
        # Fix: Preserve the complete directory structure
        # Extract the relative path from the input directory for proper structuring
        # Assuming the input structure is like: /hypersim_hdr_ground_truth/ai_001_001/images/scene_cam_00_final_hdf5/...
        
        # Extract the proper directory structure from the input path
        # Input path structure: /hypersim_hdr_ground_truth/ai_XXX_XXX/images/scene_cam_XX_final_hdf5/...
        # We need to preserve the exact directory names
        
        # Get the relative path from the input directory that includes scene and camera folders
        input_dir_parts = color_file.split(os.sep)
        
        # Find the index of the scene directory (ai_XXX_XXX)
        scene_index = -1
        for i, part in enumerate(input_dir_parts):
            if part.startswith('ai_') and '_' in part:
                scene_index = i
                break
        
        if scene_index != -1:
            # Extract the relative path starting from the scene directory
            # This preserves both the scene (ai_XXX_XXX) and the exact camera directory name
            scene_output_path = os.path.sep.join(input_dir_parts[scene_index:-1])
        else:
            # Fallback if scene directory can't be found
            scene_output_path = os.path.basename(os.path.dirname(color_file))
        
        # Create separate output directories for linear and gamma-corrected images
        linear_output_dir = os.path.join(output_dir, "ldr_linear", scene_output_path)
        gamma_output_dir = os.path.join(output_dir, "ldr_gamma", scene_output_path)
        
        os.makedirs(linear_output_dir, exist_ok=True)
        os.makedirs(gamma_output_dir, exist_ok=True)
        
        # Generate output filenames
        base_filename = os.path.basename(color_file).replace('.color.hdf5', '')
        ldr_linear_file = os.path.join(linear_output_dir, f"{base_filename}.ldr_color.hdf5")
        ldr_gamma_file = os.path.join(gamma_output_dir, f"{base_filename}.ldr_gamma_correction_color.hdf5")
        
        # Process with gamma=1 (linear)
        ldr_linear_tensor, scale = compute_tonemapped_image_gpu(rgb_color_tensor, render_entity_id_tensor, gamma=1.0, device=device)
        
        # Use the same scale for gamma=1.0/2.2
        # Apply scale and gamma correction directly
        ldr_gamma_tensor = torch.pow(torch.clamp(scale * rgb_color_tensor, min=0.0), 1.0/2.2)
        ldr_gamma_tensor = torch.clamp(ldr_gamma_tensor, 0.0, 1.0)
        
        # Move tensors back to CPU and convert to numpy arrays for saving
        ldr_linear_np = ldr_linear_tensor.cpu().numpy()
        ldr_gamma_np = ldr_gamma_tensor.cpu().numpy()
        
        # Save LDR linear image
        with h5py.File(ldr_linear_file, 'w') as f:
            f.create_dataset('dataset', data=ldr_linear_np, compression='gzip')
        
        # Save LDR gamma-corrected image
        with h5py.File(ldr_gamma_file, 'w') as f:
            f.create_dataset('dataset', data=ldr_gamma_np, compression='gzip')
        
        # Free GPU memory explicitly
        del rgb_color_tensor, ldr_linear_tensor, ldr_gamma_tensor
        if render_entity_id_tensor is not None:
            del render_entity_id_tensor
        torch.cuda.empty_cache()
        
        return True, f"Processed {os.path.basename(color_file)} with scale={scale}{exposure_info}"
    
    except Exception as e:
        return False, f"Error processing {color_file}: {str(e)}"

def find_color_files(directory):
    """Find all color.hdf5 files in the directory."""
    color_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.color.hdf5'):
                color_files.append(os.path.join(root, file))
    return color_files

def main():
    parser = argparse.ArgumentParser(description="Generate LDR images from Hypersim HDR color images using GPU acceleration")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing color.hdf5 files")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Output directory for LDR images")
    parser.add_argument("--scene", type=str, default=None,
                        help="Process only this scene (e.g., ai_001_001)")
    parser.add_argument("--camera", type=str, default=None,
                        help="Process only this camera (e.g., cam_00)")
    parser.add_argument("--no_random_exposure", action="store_true",
                        help="Disable random exposure scaling")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU instead of GPU")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Number of images to process in parallel (only applicable with sufficient GPU memory)")
    args = parser.parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Select device
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create separate directories for linear and gamma-corrected images
    os.makedirs(os.path.join(args.output_dir, "ldr_linear"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "ldr_gamma"), exist_ok=True)
    
    # Find all color files
    print(f"Looking for color.hdf5 files in {args.input_dir}...")
    color_files = find_color_files(args.input_dir)
    print(f"Found {len(color_files)} color.hdf5 files")
    
    # Filter by scene and camera if specified
    if args.scene or args.camera:
        filtered_files = []
        for file in color_files:
            scene, cam, _ = extract_scene_info(file)
            if (args.scene is None or args.scene == scene) and (args.camera is None or args.camera == cam):
                filtered_files.append(file)
        color_files = filtered_files
        print(f"After filtering: {len(color_files)} files to process")
    
    if not color_files:
        print("No matching color.hdf5 files found. Exiting.")
        return
    
    # Process each color file
    success_count = 0
    error_count = 0
    errors = []
    
    # Single processing for now - batch processing could be implemented but requires careful memory management
    print("Processing files...")
    for color_file in tqdm(color_files):
        # Find corresponding geometry directory
        geometry_dir = os.path.dirname(color_file).replace('final_hdf5', 'geometry_hdf5')
        
        # Process the image
        success, message = process_image_gpu(color_file, geometry_dir, args.output_dir, device, not args.no_random_exposure)
        
        if success:
            success_count += 1
        else:
            error_count += 1
            errors.append(message)
    
    print(f"\nProcessing complete. Successfully processed {success_count} files. Encountered {error_count} errors.")
    
    if errors:
        print("\nErrors encountered:")
        for i, error in enumerate(errors[:10]):
            print(f"{i+1}. {error}")
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more errors")

if __name__ == "__main__":
    main()