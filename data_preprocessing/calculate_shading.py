#!/usr/bin/env python3
import os
import numpy as np
import h5py
import argparse
from tqdm import tqdm
import re
import torch

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
    
def compute_shading_gpu(image, albedo, epsilon=1e-5, device=None, clip_values=True):
    """
    Compute shading by dividing color image by diffuse reflectance (albedo) using GPU acceleration.
    
    Args:
        image: Color image (RGB)
        albedo: Diffuse reflectance image (RGB)
        epsilon: Small value to avoid division by zero
        device: PyTorch device to use
        clip_values: Whether to clip shading values to [0,1] range for LDR output
        
    Returns:
        Single-channel shading image
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert numpy arrays to PyTorch tensors and move to GPU
    image_tensor = torch.from_numpy(image).to(device)
    albedo_tensor = torch.from_numpy(albedo).to(device)
    
    # Compute numerator (sum over channels: image * albedo)
    numerator = torch.sum(image_tensor * albedo_tensor, dim=2)
    
    # Compute denominator (sum of squared albedo values)
    denominator = torch.sum(albedo_tensor ** 2, dim=2) + epsilon
    
    # Shading is numerator / denominator
    shading = numerator / denominator
    
    # Clip values between 0 and 1 for LDR output if requested
    if clip_values:
        shading = torch.clamp(shading, 0.0, 1.0)
    
    # Move result back to CPU and convert to numpy array
    return shading.cpu().numpy()

def compute_shading_batch_gpu(images, albedos, batch_size=4, epsilon=1e-5, clip_values=True):
    """
    Process multiple shading computations in batches for better GPU utilization.
    
    Args:
        images: List of color images
        albedos: List of albedo images
        batch_size: Number of images to process in each batch
        epsilon: Small value to avoid division by zero
        clip_values: Whether to clip shading values to [0,1] range for LDR output
        
    Returns:
        List of shading images
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_albedos = albedos[i:i+batch_size]
        
        # Process each image in the batch
        batch_results = [compute_shading_gpu(img, alb, epsilon, device, clip_values) 
                         for img, alb in zip(batch_images, batch_albedos)]
        results.extend(batch_results)
    
    return results

def find_matching_files(color_dir, albedo_dir):
    """Find matching color and albedo files."""
    matching_pairs = []
    errors = []
    
    # Walk through color directory
    for root, _, files in os.walk(color_dir):
        for file in files:
            if file.endswith('.color.hdf5'):
                color_file_path = os.path.join(root, file)
                
                # Extract scene, camera and frame info
                scene, cam, frame = extract_scene_info(color_file_path)
                
                if not all([scene, cam, frame]):
                    errors.append(f"Could not extract scene info from {color_file_path}")
                    continue
                
                # Construct expected albedo file path
                relative_path = os.path.relpath(root, color_dir)
                expected_albedo_dir = os.path.join(albedo_dir, relative_path)
                albedo_filename = file.replace('.color.hdf5', '.diffuse_reflectance.hdf5')
                expected_albedo_path = os.path.join(expected_albedo_dir, albedo_filename)
                
                if os.path.exists(expected_albedo_path):
                    matching_pairs.append((color_file_path, expected_albedo_path))
                else:
                    errors.append(f"Missing matching albedo file for {color_file_path}")
    
    return matching_pairs, errors

def process_files(color_file, albedo_file, output_dir, epsilon=1e-8, clip_values=True):
    """Process a pair of color and albedo files to generate a shading file."""
    try:
        # Extract scene, camera and frame info for verification
        color_scene, color_cam, color_frame = extract_scene_info(color_file)
        albedo_scene, albedo_cam, albedo_frame = extract_scene_info(albedo_file)
        
        # Verify that both files are from the same scene, camera, and frame
        if color_scene != albedo_scene or color_cam != albedo_cam or color_frame != albedo_frame:
            return False, f"Mismatch: {color_file} and {albedo_file} are not from the same scene/camera/frame"
        
        # Load the color image
        with h5py.File(color_file, 'r') as f:
            color_data = f['dataset'][:].astype(np.float32)
        
        # Load the albedo image
        with h5py.File(albedo_file, 'r') as f:
            albedo_data = f['dataset'][:].astype(np.float32)
        
        # Compute shading using GPU
        shading = compute_shading_gpu(color_data, albedo_data, epsilon, clip_values=clip_values)
        
        # Create output directory structure
        relative_dir = os.path.dirname(os.path.relpath(color_file, os.path.dirname(color_dir)))
        output_subdir = os.path.join(output_dir, relative_dir)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Generate output filename
        base_filename = os.path.basename(color_file).replace('.color.hdf5', '.shading.hdf5')
        output_path = os.path.join(output_subdir, base_filename)
        
        # Save shading as single-channel HDF5
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('dataset', data=shading, compression='gzip')
            
        return True, f"Saved shading to {output_path}"
    
    except Exception as e:
        return False, f"Error processing {color_file}: {str(e)}"

def process_files_in_batches(file_pairs, output_dir, batch_size=4, epsilon=1e-8, clip_values=True):
    """Process multiple file pairs in batches for better GPU utilization."""
    success_count = 0
    error_count = 0
    
    for i in range(0, len(file_pairs), batch_size):
        batch_pairs = file_pairs[i:i+batch_size]
        
        # Load all images and albedos for this batch
        colors = []
        albedos = []
        valid_pairs = []
        error_messages = []
        
        for color_file, albedo_file in batch_pairs:
            try:
                # Extract scene info for verification
                color_scene, color_cam, color_frame = extract_scene_info(color_file)
                albedo_scene, albedo_cam, albedo_frame = extract_scene_info(albedo_file)
                
                # Verify matching
                if color_scene != albedo_scene or color_cam != albedo_cam or color_frame != albedo_frame:
                    error_messages.append(f"Mismatch: {color_file} and {albedo_file} are not from the same scene/camera/frame")
                    error_count += 1
                    continue
                
                # Load data
                with h5py.File(color_file, 'r') as f:
                    color_data = f['dataset'][:].astype(np.float32)
                
                with h5py.File(albedo_file, 'r') as f:
                    albedo_data = f['dataset'][:].astype(np.float32)
                
                colors.append(color_data)
                albedos.append(albedo_data)
                valid_pairs.append((color_file, albedo_file))
                
            except Exception as e:
                error_messages.append(f"Error loading {color_file}: {str(e)}")
                error_count += 1
        
        # Process the batch if any valid pairs exist
        if valid_pairs:
            shadings = compute_shading_batch_gpu(colors, albedos, batch_size, epsilon, clip_values)
            
            # Save results
            for (color_file, _), shading in zip(valid_pairs, shadings):
                try:
                    # Create output directory structure
                    relative_dir = os.path.dirname(os.path.relpath(color_file, os.path.dirname(color_dir)))
                    output_subdir = os.path.join(output_dir, relative_dir)
                    os.makedirs(output_subdir, exist_ok=True)
                    
                    # Generate output filename
                    base_filename = os.path.basename(color_file).replace('.color.hdf5', '.shading.hdf5')
                    output_path = os.path.join(output_subdir, base_filename)
                    
                    # Save shading as single-channel HDF5
                    with h5py.File(output_path, 'w') as f:
                        f.create_dataset('dataset', data=shading, compression='gzip')
                    
                    success_count += 1
                except Exception as e:
                    error_messages.append(f"Error saving {color_file}: {str(e)}")
                    error_count += 1
        
        # Print errors for this batch
        for error in error_messages:
            print(error)
    
    return success_count, error_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate shading images from Hypersim color and albedo images (GPU-accelerated)")
    parser.add_argument("--color_dir", type=str, required=True, help="Directory containing color images (.color.hdf5)")
    parser.add_argument("--albedo_dir", type=str, required=True, help="Directory containing albedo images (.diffuse_reflectance.hdf5)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for shading images")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Small value to add when albedo is 0")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of images to process in each batch")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--no_clip", action="store_true", help="Don't clip shading values to [0,1] range")
    args = parser.parse_args()
    
    # Determine whether to clip values
    clip_values = not args.no_clip
    clip_str = "disabled" if args.no_clip else "enabled"
    print(f"LDR clipping to [0,1] range is {clip_str}")
    
    # Check for GPU availability
    use_gpu = args.gpu and torch.cuda.is_available() and not args.cpu_only
    device_str = "GPU" if use_gpu else "CPU"
    print(f"Using {device_str} for computation")
    
    if args.gpu and not torch.cuda.is_available():
        print("Warning: GPU requested but not available. Using CPU instead.")
    
    # Store directory paths
    color_dir = args.color_dir
    albedo_dir = args.albedo_dir
    output_dir = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Looking for matching files in {color_dir} and {albedo_dir}...")
    matching_pairs, errors = find_matching_files(color_dir, albedo_dir)
    
    if errors:
        print(f"Found {len(errors)} errors while matching files:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    print(f"Found {len(matching_pairs)} matching file pairs")
    
    if not matching_pairs:
        print("No matching files found. Exiting.")
        exit(1)
    
    # Process all matching pairs
    print(f"Processing files with batch size {args.batch_size}...")
    
    if use_gpu:
        # Batch processing for GPU
        success_count, error_count = process_files_in_batches(
            matching_pairs, output_dir, args.batch_size, args.epsilon, clip_values
        )
    else:
        # Individual processing for CPU (original method)
        success_count = 0
        error_count = 0
        for color_file, albedo_file in tqdm(matching_pairs):
            success, message = process_files(color_file, albedo_file, output_dir, args.epsilon, clip_values)
            if success:
                success_count += 1
            else:
                error_count += 1
                print(message)
    
    print(f"Processing complete. Successfully processed {success_count} files. Encountered {error_count} errors.")