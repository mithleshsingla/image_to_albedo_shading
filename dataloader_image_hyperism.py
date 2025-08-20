import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import re
import OpenEXR
import Imath
import array
import numpy as np
import torchvision.transforms.functional as F
import random                
import h5py


class ImageDatasetwv_h5_al(Dataset):
    def __init__(self, im_path, im_size, split='train', file_suffix='.diffuse_reflectance.exr', latent_path=None):
        self.im_path = im_path
        self.im_size = im_size
        self.file_suffix = file_suffix
        self.latent_path = latent_path
        self.image_files = []
        self.metadata = []
        
        # Load images using the same logic as ldr_to_sh_Dataset
        if latent_path and latent_path.endswith('.h5') and os.path.exists(latent_path):
            self._load_with_h5_reference(split)
        else:
            self._load_with_default_ordering(split)
        
        print(f"Found {len(self.image_files)} images for {split} in ImageDatasetwv_al")
        
        self.transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.CenterCrop((im_size, im_size)),
            transforms.ToTensor()
        ])
    
    # Fix for ImageDatasetwv_h5_al class to ensure proper synchronization
# Replace the matching section in _load_with_h5_reference method

    # Complete replacement for the ImageDatasetwv_h5_al._load_with_h5_reference method
# This ensures EXACT synchronization with the latent dataset

    def _load_with_h5_reference(self, split):
        """Load images using exactly the same logic and ORDER as ldr_to_sh_Dataset"""
        try:
            with h5py.File(self.latent_path, 'r') as f:
                print("Keys in H5 file:", list(f.keys()))
                
                # Load latents to know the count
                latents = f['latents'][:]
                print(f"Found {len(latents)} latent vectors in H5 file")
                
                # Extract metadata exactly as in ldr_to_sh_Dataset
                metadata = {}
                if 'metadata' in f:
                    metadata_group = f['metadata']
                    print("Metadata keys:", list(metadata_group.keys()))
                    
                    for key in metadata_group.keys():
                        if isinstance(metadata_group[key], h5py.Dataset):
                            data = metadata_group[key][:]
                            if data.dtype.kind == 'O' and isinstance(data[0], bytes):
                                data = [item.decode('utf-8') for item in data]
                            metadata[key] = data
                            print(f"Loaded metadata/{key} with {len(data)} entries")
                
                # Get all available albedo images
                print(f"Loading albedo images from {self.im_path}")
                
                image_files = []
                for ai_folder in os.listdir(self.im_path):
                    ai_path = os.path.join(self.im_path, ai_folder)
                    if not os.path.isdir(ai_path) or not ai_folder.startswith('ai_'):
                        continue
                    
                    images_path = os.path.join(ai_path, 'images')
                    if not os.path.isdir(images_path):
                        continue
                    
                    for scene_folder in os.listdir(images_path):
                        scene_path = os.path.join(images_path, scene_folder)
                        if not os.path.isdir(scene_path) or not scene_folder.startswith('scene_cam_'):
                            continue
                        
                        for file in os.listdir(scene_path):
                            if file.startswith('frame.') and file.endswith(self.file_suffix):
                                image_files.append(os.path.join(scene_path, file))
                
                print(f"Found {len(image_files)} albedo image files")
                
                if len(image_files) == 0:
                    print("No albedo image files found")
                    self.image_files = []
                    return
                
                # Create a lookup dictionary for fast albedo image finding
                albedo_lookup = {}
                for img_path in image_files:
                    parts = os.path.normpath(img_path).split(os.sep)
                    
                    ai_folder = None
                    scene_folder = None
                    frame_num = None
                    
                    for part in parts:
                        if part.startswith('ai_'):
                            ai_folder = part
                        elif part.startswith('scene_cam_'):
                            scene_folder = part
                    
                    filename = os.path.basename(img_path)
                    match = re.search(r'frame\.(\d+)\.diffuse_reflectance\.exr', filename)
                    if match:
                        frame_num = match.group(1)
                    
                    if ai_folder and scene_folder and frame_num:
                        key = (ai_folder, scene_folder, frame_num)
                        albedo_lookup[key] = img_path
                
                print(f"Created lookup table with {len(albedo_lookup)} albedo images")
                
                # CRITICAL: Use metadata to build image list in EXACT same order as latents
                if 'ai_folders' in metadata and 'scene_folders' in metadata and 'frame_nums' in metadata:
                    print("Building albedo image list in EXACT metadata order")
                    
                    ordered_albedo_paths = []
                    missing_count = 0
                    
                    # Process metadata in EXACT order (same as latent dataset does)
                    for i in range(len(metadata['ai_folders'])):
                        ai_folder = metadata['ai_folders'][i]
                        scene_folder = metadata['scene_folders'][i]
                        frame_num = metadata['frame_nums'][i]
                        
                        # Look up corresponding albedo image
                        key = (ai_folder, scene_folder, frame_num)
                        
                        if key in albedo_lookup:
                            ordered_albedo_paths.append(albedo_lookup[key])
                        else:
                            print(f"WARNING: Missing albedo image for {key}")
                            missing_count += 1
                            # You might want to add a placeholder or skip this entry
                            # For now, we'll skip missing entries
                    
                    print(f"Successfully ordered {len(ordered_albedo_paths)} albedo images")
                    print(f"Missing albedo images: {missing_count}")
                    
                    if len(ordered_albedo_paths) == 0:
                        print("ERROR: No albedo images could be matched!")
                        self.image_files = []
                        return
                    
                    # Ensure we don't exceed the latent count
                    count = min(len(latents), len(ordered_albedo_paths))
                    self.image_files = ordered_albedo_paths[:count]
                    
                    # Apply train/val split EXACTLY as in ldr_to_sh_Dataset
                    if split == 'train':
                        split_idx = int(len(self.image_files) * 0.9)
                        self.image_files = self.image_files[:split_idx]
                    else:
                        split_idx = int(len(self.image_files) * 0.9)
                        self.image_files = self.image_files[split_idx:]
                    
                    print(f"Final albedo dataset size for {split}: {len(self.image_files)}")
                    
                    # Debug: Print first few to verify order
                    print("First 5 albedo images after ordering:")
                    for i in range(min(5, len(self.image_files))):
                        filename = os.path.basename(self.image_files[i])
                        frame_match = re.search(r'frame\.(\d+)', filename)
                        frame_num = frame_match.group(1) if frame_match else "unknown"
                        print(f"  {i}: frame.{frame_num}")
                    
                    return
                
                # Fallback if metadata is not available
                print("ERROR: Required metadata not found for albedo synchronization")
                self.image_files = []
                
        except Exception as e:
            print(f"Error in albedo H5 loading: {e}")
            import traceback
            traceback.print_exc()
            self._load_with_default_ordering(split)

    def get_scene_info(self, idx):
        """Extract scene info from image path"""
        img_path = self.image_files[idx]
        parts = os.path.normpath(img_path).split(os.sep)
        
        ai_folder = None
        scene_folder = None
        frame_num = None
        
        for part in parts:
            if part.startswith('ai_'):
                ai_folder = part
            elif part.startswith('scene_cam_'):
                scene_folder = part
        
        filename = os.path.basename(img_path)
        match = re.search(r'frame\.(\d+)', filename)
        if match:
            frame_num = match.group(1)
        
        return {
            'ai_folder': ai_folder,
            'scene_folder': scene_folder, 
            'frame_num': frame_num,
            'path': img_path
        }
                    
    def _load_with_h5_reference(self, split):
        """Load images using exactly the same logic as ldr_to_sh_Dataset"""
        try:
            with h5py.File(self.latent_path, 'r') as f:
                print("Keys in H5 file:", list(f.keys()))
                
                # Load latents to know the count
                latents = f['latents'][:]
                print(f"Found {len(latents)} latent vectors in H5 file")
                
                # Extract metadata exactly as in ldr_to_sh_Dataset
                metadata = {}
                if 'metadata' in f:
                    metadata_group = f['metadata']
                    print("Metadata keys:", list(metadata_group.keys()))
                    
                    for key in metadata_group.keys():
                        if isinstance(metadata_group[key], h5py.Dataset):
                            data = metadata_group[key][:]
                            if data.dtype.kind == 'O' and isinstance(data[0], bytes):
                                data = [item.decode('utf-8') for item in data]
                            metadata[key] = data
                            print(f"Loaded metadata/{key} with {len(data)} entries")
                
                # Get all available images with EXACT same logic as ldr_to_sh_Dataset
                print(f"Loading images from {self.im_path} with correct nested structure")
                
                image_files = []
                for ai_folder in os.listdir(self.im_path):
                    ai_path = os.path.join(self.im_path, ai_folder)
                    if not os.path.isdir(ai_path) or not ai_folder.startswith('ai_'):
                        continue
                    
                    images_path = os.path.join(ai_path, 'images')
                    if not os.path.isdir(images_path):
                        continue
                    
                    for scene_folder in os.listdir(images_path):
                        scene_path = os.path.join(images_path, scene_folder)
                        if not os.path.isdir(scene_path) or not scene_folder.startswith('scene_cam_'):
                            continue
                        
                        for file in os.listdir(scene_path):
                            if file.startswith('frame.') and file.endswith(self.file_suffix):
                                image_files.append(os.path.join(scene_path, file))
                
                print(f"Found {len(image_files)} image files with pattern 'frame.*.diffuse_reflectance.exr'")
                
                if len(image_files) == 0:
                    print("No image files found")
                    self.image_files = []
                    return
                
                # Extract info from image paths EXACTLY as in ldr_to_sh_Dataset
                extracted_info = []
                for img_path in image_files:
                    parts = os.path.normpath(img_path).split(os.sep)
                    
                    ai_folder = None
                    scene_folder = None
                    frame_num = None
                    
                    for part in parts:
                        if part.startswith('ai_'):
                            ai_folder = part
                        elif part.startswith('scene_cam_'):
                            scene_folder = part
                    
                    filename = os.path.basename(img_path)
                    match = re.search(r'frame\.(\d+)\.', filename)
                    if match:
                        frame_num = match.group(1)
                    
                    extracted_info.append({
                        'path': img_path,
                        'ai_folder': ai_folder,
                        'scene_folder': scene_folder,
                        'frame_num': frame_num
                    })
                
                # Try metadata matching first (same as ldr_to_sh_Dataset)
                if 'ai_folders' in metadata and 'scene_folders' in metadata and 'frame_nums' in metadata:
                    print("Attempting to match with ai_folders, scene_folders, and frame_nums metadata")
                    
                    matched_indices = []
                    matched_image_paths = []
                    
                    for info in extracted_info:
                        for i in range(len(metadata['ai_folders'])):
                            if (info['ai_folder'] == metadata['ai_folders'][i] and
                                info['scene_folder'] == metadata['scene_folders'][i] and
                                info['frame_num'] == metadata['frame_nums'][i]):
                                matched_indices.append(i)
                                matched_image_paths.append(info['path'])
                                break
                    
                    print(f"Matched {len(matched_indices)} images with metadata")
                    
                    if len(matched_indices) > 0:
                        # Use the SAME ORDER as the matched indices
                        self.image_files = matched_image_paths
                        
                        # Apply train/val split EXACTLY as in ldr_to_sh_Dataset
                        if split == 'train':
                            split_idx = int(len(self.image_files) * 0.9)
                            self.image_files = self.image_files[:split_idx]
                        else:
                            split_idx = int(len(self.image_files) * 0.9)
                            self.image_files = self.image_files[split_idx:]
                        
                        print(f"Loaded {len(self.image_files)} matched images for {split}")
                        return
                
                # Fallback to sequential approach (same as ldr_to_sh_Dataset)
                print("No matches found with metadata, using sequential approach")
                
                # Sort EXACTLY as in ldr_to_sh_Dataset
                sorted_images = sorted(extracted_info, key=lambda x: (
                    x['ai_folder'] or '',
                    x['scene_folder'] or '',
                    x['frame_num'] or ''
                ))
                
                sorted_image_paths = [info['path'] for info in sorted_images]
                
                # Use min of latent count and image count (same as ldr_to_sh_Dataset)
                count = min(len(latents), len(sorted_image_paths))
                
                if count == 0:
                    raise ValueError("No data available - either no latents or no images were found")
                
                # Get subset (same as ldr_to_sh_Dataset)
                images_subset = sorted_image_paths[:count]
                
                # Apply train/val split EXACTLY as in ldr_to_sh_Dataset
                if split == 'train':
                    split_idx = int(count * 0.9)
                    self.image_files = images_subset[:split_idx]
                else:
                    split_idx = int(count * 0.9)
                    self.image_files = images_subset[split_idx:]
                
                print(f"Loaded {len(self.image_files)} images for {split} using sequential matching")
                
        except Exception as e:
            print(f"Error loading with H5 reference: {e}")
            self._load_with_default_ordering(split)
    
    def _load_with_default_ordering(self, split):
        """Fallback method when H5 is not available"""
        all_image_files = []
        
        # Same nested structure traversal
        for ai_folder in os.listdir(self.im_path):
            ai_path = os.path.join(self.im_path, ai_folder)
            if not os.path.isdir(ai_path) or not ai_folder.startswith('ai_'):
                continue
            
            images_path = os.path.join(ai_path, 'images')
            if not os.path.isdir(images_path):
                continue
            
            for scene_folder in os.listdir(images_path):
                scene_path = os.path.join(images_path, scene_folder)
                if not os.path.isdir(scene_path) or not scene_folder.startswith('scene_cam_'):
                    continue
                
                for file in os.listdir(scene_path):
                    if file.startswith('frame.') and file.endswith(self.file_suffix):
                        all_image_files.append(os.path.join(scene_path, file))
        
        # Sort by path to ensure consistent ordering
        all_image_files.sort()
        
        # Apply train/val split
        if split == 'train':
            split_idx = int(len(all_image_files) * 0.9)
            self.image_files = all_image_files[:split_idx]
        else:
            split_idx = int(len(all_image_files) * 0.9)
            self.image_files = all_image_files[split_idx:]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        if img_path.endswith('.exr'):
            try:
                if OpenEXR is None:
                    raise ImportError("OpenEXR not available")
                
                # Open the input file
                exr_file = OpenEXR.InputFile(img_path)
                
                # Get the header and extract data window dimensions
                header = exr_file.header()
                dw = header['dataWindow']
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1
                
                # Read the RGB channels
                FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                r_str = exr_file.channel('R', FLOAT)
                g_str = exr_file.channel('G', FLOAT)
                b_str = exr_file.channel('B', FLOAT)
                
                # Convert to numpy arrays
                r = np.array(array.array('f', r_str)).reshape(height, width)
                g = np.array(array.array('f', g_str)).reshape(height, width)
                b = np.array(array.array('f', b_str)).reshape(height, width)
                
                # Create RGB image and handle -1 values
                rgb = np.stack([r, g, b], axis=2)
                
                # Transform from [0,∞) to [-1,1] range (assuming original is in [0,1] range)
                rgb = (2*rgb) - 1  # Convert to [-1, 1] range
                
                # Convert directly to PyTorch tensor, preserving float values
                img_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()  # Convert to CxHxW format
                
                return img_tensor
                
            except (ImportError, Exception) as e:
                # Fallback method if OpenEXR is not available or other error
                print(f"Warning: Could not load EXR file {img_path}: {e}")
                # Return a zero tensor with the right dimensions
                return torch.zeros(3, self.im_size, self.im_size)
        else:
            # For standard images, use the regular PIL pipeline
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img                            
        

class ImageDataset(Dataset):
    def __init__(self, im_path, im_size, file_suffix):
        """
        im_path: Base path for the dataset
        im_size: Size to resize images to
        file_suffix: Suffix part of the filename to match (e.g., 'diffuse_reflectance.exr' or 'dequantize.exr')
        """
        self.im_path = im_path
        self.im_size = im_size
        self.file_suffix = file_suffix
        
        # Find all image files with the specified pattern
        self.image_files = []
        self.scene_info = []  # Store metadata for each image
        
        # Walk through the nested directories
        for ai_folder in os.listdir(im_path):
            ai_path = os.path.join(im_path, ai_folder)
            if not os.path.isdir(ai_path) or not ai_folder.startswith('ai_'):
                continue
                
            images_path = os.path.join(ai_path, 'images')
            if not os.path.isdir(images_path):
                continue
                
            for scene_folder in os.listdir(images_path):
                scene_path = os.path.join(images_path, scene_folder)
                if not os.path.isdir(scene_path) or not scene_folder.startswith('scene_cam_'):
                    continue
                    
                for file in os.listdir(scene_path):
                    if file.startswith('frame.') and file.endswith(file_suffix):
                        self.image_files.append(os.path.join(scene_path, file))
                        
                        # Extract frame number
                        frame_match = re.search(r'frame\.(\d+)\.', file)
                        frame_num = frame_match.group(1) if frame_match else None
                        
                        # Store metadata for matching
                        self.scene_info.append({
                            'ai_folder': ai_folder,
                            'scene_folder': scene_folder,
                            'frame_num': frame_num
                        })
        
        self.transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.CenterCrop((im_size, im_size)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def get_scene_info(self, idx):
        """Return scene information for matching across datasets"""
        return self.scene_info[idx]
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        if img_path.endswith('.exr'):
            try:
                
                # Open the input file
                exr_file = OpenEXR.InputFile(img_path)
                
                # Get the header and extract data window dimensions
                header = exr_file.header()
                dw = header['dataWindow']
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1
                
                # Read the RGB channels
                FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                r_str = exr_file.channel('R', FLOAT)
                g_str = exr_file.channel('G', FLOAT)
                b_str = exr_file.channel('B', FLOAT)
                
                # Convert to numpy arrays
                r = np.array(array.array('f', r_str)).reshape(height, width)
                g = np.array(array.array('f', g_str)).reshape(height, width)
                b = np.array(array.array('f', b_str)).reshape(height, width)
                
                # Create RGB image and handle -1 values
                rgb = np.stack([r, g, b], axis=2)
                epsilon = 1e-8
                #rgb = np.where(rgb == 0, rgb + epsilon, rgb)
                rgb=2*rgb-1
                
                # Convert directly to PyTorch tensor, preserving float values
                img_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()  # Convert to CxHxW format
                
                # Handle resize with aspect ratio preservation
                # First determine scale factor to get shortest side to 256
                # c, h, w = img_tensor.shape
                # scale_factor = self.im_size / min(h, w)
                # new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                
                # # Resize the image preserving aspect ratio
                # img_tensor = torch.nn.functional.interpolate(
                #     img_tensor.unsqueeze(0),  # Add batch dimension
                #     size=(new_h, new_w),
                #     mode='bilinear',
                #     align_corners=False
                # ).squeeze(0)  # Remove batch dimension
                
                # # Center crop to get 256x256
                # _, h, w = img_tensor.shape
                # h_start = max(0, (h - self.im_size) // 2)
                # w_start = max(0, (w - self.im_size) // 2)
                # img_tensor = img_tensor[:, h_start:h_start+self.im_size, w_start:w_start+self.im_size]
                # Random Horizontal Flip
                if random.random() > 0.5:
                    img_tensor = F.hflip(img_tensor)

                # (Optional) Basic color jitter – simulate brightness change
                brightness_factor = random.uniform(0.8, 1.2)
                img_tensor = img_tensor * brightness_factor
                img_tensor = torch.clamp(img_tensor, -1.0, 1.0)

                return img_tensor
                
            except ImportError:
                # Fallback method if OpenEXR is not available
                print(f"Warning: OpenEXR package not found. Skipping {img_path}")
                # Return a zero tensor with the right dimensions
                return torch.zeros(3, self.im_size, self.im_size)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                return torch.zeros(3, self.im_size, self.im_size)
        else:
            # For standard images, use the regular PIL pipeline
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img
        


class ImageDatasetwv_h5(Dataset):
    def __init__(self, im_path, im_size, split='train', file_suffix='_dequantize.exr', latent_path=None):
        self.im_path = im_path
        self.im_size = im_size
        self.file_suffix = file_suffix
        self.latent_path = latent_path
        self.image_files = []
        self.metadata = []
        
        # Load images using the same logic as ldr_to_sh_Dataset
        if latent_path and latent_path.endswith('.h5') and os.path.exists(latent_path):
            self._load_with_h5_reference(split)
        else:
            self._load_with_default_ordering(split)
        
        print(f"Found {len(self.image_files)} images for {split} in ImageDatasetwv")
        
        self.transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.CenterCrop((im_size, im_size)),
            transforms.ToTensor()
        ])
    
    def _load_with_h5_reference(self, split):
        """Load images using exactly the same logic as ldr_to_sh_Dataset"""
        try:
            with h5py.File(self.latent_path, 'r') as f:
                print("Keys in H5 file:", list(f.keys()))
                
                # Load latents to know the count
                latents = f['latents'][:]
                print(f"Found {len(latents)} latent vectors in H5 file")
                
                # Extract metadata exactly as in ldr_to_sh_Dataset
                metadata = {}
                if 'metadata' in f:
                    metadata_group = f['metadata']
                    print("Metadata keys:", list(metadata_group.keys()))
                    
                    for key in metadata_group.keys():
                        if isinstance(metadata_group[key], h5py.Dataset):
                            data = metadata_group[key][:]
                            if data.dtype.kind == 'O' and isinstance(data[0], bytes):
                                data = [item.decode('utf-8') for item in data]
                            metadata[key] = data
                            print(f"Loaded metadata/{key} with {len(data)} entries")
                
                # Get all available images with EXACT same logic as ldr_to_sh_Dataset
                print(f"Loading images from {self.im_path} with correct nested structure")
                
                image_files = []
                for ai_folder in os.listdir(self.im_path):
                    ai_path = os.path.join(self.im_path, ai_folder)
                    if not os.path.isdir(ai_path) or not ai_folder.startswith('ai_'):
                        continue
                    
                    images_path = os.path.join(ai_path, 'images')
                    if not os.path.isdir(images_path):
                        continue
                    
                    for scene_folder in os.listdir(images_path):
                        scene_path = os.path.join(images_path, scene_folder)
                        if not os.path.isdir(scene_path) or not scene_folder.startswith('scene_cam_'):
                            continue
                        
                        for file in os.listdir(scene_path):
                            if file.startswith('frame.') and file.endswith('_dequantize.exr'):
                                image_files.append(os.path.join(scene_path, file))
                
                print(f"Found {len(image_files)} image files with pattern 'frame.*_dequantize.exr'")
                
                if len(image_files) == 0:
                    print("No image files found")
                    self.image_files = []
                    return
                
                # Extract info from image paths EXACTLY as in ldr_to_sh_Dataset
                extracted_info = []
                for img_path in image_files:
                    parts = os.path.normpath(img_path).split(os.sep)
                    
                    ai_folder = None
                    scene_folder = None
                    frame_num = None
                    
                    for part in parts:
                        if part.startswith('ai_'):
                            ai_folder = part
                        elif part.startswith('scene_cam_'):
                            scene_folder = part
                    
                    filename = os.path.basename(img_path)
                    match = re.search(r'frame\.(\d+)\_', filename)
                    if match:
                        frame_num = match.group(1)
                    
                    extracted_info.append({
                        'path': img_path,
                        'ai_folder': ai_folder,
                        'scene_folder': scene_folder,
                        'frame_num': frame_num
                    })
                
                # Try metadata matching first (same as ldr_to_sh_Dataset)
                if 'ai_folders' in metadata and 'scene_folders' in metadata and 'frame_nums' in metadata:
                    print("Attempting to match with ai_folders, scene_folders, and frame_nums metadata")
                    
                    matched_indices = []
                    matched_image_paths = []
                    
                    for info in extracted_info:
                        for i in range(len(metadata['ai_folders'])):
                            if (info['ai_folder'] == metadata['ai_folders'][i] and
                                info['scene_folder'] == metadata['scene_folders'][i] and
                                info['frame_num'] == metadata['frame_nums'][i]):
                                matched_indices.append(i)
                                matched_image_paths.append(info['path'])
                                break
                    
                    print(f"Matched {len(matched_indices)} images with metadata")
                    
                    if len(matched_indices) > 0:
                        # Use the SAME ORDER as the matched indices
                        self.image_files = matched_image_paths
                        
                        # Apply train/val split EXACTLY as in ldr_to_sh_Dataset
                        if split == 'train':
                            split_idx = int(len(self.image_files) * 0.9)
                            self.image_files = self.image_files[:split_idx]
                        else:
                            split_idx = int(len(self.image_files) * 0.9)
                            self.image_files = self.image_files[split_idx:]
                        
                        print(f"Loaded {len(self.image_files)} matched images for {split}")
                        return
                
                # Fallback to sequential approach (same as ldr_to_sh_Dataset)
                print("No matches found with metadata, using sequential approach")
                
                # Sort EXACTLY as in ldr_to_sh_Dataset
                sorted_images = sorted(extracted_info, key=lambda x: (
                    x['ai_folder'] or '',
                    x['scene_folder'] or '',
                    x['frame_num'] or ''
                ))
                
                sorted_image_paths = [info['path'] for info in sorted_images]
                
                # Use min of latent count and image count (same as ldr_to_sh_Dataset)
                count = min(len(latents), len(sorted_image_paths))
                
                if count == 0:
                    raise ValueError("No data available - either no latents or no images were found")
                
                # Get subset (same as ldr_to_sh_Dataset)
                images_subset = sorted_image_paths[:count]
                
                # Apply train/val split EXACTLY as in ldr_to_sh_Dataset
                if split == 'train':
                    split_idx = int(count * 0.9)
                    self.image_files = images_subset[:split_idx]
                else:
                    split_idx = int(count * 0.9)
                    self.image_files = images_subset[split_idx:]
                
                print(f"Loaded {len(self.image_files)} images for {split} using sequential matching")
                
        except Exception as e:
            print(f"Error loading with H5 reference: {e}")
            self._load_with_default_ordering(split)
    
    def _load_with_default_ordering(self, split):
        """Fallback method when H5 is not available"""
        all_image_files = []
        
        # Same nested structure traversal
        for ai_folder in os.listdir(self.im_path):
            ai_path = os.path.join(self.im_path, ai_folder)
            if not os.path.isdir(ai_path) or not ai_folder.startswith('ai_'):
                continue
            
            images_path = os.path.join(ai_path, 'images')
            if not os.path.isdir(images_path):
                continue
            
            for scene_folder in os.listdir(images_path):
                scene_path = os.path.join(images_path, scene_folder)
                if not os.path.isdir(scene_path) or not scene_folder.startswith('scene_cam_'):
                    continue
                
                for file in os.listdir(scene_path):
                    if file.startswith('frame.') and file.endswith(self.file_suffix):
                        all_image_files.append(os.path.join(scene_path, file))
        
        # Sort by path to ensure consistent ordering
        all_image_files.sort()
        
        # Apply train/val split
        if split == 'train':
            split_idx = int(len(all_image_files) * 0.9)
            self.image_files = all_image_files[:split_idx]
        else:
            split_idx = int(len(all_image_files) * 0.9)
            self.image_files = all_image_files[split_idx:]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        if img_path.endswith('.exr'):
            try:
                if OpenEXR is None:
                    raise ImportError("OpenEXR not available")
                
                # Open the input file
                exr_file = OpenEXR.InputFile(img_path)
                
                # Get the header and extract data window dimensions
                header = exr_file.header()
                dw = header['dataWindow']
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1
                
                # Read the RGB channels
                FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                r_str = exr_file.channel('R', FLOAT)
                g_str = exr_file.channel('G', FLOAT)
                b_str = exr_file.channel('B', FLOAT)
                
                # Convert to numpy arrays
                r = np.array(array.array('f', r_str)).reshape(height, width)
                g = np.array(array.array('f', g_str)).reshape(height, width)
                b = np.array(array.array('f', b_str)).reshape(height, width)
                
                # Create RGB image and handle -1 values
                rgb = np.stack([r, g, b], axis=2)
                
                # Transform from [0,∞) to [-1,1] range (assuming original is in [0,1] range)
                rgb = (2*rgb) - 1  # Convert to [-1, 1] range
                
                # Convert directly to PyTorch tensor, preserving float values
                img_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()  # Convert to CxHxW format
                
                return img_tensor
                
            except (ImportError, Exception) as e:
                # Fallback method if OpenEXR is not available or other error
                print(f"Warning: Could not load EXR file {img_path}: {e}")
                # Return a zero tensor with the right dimensions
                return torch.zeros(3, self.im_size, self.im_size)
        else:
            # For standard images, use the regular PIL pipeline
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img            