
from torchvision import transforms
from torch.utils.data import Dataset
import os
import glob
import h5py
import torch
from PIL import Image
from tqdm import tqdm
import re
import numpy as np
import OpenEXR
import Imath
import array
import numpy as np
                    

class ldr_to_sh_Dataset(Dataset):
    def __init__(self, split='train', im_path=None, im_size=8, im_channels=8, use_latents=True, latent_path='latent_output/latent_vectors.h5'):
        self.im_size = im_size
        self.im_channels = im_channels
        self.split = split
        self.use_latents = use_latents
        self.images = []
        self.latents = None
        self.filenames = None
        self.im_path = im_path  # Store the image path
        
        # Setup transform for regular images
        self.transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))  # Normalize to [-1, 1]
        ])
        
        print(f"Using latent path: {latent_path}")
        if use_latents:
            if latent_path.endswith('.h5'):
                # Load latent vectors from H5 file
                print(f"Loading latent vectors from {latent_path}")
                self._load_latents_from_h5(latent_path)
            else:
                print(f"Loading latents from directory: {latent_path}")
                # Directory-based latent loading
                self._load_latents_from_directory(latent_path)
        else:
            print(f"Loading regular images from {im_path}")
            # Load regular images
            self._load_regular_images(im_path)
            
        # Check if we have valid data
        if len(self.images) == 0 and (self.latents is None or len(self.latents) == 0):
            raise ValueError(f"No data was loaded! Check your paths and H5 file structure.")

        print(f"Total dataset size for {split}: {len(self)}")
        
    def _load_latents_from_h5(self, h5_path):
        """Load latent vectors from an H5 file with structured paths"""
        try:
            with h5py.File(h5_path, 'r') as f:
                print("Keys in H5 file:", list(f.keys()))
                
                # Load latents
                latents = f['latents'][:]
                print(f"Found {len(latents)} latent vectors in H5 file with shape {latents.shape}")
                
                # Extract metadata
                metadata = {}
                if 'metadata' in f:
                    metadata_group = f['metadata']
                    print("Metadata is a group with keys:", list(metadata_group.keys()))
                    
                    # Extract all metadata fields
                    for key in metadata_group.keys():
                        if isinstance(metadata_group[key], h5py.Dataset):
                            data = metadata_group[key][:]
                            # Convert bytes to strings if needed
                            if data.dtype.kind == 'O' and isinstance(data[0], bytes):
                                data = [item.decode('utf-8') for item in data]
                            metadata[key] = data
                            print(f"Loaded metadata/{key} with {len(data)} entries")
                
                # Check if image path exists
                if self.im_path is None or not os.path.exists(self.im_path):
                    print(f"Image path {self.im_path} does not exist. Using only latent vectors.")
                    # Use only latent vectors without image paths
                    self.latents = torch.from_numpy(latents).float()
                    # Split into train/val
                    if self.split == 'train':
                        split_idx = int(len(self.latents) * 0.9)
                        self.latents = self.latents[:split_idx]
                    else:
                        split_idx = int(len(self.latents) * 0.9)
                        self.latents = self.latents[split_idx:]
                    
                    # Set dummy image paths (just for length calculation)
                    self.images = [f"dummy_path_{i}" for i in range(len(self.latents))]
                    
                    print(f"Loaded {len(self.latents)} latent vectors for {self.split} split")
                    return
                
                # Load all images from path with correct nested structure
                print(f"Loading images from {self.im_path} with correct nested structure")
                
                # Walk through the nested directories with the correct structure
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
                            # Match the correct pattern: frame.XXXX_dequantize.exr
                            if file.startswith('frame.') and file.endswith('_dequantize.exr'):
                                image_files.append(os.path.join(scene_path, file))
                
                print(f"Found {len(image_files)} image files with pattern 'frame.*_dequantize.exr'")
                
                # Handle empty image files case
                if len(image_files) == 0:
                    print("No image files found - using only latent vectors")
                    self.latents = torch.from_numpy(latents).float()
                    # Split into train/val
                    if self.split == 'train':
                        split_idx = int(len(self.latents) * 0.9)
                        self.latents = self.latents[:split_idx]
                    else:
                        split_idx = int(len(self.latents) * 0.9)
                        self.latents = self.latents[split_idx:]
                    
                    # Set dummy image paths (just for length calculation)
                    self.images = [f"dummy_path_{i}" for i in range(len(self.latents))]
                    
                    print(f"Loaded {len(self.latents)} latent vectors for {self.split} split")
                    return
                
                # Extract relevant parts from image paths for matching
                extracted_info = []
                for img_path in image_files:
                    # Extract components from path
                    parts = os.path.normpath(img_path).split(os.sep)
                    
                    # Find ai_folder and scene_folder
                    ai_folder = None
                    scene_folder = None
                    frame_num = None
                    
                    for part in parts:
                        if part.startswith('ai_'):
                            ai_folder = part
                        elif part.startswith('scene_cam_'):
                            scene_folder = part
                    
                    # Extract frame number - match the pattern for dequantize.exr files
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
                
                # Try to match with metadata
                if 'ai_folders' in metadata and 'scene_folders' in metadata and 'frame_nums' in metadata:
                    print("Attempting to match with ai_folders, scene_folders, and frame_nums metadata")
                    
                    matched_indices = []
                    matched_image_paths = []
                    
                    # For each image, try to find matching metadata
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
                        # Use matched indices
                        matched_latents = latents[matched_indices]
                        self.images = matched_image_paths
                        self.latents = torch.from_numpy(matched_latents).float()
                        
                        # Split into train/val
                        if self.split == 'train':
                            split_idx = int(len(self.latents) * 0.9)
                            self.latents = self.latents[:split_idx]
                            self.images = self.images[:split_idx]
                        else:
                            split_idx = int(len(self.latents) * 0.9)
                            self.latents = self.latents[split_idx:]
                            self.images = self.images[split_idx:]
                        
                        print(f"Loaded {len(self.latents)} matched latent vectors for {self.split}")
                        return
                
                # If no matches were found, use a sequential approach
                print("No matches found with metadata, using sequential approach")
                
                # Sort images by ai_folder, scene_folder, and frame_num
                sorted_images = sorted(extracted_info, key=lambda x: (
                    x['ai_folder'] or '',
                    x['scene_folder'] or '',
                    x['frame_num'] or ''
                ))
                
                # Get sorted image paths
                sorted_image_paths = [info['path'] for info in sorted_images]
                
                # Use min of latent count and image count
                count = min(len(latents), len(sorted_image_paths))
                
                # Ensure we have at least one item
                if count == 0:
                    raise ValueError("No data available - either no latents or no images were found")
                
                # Get subset of latents and images
                latents_subset = latents[:count]
                images_subset = sorted_image_paths[:count]
                
                # Split into train/val
                if self.split == 'train':
                    split_idx = int(count * 0.9)
                    self.latents = torch.from_numpy(latents_subset[:split_idx]).float()
                    self.images = images_subset[:split_idx]
                else:
                    split_idx = int(count * 0.9)
                    self.latents = torch.from_numpy(latents_subset[split_idx:]).float()
                    self.images = images_subset[split_idx:]
                
                print(f"Loaded {len(self.latents)} latent vectors for {self.split} using sequential matching")
                
        except Exception as e:
            print(f"Error loading H5 file: {e}")
            # Print H5 file structure for debugging
            try:
                with h5py.File(h5_path, 'r') as f:
                    print("Keys in H5 file:", list(f.keys()))
                    
                    # Print structure of the file for debugging
                    def print_attrs(name, obj):
                        print(f"Found object: {name}, Type: {type(obj)}")
                        if isinstance(obj, h5py.Dataset):
                            print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")
                            # Print a small sample if it's not too large
                            if len(obj.shape) > 0 and obj.shape[0] > 0 and obj.shape[0] < 10:
                                try:
                                    sample = obj[0]
                                    if isinstance(sample, bytes):
                                        sample = sample.decode('utf-8')
                                    print(f"  Sample: {sample}")
                                except:
                                    pass
                    
                    # Recursively explore the H5 file structure
                    f.visititems(print_attrs)
                    
            except Exception as inner_e:
                print(f"Could not print H5 structure: {inner_e}")
            raise
    
    # Add this method to ldr_to_sh_Dataset class (for latents)
    def get_scene_info(self, idx):
        """Extract scene info - for latents we use the corresponding image path"""
        if hasattr(self, 'images') and idx < len(self.images):
            img_path = self.images[idx]
            if img_path.startswith('dummy_path_'):
                # For dummy paths, we need to get info from metadata
                return self._get_scene_info_from_metadata(idx)
            
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
        
        return {'ai_folder': None, 'scene_folder': None, 'frame_num': None, 'path': None}

    def _load_latents_from_directory(self, latent_path):
        """Original method to load latents from directory structure"""
        if not os.path.exists(latent_path):
            os.makedirs(latent_path)
            print(f"Created directory {latent_path} for latents")
            self.use_latents = False
            return
        
        for d_name in tqdm(os.listdir(latent_path)):
            fnames = glob.glob(os.path.join(latent_path, d_name, '*.{}'.format('pt')))
            for fname in fnames:
                self.images.append(fname)
        
    def _load_regular_images(self, im_path):
        """Modified method to load regular images from the nested structure"""
        self.images = []
        
        # Walk through the nested directories with the correct structure
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
                    # Match the correct pattern: frame.XXXX_dequantize.exr
                    if file.startswith('frame.') and file.endswith('_dequantize.exr'):
                        self.images.append(os.path.join(scene_path, file))
        
        print(f"Found {len(self.images)} images with pattern 'frame.*_dequantize.exr' in the nested structure")
        
        # Split into train/val if we have images
        if len(self.images) > 0:
            if self.split == 'train':
                split_idx = int(len(self.images) * 0.9)
                self.images = self.images[:split_idx]
            else:
                split_idx = int(len(self.images) * 0.9)
                self.images = self.images[split_idx:]
            
            print(f"After splitting, using {len(self.images)} images for {self.split}")
    
    def __len__(self):
        if self.latents is not None:
            return len(self.latents)
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.use_latents:
            if self.latents is not None:
                # Return pre-loaded latent from H5 file
                return self.latents[idx]
            else:
                # Load latent from file path
                return torch.load(self.images[idx])
        else:
            # Load and transform regular image
            img_path = self.images[idx]
            
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
                    channels = header['channels']
                    
                    # Determine if this is an RGB or grayscale image
                    if 'R' in channels and 'G' in channels and 'B' in channels:
                        r_str = exr_file.channel('R', FLOAT)
                        g_str = exr_file.channel('G', FLOAT)
                        b_str = exr_file.channel('B', FLOAT)
                        
                        # Convert to numpy arrays
                        r = np.array(array.array('f', r_str)).reshape(height, width)
                        g = np.array(array.array('f', g_str)).reshape(height, width)
                        b = np.array(array.array('f', b_str)).reshape(height, width)
                        
                        # Create RGB image
                        img_data = np.stack([r, g, b], axis=2)
                    else:
                        # Assume it's a grayscale image
                        key = list(channels.keys())[0]  # Get the first channel name
                        data_str = exr_file.channel(key, FLOAT)
                        img_data = np.array(array.array('f', data_str)).reshape(height, width)
                        img_data = np.stack([img_data] * 3, axis=2)  # Convert to 3 channels
                    
                    # Handle negative values and normalize to [0,1]
                    img_data = np.maximum(img_data, 0.0)  # Clamp negative values
                    img_data = np.minimum(img_data, 1.0)  # Clamp values above 1.0
                    
                    # Convert to tensor
                    if img_data.ndim == 2:
                        img_tensor = torch.from_numpy(img_data).float().unsqueeze(0)  # Add channel dim for grayscale
                    else:
                        img_tensor = torch.from_numpy(img_data.transpose(2, 0, 1)).float()  # Channels first for RGB
                    
                    # Resize if needed
                    if img_tensor.shape[1] != self.im_size or img_tensor.shape[2] != self.im_size:
                        img_tensor = torch.nn.functional.interpolate(
                            img_tensor.unsqueeze(0),  # Add batch dimension
                            size=(self.im_size, self.im_size),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)  # Remove batch dimension
                    
                    # Normalize to [-1, 1]
                    img_tensor = img_tensor * 2.0 - 1.0
                    
                    return img_tensor
                    
                except Exception as e:
                    print(f"Error loading EXR file {img_path}: {e}")
                    # Return a zero tensor with the right dimensions
                    return torch.zeros(3 if self.im_channels >= 3 else 1, self.im_size, self.im_size)
            else:
                # Standard image processing with PIL
                img = Image.open(img_path).convert('RGB' if self.im_channels >= 3 else 'L')
                img = self.transform(img)
                return img            