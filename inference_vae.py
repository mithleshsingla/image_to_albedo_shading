import torch
import yaml
import argparse
import os
import re
import numpy as np
from tqdm import tqdm
from vae import VAE
from PIL import Image
import h5py
from torch.utils.data import DataLoader, Dataset
import pickle
import OpenEXR
import Imath
import array
import numpy as np
from scipy.ndimage import zoom
            

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
                            'frame_num': frame_num,
                            'full_path': os.path.join(scene_path, file),
                            'relative_path': os.path.join(ai_folder, 'images', scene_folder, file)
                        })
        
        
    def __len__(self):
        return len(self.image_files)
    
    def get_scene_info(self, idx):
        """Return scene information for matching across datasets"""
        return self.scene_info[idx]
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_info = self.scene_info[idx]
        
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
                
                rgb=2*rgb-1
                
                # Convert directly to PyTorch tensor, preserving float values
                img_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()  # Convert to CxHxW format
                
                
                return img_tensor, img_info
                
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
            return img, img_info


def extract_latents(args):
    # Read the config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    print("Loading configuration...")
    print(config)
    
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create the VAE model
    print("Creating VAE model...")
    model = VAE(latent_dim=16).to(device)

    # Load the trained model weights
    checkpoint_path = "checkpoints/epoch_41_best_autoencoder_model_checkpoint.pth"
    print(f"Loading model weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
        
    #model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Create dataset and dataloader for inference
    print("Creating dataset...")
    if args.inference_path:
        inference_path = args.inference_path
    else:
        inference_path = dataset_config['im_path']
    
    dataset = ImageDataset(im_path=dataset_config['im_path'],
                               im_size=dataset_config['im_size'], file_suffix='diffuse_reflectance.exr')
    
    #HDRGrayscaleEXRDataset(im_path=inference_path, im_size=dataset_config['im_size'])
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting inference...")
    
    # Choose your preferred storage format
    if args.format == 'hdf5':
        # Use HDF5 format for large datasets
        h5_file = h5py.File(os.path.join(output_dir, 'latent_vectors.h5'), 'w')
        latent_dset = None
        
        # Create dataset for metadata
        metadata_group = h5_file.create_group('metadata')
        ai_folders_dset = metadata_group.create_dataset('ai_folders', 
                                                      shape=(len(dataset),), 
                                                      dtype=h5py.special_dtype(vlen=str))
        scene_folders_dset = metadata_group.create_dataset('scene_folders', 
                                                         shape=(len(dataset),), 
                                                         dtype=h5py.special_dtype(vlen=str))
        frame_nums_dset = metadata_group.create_dataset('frame_nums', 
                                                      shape=(len(dataset),), 
                                                      dtype=h5py.special_dtype(vlen=str))
        full_paths_dset = metadata_group.create_dataset('full_paths', 
                                                      shape=(len(dataset),), 
                                                      dtype=h5py.special_dtype(vlen=str))
        relative_paths_dset = metadata_group.create_dataset('relative_paths', 
                                                         shape=(len(dataset),), 
                                                         dtype=h5py.special_dtype(vlen=str))
    else:
        # Use dictionary for smaller datasets
        latent_vectors = {}
        latent_metadata = {}
    
    total_processed = 0

    with torch.no_grad():
        for batch_idx, (images, img_infos) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            
            # Get latent vectors from encoder
            _ ,_ ,z = model(images)
            print(f"Batch {batch_idx}: Latent shape: {z.shape}")
            
            # Move to CPU and convert to numpy for storage
            z_cpu = z.cpu().numpy()
            
            if args.format == 'hdf5':
                if latent_dset is None:
                    latent_shape = (len(dataset), *z_cpu.shape[1:])
                    latent_dset = h5_file.create_dataset('latents', shape=latent_shape, dtype='float32')
                    print(f"Created latent dataset with shape: {latent_shape}")
                
                # Store batch data
                batch_size = z_cpu.shape[0]
                start_idx = total_processed
                end_idx = start_idx + batch_size
                
                if end_idx > len(dataset):
                    end_idx = len(dataset)
                    batch_size = end_idx - start_idx
                    z_cpu = z_cpu[:batch_size]
                
                latent_dset[start_idx:end_idx] = z_cpu
                
                # Handle metadata - img_infos is a list of dictionaries
                for i in range(batch_size):
                    idx = start_idx + i
                    if idx < len(dataset):
                        img_info = {
                            'ai_folder': img_infos['ai_folder'][i],
                            'scene_folder': img_infos['scene_folder'][i], 
                            'frame_num': img_infos['frame_num'][i],
                            'full_path': img_infos['full_path'][i],
                            'relative_path': img_infos['relative_path'][i]
                        }
                        ai_folders_dset[idx] = img_info['ai_folder']
                        scene_folders_dset[idx] = img_info['scene_folder']
                        frame_nums_dset[idx] = img_info['frame_num'] if img_info['frame_num'] else ''
                        full_paths_dset[idx] = img_info['full_path']
                        relative_paths_dset[idx] = img_info['relative_path']
                
                total_processed += batch_size
                
                if total_processed >= len(dataset):
                    break
            else:

                # Store in dictionary with hierarchical structure
                for i, img_info in enumerate(img_infos):
                    # Create a unique key based on the relative path
                    key = img_info['relative_path']
                    latent_vectors[key] = z_cpu[i]
                    latent_metadata[key] = {
                        'ai_folder': img_info['ai_folder'],
                        'scene_folder': img_info['scene_folder'],
                        'frame_num': img_info['frame_num'],
                        'full_path': img_info['full_path']
                    }
            
            
    # Save latent vectors
    if args.format == 'hdf5':
        h5_file.close()
        print(f"Saved latent vectors to {os.path.join(output_dir, 'latent_vectors.h5')}")
    else:
        # Save hierarchical structure
        with open(os.path.join(output_dir, 'latent_vectors.pkl'), 'wb') as f:
            pickle.dump(latent_vectors, f)
        with open(os.path.join(output_dir, 'latent_metadata.pkl'), 'wb') as f:
            pickle.dump(latent_metadata, f)
        print(f"Saved latent vectors to {os.path.join(output_dir, 'latent_vectors.pkl')}")
        print(f"Saved metadata to {os.path.join(output_dir, 'latent_metadata.pkl')}")
    
    # Save example latent vector dimensions
    with open(os.path.join(output_dir, 'latent_info.txt'), 'w') as f:
        f.write(f"Latent dimensions: {z_cpu.shape[1:]}\n")
        f.write(f"Number of samples: {len(dataset)}\n")
        f.write(f"Directory structure: {inference_path}/ai_XXX_XXX/images/scene_cam_XX_final_hdf5/frame.XXXX.diffuse_reflectance.exr\n")
    
    print("Inference completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Inference')
    parser.add_argument('--config', dest='config_path', default='config/autoen_alb.yaml', type=str,
                        help='Path to the configuration file')
    parser.add_argument('--inference_path', type=str, default=None,
                        help='Path to inference images (defaults to training dataset path)')
    parser.add_argument('--output_dir', type=str, default='latent_output',
                        help='Directory to save latent vectors')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--format', type=str, choices=['hdf5', 'pickle'], default='hdf5',
                        help='Storage format for latent vectors')
    
    # Use parse_known_args to ignore unrecognized arguments
    import sys
    args, unknown = parser.parse_known_args(sys.argv[1:])
    extract_latents(args)