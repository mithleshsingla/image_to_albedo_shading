import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from models.unet import Unet
from models.vae import VAE
from models.unet import Encoder
import gc
from src.dataloader_latent_hyperism import ldr_to_sh_Dataset
from src.dataloader_image_hyperism import ImageDatasetwv_h5, ImageDatasetwv_h5_al
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
import logging
import re
import torchvision.utils as vutils
# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
import lpips

from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU device index: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"GPU Capability: {torch.cuda.get_device_capability(torch.cuda.current_device())}")
else:
    print("CUDA is not available. Using CPU.")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from torch.amp import autocast, GradScaler
#scaler = GradScaler()
torch.set_float32_matmul_precision('high')
import wandb
wandb.init(project="ldr_to_al_training_latent_flow_matching")  
loss_fn_alex = lpips.LPIPS(net='vgg').to(device)
from collections import OrderedDict


# Step 2: Create the CombinedDataset class adapted for your use case
class CombinedSyncedDataset(Dataset):
    def __init__(self, latent_dataset, encoder_dataset, albedo_dataset, split='train'):
        """
        A dataset that matches corresponding images/latents across three datasets based on scene metadata.
        
        Args:
            latent_dataset: The ldr_to_sh_Dataset for latents
            encoder_dataset: The ImageDatasetwv_h5 for encoder images
            albedo_dataset: The ImageDatasetwv_h5_al for albedo images
            split: 'train' or 'val'
        """
        self.latent_dataset = latent_dataset
        self.encoder_dataset = encoder_dataset
        self.albedo_dataset = albedo_dataset
        self.split = split
        
        # Create a mapping from scene info to indices for each dataset
        self.matching_indices = self._find_matching_indices()
        
        print(f"CombinedSyncedDataset for {split}:")
        print(f"  Found {len(self.matching_indices)} matching triplets out of:")
        print(f"    {len(latent_dataset)} latent vectors")
        print(f"    {len(encoder_dataset)} encoder images") 
        print(f"    {len(albedo_dataset)} albedo images")
        
        if len(self.matching_indices) == 0:
            print("WARNING: No matching triplets found!")
    
    def _find_matching_indices(self):
        """Find matching indices across all three datasets based on scene info"""
        
        # Create dictionaries to map scene info to indices for each dataset
        latent_indices = {}
        encoder_indices = {}
        albedo_indices = {}
        
        print("Building index mappings...")
        
        # Map latent dataset
        for idx in range(len(self.latent_dataset)):
            info = self.latent_dataset.get_scene_info(idx)
            if info['ai_folder'] and info['scene_folder'] and info['frame_num']:
                key = (info['ai_folder'], info['scene_folder'], info['frame_num'])
                latent_indices[key] = idx
        
        # Map encoder dataset  
        for idx in range(len(self.encoder_dataset)):
            info = self.encoder_dataset.get_scene_info(idx)
            if info['ai_folder'] and info['scene_folder'] and info['frame_num']:
                key = (info['ai_folder'], info['scene_folder'], info['frame_num'])
                encoder_indices[key] = idx
        
        # Map albedo dataset
        for idx in range(len(self.albedo_dataset)):
            info = self.albedo_dataset.get_scene_info(idx)
            if info['ai_folder'] and info['scene_folder'] and info['frame_num']:
                key = (info['ai_folder'], info['scene_folder'], info['frame_num'])
                albedo_indices[key] = idx
        
        print(f"Index mapping sizes:")
        print(f"  Latent: {len(latent_indices)} valid entries")
        print(f"  Encoder: {len(encoder_indices)} valid entries")
        print(f"  Albedo: {len(albedo_indices)} valid entries")
        
        # Find common keys across all datasets
        latent_keys = set(latent_indices.keys())
        encoder_keys = set(encoder_indices.keys())
        albedo_keys = set(albedo_indices.keys())
        
        # Debug: check overlaps
        latent_encoder_overlap = latent_keys.intersection(encoder_keys)
        encoder_albedo_overlap = encoder_keys.intersection(albedo_keys)
        latent_albedo_overlap = latent_keys.intersection(albedo_keys)
        
        print(f"Key overlaps:")
        print(f"  Latent ∩ Encoder: {len(latent_encoder_overlap)}")
        print(f"  Encoder ∩ Albedo: {len(encoder_albedo_overlap)}")
        print(f"  Latent ∩ Albedo: {len(latent_albedo_overlap)}")
        
        common_keys = latent_keys.intersection(encoder_keys).intersection(albedo_keys)
        print(f"  All three: {len(common_keys)}")
        
        if len(common_keys) < 10:
            print("Sample common keys:", list(common_keys)[:5])
            print("Sample latent keys:", list(latent_keys)[:5])
            print("Sample encoder keys:", list(encoder_keys)[:5])
            print("Sample albedo keys:", list(albedo_keys)[:5])
        
        # Create a list of matching indices, sorted for consistency
        matching_indices = [
            (latent_indices[key], encoder_indices[key], albedo_indices[key]) 
            for key in sorted(common_keys)
        ]
        
        return matching_indices
    
    def __len__(self):
        return len(self.matching_indices)
        
    def __getitem__(self, idx):
        if idx >= len(self.matching_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.matching_indices)}")
            
        # Get the matching indices for all three datasets
        latent_idx, encoder_idx, albedo_idx = self.matching_indices[idx]
        
        # Get the items from each dataset
        latent = self.latent_dataset[latent_idx]          # im
        encoder_image = self.encoder_dataset[encoder_idx] # cond_img  
        albedo_image = self.albedo_dataset[albedo_idx]    # albedo_im
        
        # Return as tuple to match your existing training loop
        return latent, encoder_image, albedo_image
        
    # Optional: Keep the detailed version as a separate method
    def get_detailed_item(self, idx):
        """Get item with full details including scene info and indices"""
        if idx >= len(self.matching_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.matching_indices)}")
            
        # Get the matching indices for all three datasets
        latent_idx, encoder_idx, albedo_idx = self.matching_indices[idx]
        
        # Get the items from each dataset
        latent = self.latent_dataset[latent_idx]
        encoder_image = self.encoder_dataset[encoder_idx] 
        albedo_image = self.albedo_dataset[albedo_idx]
        
        # Get scene info for debugging/verification
        info = self.encoder_dataset.get_scene_info(encoder_idx)
        
        return {
            'latent': latent,
            'encoder_image': encoder_image,
            'albedo_image': albedo_image,
            'scene_info': info,
            'indices': (latent_idx, encoder_idx, albedo_idx)
        }

    # Also update the verify_sync method to use get_detailed_item:
    def verify_sync(self, num_samples=5):
        """Verify that the datasets are properly synchronized"""
        print(f"\n=== VERIFYING SYNCHRONIZATION ({num_samples} samples) ===")
        
        for i in range(min(num_samples, len(self))):
            item = self.get_detailed_item(i)  # Use detailed version for verification
            scene_info = item['scene_info']
            indices = item['indices']
            
            # Get scene info from all three datasets
            latent_info = self.latent_dataset.get_scene_info(indices[0])
            encoder_info = self.encoder_dataset.get_scene_info(indices[1])
            albedo_info = self.albedo_dataset.get_scene_info(indices[2])
            
            print(f"\nSample {i}:")
            print(f"  Latent  [{indices[0]:5}]: ai={latent_info['ai_folder']}, scene={latent_info['scene_folder']}, frame={latent_info['frame_num']}")
            print(f"  Encoder [{indices[1]:5}]: ai={encoder_info['ai_folder']}, scene={encoder_info['scene_folder']}, frame={encoder_info['frame_num']}")
            print(f"  Albedo  [{indices[2]:5}]: ai={albedo_info['ai_folder']}, scene={albedo_info['scene_folder']}, frame={albedo_info['frame_num']}")
            
            # Check if they match
            match = (latent_info['frame_num'] == encoder_info['frame_num'] == albedo_info['frame_num'] and
                    latent_info['ai_folder'] == encoder_info['ai_folder'] == albedo_info['ai_folder'] and
                    latent_info['scene_folder'] == encoder_info['scene_folder'] == albedo_info['scene_folder'])
            
            print(f"  Match: {'✓' if match else '✗'}")

def convert_state_dict(old_state_dict):
    new_state_dict = {}
    for k, v in old_state_dict.items():
        if k.startswith("decoder.model"):
            parts = k.split(".")
            layer_idx = int(parts[2])
            subkey = ".".join(parts[3:])
            # Map old indices to new deconv blocks
            if layer_idx in [0, 1, 2]:
                new_key = f"decoder.deconv1.{layer_idx}.{subkey}"
            elif layer_idx in [3, 4, 5]:
                new_key = f"decoder.deconv2.{layer_idx-3}.{subkey}"
            elif layer_idx in [6, 7, 8]:
                new_key = f"decoder.deconv3.{layer_idx-6}.{subkey}"
            elif layer_idx in [9, 10, 11]:
                new_key = f"decoder.deconv4.{layer_idx-9}.{subkey}"
            elif layer_idx in [12, 13]:
                new_key = f"decoder.deconv5.{layer_idx-12}.{subkey}"
            else:
                continue  # Skip unknown
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def strip_orig_mod(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
        new_state_dict[new_key] = v
    return new_state_dict


def get_time_discretization(nfes: int, rho=7):
    step_indices = torch.arange(nfes, dtype=torch.float64)
    sigma_min = 0.002
    sigma_max = 80.0
    sigma_vec = (
        sigma_max ** (1 / rho)
        + step_indices / (nfes - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigma_vec = torch.cat([sigma_vec, torch.zeros_like(sigma_vec[:1])])
    time_vec = (sigma_vec / (1 + sigma_vec)).squeeze()
    t_samples = 1.0 - torch.clip(time_vec, min=0.0, max=1.0)
    return t_samples


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

# Add this after creating all datasets, before starting the training loop
def check_dataset_sync(im_dataset, im_dataset_encoder,im_albedo):
    """Check if the two datasets are synchronized"""
    # Skip if using latents directly
    if not hasattr(im_dataset, 'images') or not isinstance(im_dataset.images, list) or not im_dataset.images:
        print("Skipping sync check - dataset doesn't have image list")
        return
    
    if not hasattr(im_dataset_encoder, 'image_files') or not im_dataset_encoder.image_files:
        print("Skipping sync check - encoder dataset doesn't have image_files list")
        return

    if not hasattr(im_albedo, 'image_files') or not im_albedo.image_files:
        print("Skipping sync check - albedo dataset doesn't have image_files list")
        return
    
    # Get basenames for comparison
    im_basenames = [os.path.basename(str(path)) for path in im_dataset.images[:5]]
    encoder_basenames = [os.path.basename(str(path)) for path in im_dataset_encoder.image_files[:5]]
    albedo_basenames = [os.path.basename(str(path)) for path in im_albedo.image_files[:5]]
    
    print("First 5 images in latent dataset:", im_basenames)
    print("First 5 images in encoder dataset:", encoder_basenames)
    print("First 5 images in albedo dataset:", albedo_basenames)
    # Check if they follow the same pattern (might not be exactly the same files)
    im_pattern = re.search(r'(ai_\d+_\d+|scene_cam_\d+)', str(im_dataset.images[0]) if im_dataset.images else "")
    encoder_pattern = re.search(r'(ai_\d+_\d+|scene_cam_\d+)', str(im_dataset_encoder.image_files[0]) if im_dataset_encoder.image_files else "")
    albedo_pattern = re.search(r'(ai_\d+_\d+|scene_cam_\d+)', str(im_albedo.image_files[0]) if im_albedo.image_files else "")

    if im_pattern and encoder_pattern and albedo_pattern and im_pattern.group(1) != encoder_pattern.group(1) and im_pattern.group(1) != albedo_pattern.group(1):
        print(f"WARNING: Datasets might be using different file patterns: {im_pattern.group(1)} vs {encoder_pattern.group(1)}")

def save_training_samples(output, image, gt_image, train_config, step_count, img_save_count, guidance_scale=None, suffix=""):
    """
    Save training samples with support for different guidance scales and directories
    
    Args:
        output: Generated samples
        image: Original/albedo images
        gt_image: Ground truth images
        train_config: Training configuration
        step_count: Current training step
        img_save_count: Image save counter
        guidance_scale: CFG guidance scale (for directory naming)
        suffix: Additional suffix for filename
    """
    
    sample_size = min(8, output.shape[0])
    gt_image = gt_image[:sample_size].detach().cpu()
    save_output = output[:sample_size].detach().cpu()
    albedo_image = image[:sample_size].detach().cpu()

    # Base save path
    #base_save_path = os.path.join('/home/project/mithlesh/Hyperism', train_config['task_name'], 'Flow_samples')
    base_save_path = os.path.join('/home/project/dataset/Hyperism', train_config['task_name'], 'Flow_samples')

    # Create guidance-scale specific directory if guidance_scale is provided
    if guidance_scale is not None:
        guidance_dir = f"guidance_{guidance_scale}"
        save_path = os.path.join(base_save_path, guidance_dir)
    else:
        save_path = base_save_path
    
    os.makedirs(save_path, exist_ok=True)
    
    # Create collage
    collage = torch.cat([save_output, albedo_image, gt_image], dim=0)
    
    # Construct filename with suffix
    filename = f"{step_count}{suffix}.png"
    output_path = os.path.join(save_path, filename)
    
    vutils.save_image(collage, output_path, nrow=4, normalize=True)
    
    return img_save_count + 1

def train(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    #print(config)
    ########################
    dataset_albedo = config['albedo_params']
    dataset_config = config['dataset_params_input']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Add image saving configuration
    image_save_steps = train_config['autoencoder_img_save_steps']
    img_save_count = 0
    step_count = 0
    
    # Check if latent path is an H5 file
    latent_path = train_config['vae_latent_dir_name']
    use_h5_latents = latent_path.endswith('.h5')
    
    print(f"Using latent path: {latent_path}")
    print(f"Use H5 latents: {use_h5_latents}")
    print(f"train_config: {train_config['im_size_lt']}")
    print(f"Image save steps: {image_save_steps}")
    
    im_dataset_cls = {
        'ldr_to_sh_flow': ldr_to_sh_Dataset,
    }.get(dataset_config['name'])
    
    # Initialize models
    encoder = Encoder(im_channels=dataset_config['im_channels']).to(device)
    encoder.train()

    # Instantiate the Unet model
    model = Unet(im_channels=autoencoder_model_config['z_channels']).to(device)
    model.train()
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count parameters
    encoder_params = count_parameters(encoder)
    model_params = count_parameters(model)

    # Print the results
    print(f"Encoder Parameters: {encoder_params:,}")
    print(f"Unet Model Parameters: {model_params:,}")
    print(f"Total Parameters: {encoder_params + model_params:,}")

    # Load VAE - Now we always load it for decoder functionality
    print('Loading VAE model for decoder functionality')
    vae = VAE(latent_dim=16).to(device)
    vae.eval()
  
    # Load vae if found
    vae_path = "checkpoints\vae_model.pth"
    if os.path.exists(vae_path):
        print(f'Loading VAE checkpoint from {vae_path}')
        checkpoint_vae = torch.load(vae_path, weights_only=False, map_location=device)
        #checkpoint_vae['model_state_dict'] = convert_state_dict(checkpoint_vae['model_state_dict'])
        vae.load_state_dict(checkpoint_vae['model_state_dict'])
        print('VAE loaded successfully')
    else:
        print(f'Warning: VAE checkpoint not found at {vae_path}. Decoder visualization will not work properly.')

    # Freeze VAE parameters since we're only using it for visualization
    for param in vae.parameters():
        param.requires_grad = False
    

    # Try to load dataset with latents first
    try:
        im_dataset = im_dataset_cls(split='train',
                                   im_path=dataset_config['im_path'],
                                   im_size=train_config['im_size_lt'],
                                   im_channels=dataset_config['im_channels'],
                                   use_latents=True,
                                   latent_path=latent_path)
        print('Successfully loaded dataset with latents')
    except Exception as e:
        print(f"Error loading dataset with latents: {e}")
        print("Falling back to regular images")
        im_dataset = im_dataset_cls(split='train',
                                   im_path=dataset_config['im_path'],
                                   im_size=train_config['im_size_lt'],
                                   im_channels=dataset_config['im_channels'],
                                   use_latents=False)
    
    # Create validation dataset
    try:
        val_dataset = im_dataset_cls(split='val',
                                    im_path=dataset_config['im_path'],
                                    im_size=train_config['im_size_lt'],
                                    im_channels=dataset_config['im_channels'],
                                    use_latents=True,
                                    latent_path=latent_path)
        print('Successfully loaded validation dataset with latents')
    except Exception as e:
        print(f"Error loading validation dataset with latents: {e}")
        print("Falling back to regular images for validation")
        val_dataset = im_dataset_cls(split='val',
                                    im_path=dataset_config['im_path'],
                                    im_size=train_config['im_size_lt'],
                                    im_channels=dataset_config['im_channels'],
                                    use_latents=False)
    
    # Check if we have any data
    if len(im_dataset) == 0:
        raise ValueError("Training dataset is empty! Check your data paths.")
    
    if len(val_dataset) == 0:
        print("Warning: Validation dataset is empty. Will skip validation.")
    
    albedo_dataset = ImageDatasetwv_h5_al(im_path=dataset_albedo['im_path'],
                               im_size=dataset_config['im_size'], file_suffix='.diffuse_reflectance.exr',split='train',
                                    latent_path=latent_path)
    
    val_albedo_dataset = ImageDatasetwv_h5_al(im_path=dataset_albedo['im_path'],
                               im_size=dataset_config['im_size'], file_suffix='.diffuse_reflectance.exr',split='val',
                                    latent_path=latent_path)
    
    # Load the encoder dataset (always uses images, not latents)
    im_dataset_encoder = ImageDatasetwv_h5(im_path=dataset_config['im_path'],
                                    im_size=dataset_config['im_size'],
                                    split='train',
                                    latent_path=latent_path)

    val_dataset_encoder = ImageDatasetwv_h5(im_path=dataset_config['im_path'],
                                    im_size=dataset_config['im_size'],
                                    split='val',
                                    latent_path=latent_path)
    
    
    # Create synchronized combined datasets
    train_combined = CombinedSyncedDataset(im_dataset, im_dataset_encoder, albedo_dataset, split='train')
    val_combined = CombinedSyncedDataset(val_dataset, val_dataset_encoder, val_albedo_dataset, split='val')

    # Verify synchronization
    train_combined.verify_sync()
    val_combined.verify_sync()

    # Create data loaders from combined datasets
    data_loader = DataLoader(train_combined, batch_size=train_config['ldm_batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_combined, batch_size=train_config['ldm_batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    print(f'Found {len(im_dataset)} images/latents for training')
    print(f'Found {len(val_dataset)} images/latents for validation')
    print(f'Found {len(im_dataset_encoder)} images for encoder training')
    print(f'Found {len(val_dataset_encoder)} images for encoder validation')
    print(f'Found {len(albedo_dataset)} albedo images for training')
    print(f'Found {len(val_albedo_dataset)} albedo images for validation')
    
    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer_model = Adam(model.parameters(), lr=train_config['ldm_lr'])
    optimizer_encoder = Adam(encoder.parameters(), lr=train_config['ldm_lr'])
    
    # Add learning rate schedulers
    scheduler_model = ReduceLROnPlateau(optimizer_model, mode='min', factor=0.5, patience=5)
    scheduler_encoder = ReduceLROnPlateau(optimizer_encoder, mode='min', factor=0.5, patience=5)
    scaler = GradScaler()

    # Initialize variables for tracking best model
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # instantiate an affine path object
    path = AffineProbPath(scheduler=CondOTScheduler())
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(train_config['task_name'], exist_ok=True)

    # Try to load existing checkpoints if available
    checkpoint_path="checkpoints\unet_encoder_inference.pths"

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(strip_orig_mod(checkpoint['model_state_dict']))
            encoder.load_state_dict(strip_orig_mod(checkpoint['encoder_state_dict']))
            optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
            optimizer_encoder.load_state_dict(checkpoint['optimizer_encoder_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            step_count = checkpoint.get('step_count', 0)
            img_save_count = checkpoint.get('img_save_count', 0)
            print(f"Resuming from epoch {start_epoch} with best val loss: {best_val_loss:.4f}")
        else:
            model.load_state_dict(strip_orig_mod(checkpoint))
            print("Loaded model weights only")
            start_epoch = 0
    else:
        start_epoch = 0

    # Check if the model is already compiled
    if hasattr(torch, 'compile'):
       model = torch.compile(model)
       encoder = torch.compile(encoder)
    
    # Training loop
    for epoch_idx in range(start_epoch, num_epochs):
        losses = []
        flow_losses = []
        lpm_losses = []
        
        # Training phase
        model.train()
        encoder.train()

        for batch_idx, (im, cond_img, albedo_im) in enumerate(tqdm(data_loader)):
            step_count += 1
            
            optimizer_encoder.zero_grad()
            optimizer_model.zero_grad()
            
            im = im.float().to(device)
            cond_img = cond_img.float().to(device)
            albedo_im = albedo_im.float().to(device)
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            t = torch.rand(im.shape[0]).to(device)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                path_sample = path.sample(t=t, x_0=noise, x_1=im)
            
                # Process conditional image through encoder
                encoder_out = encoder(cond_img)
                
            
                # Calculate flow matching loss
                model_out = model(path_sample.x_t, path_sample.t, encoder_out)
                #loss = torch.nn.functional.mse_loss(model_out, path_sample.dx_t)
                # Base flow matching loss
                flow_loss = torch.nn.functional.mse_loss(model_out, path_sample.dx_t)

                # Reconstruct image from predicted and ground truth latents
                recon_pred_z = path_sample.x_t + (1.0 - t).view(-1,1,1,1) * model_out  # Predicted latent
            
                recon_pred_z = recon_pred_z.float()
                output = vae.decoder(recon_pred_z)
                lpm = (loss_fn_alex(output, albedo_im).mean())  # Ensure lpips_loss is a scalar
                # Total loss: flow + perceptual
                loss = flow_loss + train_config["perceptual_weight"] * lpm

            losses.append(loss.item())
            flow_losses.append(flow_loss.item())
            lpm_losses.append(train_config["perceptual_weight"] *lpm.item())
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer_model)
            scaler.unscale_(optimizer_encoder)

            # Add gradient clipping here
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)

            scaler.step(optimizer_model)
            scaler.step(optimizer_encoder)
            scaler.update()
            
            # Log batch loss to W&B
            wandb.log({"batch_loss": loss.item(), "step": step_count})
            
            # Image Saving Logic - Generate proper samples using velocity integration
            
            # Image Saving Logic - Generate proper samples using velocity integration
            if step_count % image_save_steps == 0 or step_count == 1:
                model.eval()
                encoder.eval()
                
                with torch.no_grad():
                    try:
                        # Create wrapped model for sampling (similar to inference code)
                        wrapped_model = WrappedModel(model, encoder)
                        solver = ODESolver(velocity_model=wrapped_model)
                        
                        # Use fewer time steps for training visualization (faster)
                        T = get_time_discretization(10, rho=5).to(device)  # Use 10 steps instead of 20 for faster training
                        
                        # Generate initial noise with same shape as latent
                        current_batch_size = cond_img.size(0)
                        #latent_channels = train_config.get('latent_channels', 4)  # Adjust based on your config
                        latent_channels = config['autoencoder_params']['z_channels']
                        latent_size = train_config['im_size_lt']
                        
                        x_init = torch.randn(current_batch_size, latent_channels, latent_size, latent_size).to(device)
                        
                        # Process condition images through encoder
                        cond_encoded = encoder(cond_img)
                        
                        # Set condition for the wrapped model
                        wrapped_model.set_condition(cond_encoded)
                        
                        # Sample from the model (integrate velocity to get actual sample)
                        samples = solver.sample(
                            time_grid=T,
                            x_init=x_init,
                            method='midpoint',
                            step_size=0.1,  # Larger step size for faster training visualization
                            return_intermediates=False
                        )
                        
                        # Now decode the proper samples using VAE decoder
                        samples = samples.float()
                        decoded_output = vae.decoder(samples)
                        #image=vae.decoder(im)
                        
                        # Save the decoded samples
                        img_save_count = save_training_samples(
                            decoded_output,albedo_im, cond_img, train_config, step_count, img_save_count
                        )
                        #print(f"Saved decoded training samples at step {step_count}")
                        
                    except Exception as e:
                        print(f"Error saving training samples: {e}")
                        # Fallback: save a simple visualization of the velocity field
                        try:
                            # Just save the raw model output for debugging
                            velocity_viz = model_out.float()
                            # You could also save this for debugging purposes
                            print(f"Saved velocity visualization at step {step_count}")
                        except:
                            print("Failed to save any visualization")
                
        # End of training epoch
        train_loss_avg = np.mean(losses)
        train_flow_avg = np.mean(flow_losses)
        train_lpm_avg = np.mean(lpm_losses)
        print('Finished epoch:{} | Average Training Loss: {:.4f}'.format(
            epoch_idx + 1, train_loss_avg))
        
        wandb.log({"epoch": epoch_idx + 1, "train_loss": train_loss_avg, "step": step_count})
        wandb.log({"epoch": epoch_idx + 1, "train_flow": train_flow_avg, "step": step_count})
        wandb.log({"epoch": epoch_idx + 1, "train_lpm": train_lpm_avg, "step": step_count})

        # Validation phase
        if len(val_dataset) > 0:
            model.eval()
            encoder.eval()
            val_losses = []
            val_flow_losses = []
            val_lpm_losses = []
            
            # Create validation iterator
            # val_iter = zip(val_loader, val_loader_encoder, val_loader_albedo)
            # val_total_batches = min(len(val_loader), len(val_loader_encoder), len(val_loader_albedo))

            with torch.no_grad():
                for val_im, val_cond_img, val_albedo_im in tqdm(val_loader, desc="Validation"):
                    val_im = val_im.float().to(device)
                    val_cond_img = val_cond_img.float().to(device)
                    val_albedo_im = val_albedo_im.float().to(device)

                    # Sample random noise
                    noise = torch.randn_like(val_im).to(device)
                    t = torch.rand(val_im.shape[0]).to(device)
                    
                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        path_sample = path.sample(t=t, x_0=noise, x_1=val_im)
                    
                        # Process conditional image through encoder
                        encoder_out = encoder(val_cond_img)
                        
                        # Calculate flow matching loss
                        model_out = model(path_sample.x_t, path_sample.t, encoder_out)
                        #val_loss = torch.nn.functional.mse_loss(model_out, path_sample.dx_t)
                        val_flow_loss = torch.nn.functional.mse_loss(model_out, path_sample.dx_t)

                        # Reconstruct image from predicted and ground truth latents
                        recon_pred_z = path_sample.x_t + (1.0 - t).view(-1,1,1,1) * model_out  # Predicted latent
                        recon_pred_z = recon_pred_z.float()
                        output = vae.decoder(recon_pred_z)
                        val_lpm = (loss_fn_alex(output, val_albedo_im).mean())  # Ensure lpips_loss is a scalar
                
                        # Total loss: flow + perceptual
                        val_loss = val_flow_loss + train_config["perceptual_weight"] * val_lpm


                    val_losses.append(val_loss.item())
                    val_flow_losses.append(val_flow_loss.item())
                    val_lpm_losses.append(train_config["perceptual_weight"] *val_lpm.item())
            
            val_loss_avg = np.mean(val_losses)
            val_flow_loss = np.mean(val_flow_losses)
            val_lpm = np.mean(val_lpm_losses)
            print(f'Validation Loss: {val_loss_avg:.4f}')
            
            # Log validation metrics to W&B
            wandb.log({"val_loss": val_loss_avg, "step": step_count})
            wandb.log({"val_flow_loss": val_flow_loss, "step": step_count})
            wandb.log({"val_lpm": val_lpm, "step": step_count})
            
            # Update learning rate schedulers
            scheduler_model.step(val_loss_avg)
            scheduler_encoder.step(val_loss_avg)
            
            # Check if model improved
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                patience_counter = 0
                
                # Save best model
                best_checkpoint_dict = {
                    'epoch': epoch_idx + 1,
                    'model_state_dict': model.state_dict(),
                    'encoder_state_dict': encoder.state_dict(),
                    'optimizer_model_state_dict': optimizer_model.state_dict(),
                    'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
                    'best_val_loss': best_val_loss,
                    'train_loss': train_loss_avg,
                    'step_count': step_count,
                    'img_save_count': img_save_count
                }
                best_checkpoint_path = os.path.join(train_config['task_name'], 'best_' + train_config['ldm_ckpt_name'])
                torch.save(best_checkpoint_dict, best_checkpoint_path)
                print(f"Saved best model with validation loss: {val_loss_avg:.4f}")
            else:
                patience_counter += 1
                print(f"Validation did not improve. Patience: {patience_counter}/{patience}")
            
            if epoch_idx % 5 == 0:
                best_val_loss = val_loss_avg
                patience_counter = 0
                
                # Save best model
                best_checkpoint_dict = {
                    'epoch': epoch_idx + 1,
                    'model_state_dict': model.state_dict(),
                    'encoder_state_dict': encoder.state_dict(),
                    'optimizer_model_state_dict': optimizer_model.state_dict(),
                    'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
                    'best_val_loss': best_val_loss,
                    'train_loss': train_loss_avg,
                    'step_count': step_count,
                    'img_save_count': img_save_count
                }
                best_checkpoint_path = os.path.join(train_config['task_name'], 'epoch_' + str(epoch_idx + 1) + '_' + train_config['ldm_ckpt_name'])
                torch.save(best_checkpoint_dict, best_checkpoint_path)
                print(f"Saved best model with validation loss: {val_loss_avg:.4f}")
            
            # Early stopping check
            # if patience_counter >= patience:
            #     print(f"Early stopping triggered after {epoch_idx+1} epochs")
            #     break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Always save latest checkpoint
        checkpoint_dict = {
            'epoch': epoch_idx + 1,
            'model_state_dict': model.state_dict(),
            'encoder_state_dict': encoder.state_dict(),
            'optimizer_model_state_dict': optimizer_model.state_dict(),
            'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss': train_loss_avg,
            'step_count': step_count,
            'img_save_count': img_save_count
        }
        checkpoint_path = os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'])
        torch.save(checkpoint_dict, checkpoint_path)
        
        # Save encoder separately for convenience
        encoder_checkpoint_path = os.path.join(train_config['task_name'], 'encoder_' + train_config['ldm_ckpt_name'])
        torch.save(encoder.state_dict(), encoder_checkpoint_path)
        
        print(f"Saved checkpoint to {checkpoint_path}")
        if epoch_idx % 1 == 0:  # Every 5 epochs
            gc.collect()
            torch.cuda.empty_cache()    
        
    
    print('Done Training...')
    
    # Load best model for final save
    best_checkpoint_path = os.path.join(train_config['task_name'], 'best_' + train_config['ldm_ckpt_name'])
    if os.path.exists(best_checkpoint_path):
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        encoder.load_state_dict(best_checkpoint['encoder_state_dict'])
        print(f"Loaded best model with validation loss: {best_checkpoint['best_val_loss']:.4f}")
    
    # Final model save (state dict only, for inference)
    final_model_path = os.path.join(train_config['task_name'], 'final_model_for_inf.pth')
    final_encoder_path = os.path.join(train_config['task_name'], 'final_encoder_for_inf.pth')
    torch.save(model.state_dict(), final_model_path)
    torch.save(encoder.state_dict(), final_encoder_path)
    print(f"Saved final model to {final_model_path}")
    print(f"Saved final encoder to {final_encoder_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/unet_hyperism.yaml', type=str)
    
    import sys
    args, unknown = parser.parse_known_args(sys.argv[1:])
    
    train(args)