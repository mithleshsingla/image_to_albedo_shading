print("|| RAM ||")
import yaml
import argparse
import torch
import random
import os
import numpy as np
from tqdm import tqdm
from vae import VAE
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.utils.data import random_split
from discriminator import Discriminator
import imageio.v3 as iio
import lpips
import gc
#import time
import OpenEXR
import Imath
import torchvision.utils as vutils


# Add this at the top of your script, before importing imageio
from dataloader_image_hyperism import ImageDataset,ImageDataset_d
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
#print("imported all the libraries")
import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name(device)} (CUDA:{torch.cuda.current_device()})")
else:
    print("Using device: CPU")

loss_fn_alex = lpips.LPIPS(net='vgg').to(device)
# Set the device to GPU if available

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
wandb.init(project="ldr_sh_vae_training_trial")

from torch.utils.data import Dataset

def check_nan(tensor, name="tensor"):
    """Check if tensor contains NaN values and print debugging info"""
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        non_nan_mask = ~torch.isnan(tensor)
        if non_nan_mask.any():
            print(f"Non-NaN values stats - Min: {tensor[non_nan_mask].min().item()}, Max: {tensor[non_nan_mask].max().item()}")
        return True
    return False

def to_rgb(image):
    if image.shape[1] == 1:
        return image.repeat(1, 3, 1, 1)
    return image


def weights_init(m):
    """
    Initialize network weights properly to help with training stability.
    Apply this to your model after creation with model.apply(weights_init)
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)


def save_training_samples(output, gt_image, scene_infos, train_config, step_count, img_save_count):
    
    # Try to import OpenEXR if available
    try:
        has_openexr = True
    except ImportError:
        has_openexr = False
        print("Warning: OpenEXR not available, falling back to imageio")

    sample_size = min(8, output.shape[0])
    gt_image    = gt_image[:sample_size].detach().cpu()
    save_output = output[:sample_size].detach().cpu()#.numpy()

    # Base save path
    base_save_path = os.path.join('/home/project/dataset/Hyperism', train_config['task_name'], 'vae_autoencoder_samples')
 
    collage = torch.cat([save_output, gt_image], dim=0)
    os.makedirs("/home/project/dataset/Hyperism/ldr_to_sh/vae_autoencoder_samples/", exist_ok=True)
    output_path = f"/home/project/dataset/Hyperism/ldr_to_sh/vae_autoencoder_samples/{step_count}.png"
    vutils.save_image(collage, output_path, nrow=4, normalize=True)
    # Also save a simple numbered output for easy viewing
    simple_save_path = os.path.join(base_save_path, 'numbered_samples')
    os.makedirs(simple_save_path, exist_ok=True)

    return img_save_count + 1

# Create a combined dataset class
class CombinedDataset(Dataset):
    def __init__(self, albedo_dataset):
        """
        A dataset that matches corresponding images across the three datasets based on scene metadata.
        
        Args:
            sh_dataset: The HDRGrayscaleEXRDataset for spherical harmonics shading
            albedo_dataset: The ImageDataset for albedo (diffuse_reflectance.exr)
            ldr_dataset: The ImageDataset for LDR input (dequantize.exr)
        """
        #self.sh_dataset = sh_dataset
        self.albedo_dataset = albedo_dataset
        # self.ldr_dataset = ldr_dataset
        
        # Create a mapping from scene info to indices for each dataset
        self.matching_indices = self._find_matching_indices()
       
    def _find_matching_indices(self):
        """Find matching indices across all three datasets based on scene info"""
        # Create dictionaries to map scene info to indices for each dataset
        #sh_indices = {}
        albedo_indices = {}
       
        for idx in range(len(self.albedo_dataset)):
            info = self.albedo_dataset.get_scene_info(idx)
            key = (info['ai_folder'], info['scene_folder'], info['frame_num'])
            albedo_indices[key] = idx
        
        albedo_keys = set(albedo_indices.keys())
        # ldr_keys = set(ldr_indices.keys())

        common_keys = albedo_keys

        # Create a list of matching indices
        matching_indices = [
            (albedo_indices[key]) 
            for key in common_keys
        ]
        
        return matching_indices
    
    def __len__(self):
        return len(self.matching_indices)
    
    def __getitem__(self, idx):
        # Get the matching indices for all three datasets
        albedo_idx= self.matching_indices[idx]
        
        # Get the items from each dataset
        #sh_image = self.sh_dataset[sh_idx]
        albedo_image = self.albedo_dataset[albedo_idx]
        # ldr_image = self.ldr_dataset[ldr_idx]
        
        # Also store the scene info for saving output images
        info = self.albedo_dataset.get_scene_info(albedo_idx)

        return albedo_image, info


def kl_divergence(mu, logvar):
    """
    Compute the KL divergence between the encoded distribution and a standard normal distribution.
    This version includes proper batch averaging and clipping to prevent numerical issues.
    """
    # Clamp logvar to prevent extreme values
    logvar = torch.clamp(logvar, min=-10, max=10)
    
    # Calculate KL divergence term by term
   
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / (logvar.size(0) *  logvar.size(1)* logvar.size(2) * logvar.size(3))
    # Average over batch dimension
    return kl_loss.mean()

# Function to evaluate the model on validation set
def validate(model, val_loader, discriminator, recon_criterion, disc_criterion, train_config, kl_weight,step_count, disc_step_start):
    model.eval()
    discriminator.eval()
    val_recon_losses_albedo = []
    val_kl_losses = []
    val_perceptual_losses = []
    val_disc_losses = []
    val_gen_losses = []
    val_total_losses = []
    # For discriminator predictions
    val_real_preds = []
    val_fake_preds = []
    
    
    with torch.no_grad():
        for batch in val_loader:
            albedo_im,_ = batch
            
            # Convert to float and move to device
            #sh_im = sh_im.float().to(device)
            albedo_im = albedo_im.float().to(device)
            #ldr_im = ldr_im.float().to(device)
                       
            # Get model output
            model_output = model(albedo_im)
            output, z, _ = model_output
            mean, logvar = torch.chunk(z, 2, dim=1)
        
            # Calculate reconstruction loss for shading
            recon_loss = recon_criterion(output, albedo_im)
            recon_loss = recon_loss / train_config['autoencoder_acc_steps']
            val_recon_losses_albedo.append(train_config['albedo_weight']*recon_loss.item())

            # Calculate KL loss
            kl_loss = kl_divergence(mean, logvar)
            kl_loss = kl_loss / train_config['autoencoder_acc_steps']
            val_kl_losses.append(kl_weight * kl_loss.item())
            
            # Calculate perceptual loss
            lpips_loss = (loss_fn_alex(output.to(device), albedo_im.to(device)).mean()) # Ensure lpips_loss is a scalar
            lpips_loss = lpips_loss / train_config['autoencoder_acc_steps']
            val_perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss.item())

            gen_loss = 0
            if step_count > disc_step_start :
                # Discriminator predictions
                disc_fake_pred = discriminator(output)
                disc_real_pred = discriminator(albedo_im)

                # Store predictions
                real_probs = torch.sigmoid(disc_real_pred).mean().item()
                fake_probs = torch.sigmoid(disc_fake_pred).mean().item()
                val_real_preds.append(real_probs)
                val_fake_preds.append(fake_probs)

                # Generator adversarial loss
                gen_loss = disc_criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
                gen_loss = gen_loss / train_config['autoencoder_acc_steps']
                val_gen_losses.append(train_config['disc_weight'] * gen_loss.item())

                # Discriminator adversarial loss
                disc_fake_loss = disc_criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = disc_criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                disc_loss = disc_loss / train_config['autoencoder_acc_steps']
                val_disc_losses.append(train_config['disc_weight'] * disc_loss.item())

            # Calculate total loss
            total_loss = (train_config['albedo_weight'] * recon_loss + #(train_config['gradient_weight'] * grad_loss) +
                         (kl_weight * kl_loss) +
                         (train_config['perceptual_weight'] * lpips_loss) +
                         (train_config['disc_weight'] * gen_loss))
            val_total_losses.append(total_loss.item())
    
    model.train()
    discriminator.train()

    
    # Return average losses
    return {
        'recon_loss_albedo': np.mean(val_recon_losses_albedo),
        'kl_loss': np.mean(val_kl_losses),
        #'gradient_loss_albedo': np.mean(val_gradient_losses_albedo),
        'perceptual_loss': np.mean(val_perceptual_losses),
        'gen_loss': np.mean(val_gen_losses) if val_gen_losses else 0,
        'disc_loss': np.mean(val_disc_losses) if val_disc_losses else 0,
        'total_loss': np.mean(val_total_losses),
        'real_prediction': np.mean(val_real_preds) if val_real_preds else 0,
        'fake_prediction': np.mean(val_fake_preds) if val_fake_preds else 0,
    }

def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    #print(config)
    
    dataset_config = config['dataset_params']
    train_config = config['train_params']
    
    # Set the desired seed value #
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################
    
    # Create the model and dataset #
    model = VAE(latent_dim=16).to(device)
    model.apply(weights_init)

    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)
    discriminator.apply(weights_init)

    albedo_dataset = ImageDataset(im_path=dataset_config['im_path'],
                               im_size=dataset_config['im_size'], file_suffix='diffuse_reflectance.exr')
    
    
    # Create the combined dataset
    combined_dataset = CombinedDataset(albedo_dataset)

    # Split dataset into train and validation (90:10 ratio)
    dataset_size = len(combined_dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    

    indices = np.arange(len(combined_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(combined_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(combined_dataset, val_indices)

    print(f"Total dataset size: {dataset_size}")
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    
    wandb.config.update({
        "learning_rate_autoencoder": train_config['autoencoder_lr'],
        "learning_rate_discriminator": train_config['discriminator_lr'],
        "batch_size": train_config['autoencoder_batch_size'],
        "gradient_weight": train_config['gradient_weight'],
        "albedo_weight": train_config['albedo_weight'],
        "kl_weight": train_config['kl_weight'],
        "perceptual_weight": train_config['perceptual_weight'],
        "disc_weight": train_config['disc_weight'],
        "disc_start": train_config['disc_start'],
        "autoencoder_acc_steps": train_config['autoencoder_acc_steps']
    })

    train_loader = DataLoader(train_dataset,
                             batch_size=wandb.config['batch_size'],
                             shuffle=True , num_workers=16 , pin_memory=False)
    
    val_loader = DataLoader(val_dataset,
                           batch_size=wandb.config['batch_size'],
                           shuffle=False , num_workers=16 , pin_memory=False)
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.makedirs(train_config['task_name'], exist_ok=True)
        
    num_epochs = train_config['autoencoder_epochs']

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    # Disc Loss can even be BCEWithLogits
    disc_criterion = torch.nn.BCEWithLogitsLoss()


    optimizer_d = Adam(discriminator.parameters(), lr=wandb.config['learning_rate_discriminator'], betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=wandb.config['learning_rate_autoencoder'], betas=(0.5, 0.999) )

    scaler = GradScaler('cuda') if device == 'cuda' else None
    # Setup schedulers
    scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.9, patience=10, min_lr=0.00001)
    scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', factor=0.9, patience=10, min_lr=0.000001)

    disc_step_start = wandb.config['disc_start']
    step_count = 0
    start_epoch = 0
    best_val_loss = float('inf')
    
    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = wandb.config['autoencoder_acc_steps']
    image_save_steps = train_config['autoencoder_img_save_steps']
    img_save_count = 0
    
    # Lists to store epoch metrics
    train_losses_history = []
    val_losses_history = []
    
    # Check if checkpoint exists and load it for resuming training
    checkpoint_path = "checkpoints/epoch_41_best_autoencoder_model_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        start_epoch = checkpoint['epoch']
        step_count = checkpoint.get('step_count', 0)
        img_save_count = checkpoint.get('img_save_count', 0)
        best_val_loss = checkpoint['best_val_loss']
        train_losses_history = checkpoint.get('train_losses_history', [])
        val_losses_history = checkpoint.get('val_losses_history', [])
        logging.info(f"Resuming from epoch {start_epoch} with best validation loss: {best_val_loss}")


    #logging.info(f"Learning rates updated: Generator -> {new_lr_g}, Discriminator -> {new_lr_d}")
    for epoch_idx in range(start_epoch, num_epochs):
        # Training metrics
        #recon_losses_shading = []
        recon_losses_albedo = []
        kl_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []
        train_real_preds = []
        train_fake_preds = []
        #grad_losses_shading = []
        # grad_losses_albedo = []
        losses = []
        
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        # Training loop
        for batch in tqdm(train_loader):

            #start_time = time.perf_counter()
            step_count += 1
                        
            # Unpack the batch - each element is a batch of images from each dataset
            albedo_im, scene_infos = batch
            
            #sh_im = sh_im.float().to(device)
            albedo_im = albedo_im.float().to(device)

            with autocast(device_type='cuda'):
                # Fetch autoencoders output(reconstructions)
                model_output = model(albedo_im)
                output, h,_ = model_output
                mean, logvar = torch.chunk(h, 2, dim=1)
                if check_nan(output, "raw_model_output"):
                    print("NaN values detected in model output! Skipping this batch.")
                    continue
                            
                # Image Saving Logic
                if step_count % image_save_steps == 0 or step_count == 1:
                    img_save_count = save_training_samples(
                        output, albedo_im, scene_infos, train_config, step_count, img_save_count
                    )

                ######### Optimize Generator ##########
                # L2 Loss for shading
                recon_loss = recon_criterion(output, albedo_im)
                recon_loss = recon_loss / acc_steps 
                recon_losses_albedo.append(wandb.config['albedo_weight'] * recon_loss.item()) # add average loss for 1 image

                kl_weight = wandb.config['kl_weight']
                kl_loss = kl_divergence(mean, logvar)
                kl_loss = kl_loss / acc_steps
                kl_losses.append(kl_weight * kl_loss.item())

                # total_loss_generator
                g_loss = (wandb.config['albedo_weight'] * recon_loss + 
                        #   (wandb.config['gradient_weight'] * grad_loss) + 
                          (kl_weight * kl_loss ))
                # Adversarial loss only if disc_step_start steps passed
                if step_count > disc_step_start:
                    disc_fake_pred = discriminator(model_output[0])
                    disc_fake_loss = disc_criterion(disc_fake_pred,
                                                    torch.ones(disc_fake_pred.shape,
                                                            device=disc_fake_pred.device))
                    disc_fake_loss = disc_fake_loss / acc_steps
                    gen_losses.append(wandb.config['disc_weight'] * disc_fake_loss.item())
                    g_loss += wandb.config['disc_weight'] * disc_fake_loss
                # LPIPS Loss
                
                # Calculate perceptual loss
                lpips_loss = (loss_fn_alex(output, albedo_im).mean())  # Ensure lpips_loss is a scalar
                lpips_loss = lpips_loss / acc_steps
                
                perceptual_losses.append(wandb.config['perceptual_weight'] * lpips_loss.item())
                g_loss += wandb.config['perceptual_weight'] * lpips_loss
                losses.append(g_loss.item())
                #g_loss.backward()
                #####################################
            
            if scaler is not None:
                scaler.scale(g_loss).backward()
            else:
                g_loss.backward()

            ######### Optimize Discriminator #######
            if step_count > disc_step_start and step_count % 2 == 0:
                with autocast(device_type='cuda'):
                    fake = output
                    disc_fake_pred = discriminator(fake.detach())
                    disc_real_pred = discriminator(albedo_im)
                    disc_fake_loss = disc_criterion(disc_fake_pred,
                                                    torch.zeros(disc_fake_pred.shape,
                                                                device=disc_fake_pred.device))
                    disc_real_loss = disc_criterion(disc_real_pred,
                                                    torch.ones(disc_real_pred.shape,
                                                            device=disc_real_pred.device))
                    disc_loss = wandb.config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                    disc_loss = disc_loss / acc_steps
                    disc_losses.append(disc_loss.item())
                    with torch.no_grad():
                        # Convert logits to probabilities using sigmoid
                        real_probs = torch.sigmoid(disc_real_pred).mean().item()
                        fake_probs = torch.sigmoid(disc_fake_pred).mean().item()
                        train_real_preds.append(real_probs)
                        train_fake_preds.append(fake_probs)
                    
                # Scale the discriminator loss and backward
                if scaler is not None:
                    scaler.scale(disc_loss).backward()
                else:
                    disc_loss.backward()


                if step_count % acc_steps == 0:
                    # Apply gradient clipping (see below)
                    if scaler is not None:
                        # Unscale before clipping
                        scaler.unscale_(optimizer_d)
                        # Here we'll add gradient clipping (code below)
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                        # Step with scaler
                        scaler.step(optimizer_d)
                        scaler.update()
                    else:
                        # Here we'll add gradient clipping (code below)
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                        optimizer_d.step()
                    optimizer_d.zero_grad()    
                    #####################################
            
            
            if step_count % acc_steps == 0:
                # Apply gradient clipping (see below)
                if scaler is not None:
                    # Unscale before clipping
                    scaler.unscale_(optimizer_g)
                    # Here we'll add gradient clipping (code below)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # Step with scaler
                    scaler.step(optimizer_g)
                    scaler.update()
                else:
                    # Here we'll add gradient clipping (code below)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer_g.step()
                optimizer_g.zero_grad()
        
        if step_count > disc_step_start and step_count % 2 == 0:
             # Final optimizer steps at end of epoch
            optimizer_d.step()
            optimizer_d.zero_grad()
        
        optimizer_g.step()
        
        
        # Calculate validation metrics  
        val_metrics = validate(model, val_loader, discriminator, recon_criterion, disc_criterion, train_config,kl_weight,step_count,disc_step_start)
        

        # Store epoch metrics for plotting
        train_loss = np.mean(losses)
        val_loss = val_metrics['total_loss']
        train_losses_history.append(train_loss)
        val_losses_history.append(val_loss)
        


        epochs_since_disc_start = max(0, epoch_idx - (disc_step_start // len(train_loader)))

        if step_count <= disc_step_start:
            # Before discriminator starts - continue normal generator training
            scheduler_g.step(val_loss)
        elif epochs_since_disc_start <= 20:
            # Stabilization period - don't adjust any learning rates
            scheduler_d.step(val_metrics["disc_loss"])
        else:
            # After stabilization - resume normal scheduling
            scheduler_g.step(val_loss)
            scheduler_d.step(val_metrics["disc_loss"])

        # After validation and calculating epoch metrics
        wandb.log({
            "epoch": epoch_idx + 1,
            "train/recon_loss_albedo": np.mean(recon_losses_albedo),
            #"train/gradient_loss_albedo": np.mean(grad_losses_albedo),
            "train/kl_loss": np.mean(kl_losses),
            "train/perceptual_loss": np.mean(perceptual_losses),
            "train/gen_loss": np.mean(gen_losses) if len(gen_losses) > 0 else 0,
            "train/disc_loss": np.mean(disc_losses) if len(disc_losses) > 0 else 0,
            "train/total_loss": train_loss,
            "train/real_prediction": np.mean(train_real_preds) if len(train_real_preds) > 0 else 0,
            "train/fake_prediction": np.mean(train_fake_preds) if len(train_fake_preds) > 0 else 0,
            "val/recon_loss_albedo": val_metrics["recon_loss_albedo"],
            #"val/gradient_loss_albedo": val_metrics["gradient_loss_albedo"],
            "val/kl_loss": val_metrics["kl_loss"],
            "val/perceptual_loss": val_metrics["perceptual_loss"],
            "val/gen_loss": val_metrics["gen_loss"],
            "val/disc_loss": val_metrics["disc_loss"],
            "val/total_loss": val_metrics["total_loss"],
            "val/real_prediction": val_metrics["real_prediction"],
            "val/fake_prediction": val_metrics["fake_prediction"],
            "learning_rate/generator": optimizer_g.param_groups[0]['lr'],
            "learning_rate/discriminator": optimizer_d.param_groups[0]['lr']
        })

        # Print epoch results
        print('\n' + '=' * 80)
        print(f'Epoch {epoch_idx + 1}/{num_epochs}')
        print('-' * 80)
        print('TRAINING:')
        print(f'Recon Loss_albedo: {np.mean(recon_losses_albedo):.4f} | '
              #f'Gradient Loss_albedo: {np.mean(grad_losses_albedo):.4f} | '
              f'KL Loss: {np.mean(kl_losses):.4f} | '
              f'Perceptual Loss: {np.mean(perceptual_losses):.4f}')
        
        if len(disc_losses) > 0 and len(gen_losses) > 0:
            print(f'Generator Loss: {np.mean(gen_losses):.4f} | '
                  f'Discriminator Loss: {np.mean(disc_losses):.4f}')
            
        print(f'Total Training Loss: {train_loss:.4f}')
        
        print('\nVALIDATION:')
        print(f'Recon Loss_shading: {val_metrics["recon_loss_albedo"]:.4f} | '
              #f'Gradient Loss_shading: {val_metrics["gradient_loss_albedo"]:.4f} | '
              f'KL Loss: {val_metrics["kl_loss"]:.4f} | '
              f'Perceptual Loss: {val_metrics["perceptual_loss"]:.4f}')
        
        print(f'Generator Loss: {val_metrics["gen_loss"]:.4f} | '
              f'Discriminator Loss: {val_metrics["disc_loss"]:.4f}')
            
        print(f'Total Validation Loss: {val_metrics["total_loss"]:.4f}')
        
        # Check if validation loss improved
        if val_loss < best_val_loss:
            print(f"\nValidation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            # Save only the best model checkpoint
            checkpoint = {
                'epoch': epoch_idx + 1,
                'model_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'best_val_loss': best_val_loss,
                'step_count': step_count,
                'img_save_count': img_save_count,
                'train_losses_history': train_losses_history,
                'val_losses_history': val_losses_history
            }
            
            torch.save(checkpoint, os.path.join(train_config['task_name'], 'best_autoencoder_model_checkpoint.pth'))
            
            
        else:
            print(f"Validation loss did not improve from {best_val_loss:.4f}")

        if (epoch_idx + 1) % 1 == 0:
            checkpoint = {
                'epoch': epoch_idx + 1,
                'model_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'best_val_loss': best_val_loss,
                'step_count': step_count,
                'img_save_count': img_save_count,
                'train_losses_history': train_losses_history,
                'val_losses_history': val_losses_history
            }

            torch.save(checkpoint, os.path.join(train_config['task_name'], f'epoch_{epoch_idx + 1}_best_autoencoder_model_checkpoint.pth'))

            # Save individual model files for compatibility with original code
            torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                       train_config['vae_autoencoder_ckpt_name']))
            torch.save(discriminator.state_dict(), os.path.join(train_config['task_name'],
                                                               train_config['vae_discriminator_ckpt_name']))  # Every 10 epochs
        if epoch_idx % 1 == 0:  # Every 5 epochs
            gc.collect()
            torch.cuda.empty_cache()    
        print('=' * 80 + '\n')
    
    print('Done Training...')
    
    # Save final training history
    np.savez(os.path.join(train_config['task_name'], 'training_history.npz'),
             train_losses=np.array(train_losses_history),
             val_losses=np.array(val_losses_history))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/autoen_alb.yaml', type=str)

    # Handle Jupyter/IPython arguments
    import sys
    args, unknown = parser.parse_known_args(sys.argv[1:])
    
    train(args)