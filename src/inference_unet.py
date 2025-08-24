import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.unet import Unet
from models.vae import VAE
from models.unet import Encoder
from src.dataloader_image_hyperism import ImageDatasetwv_h5, ImageDatasetwv_h5_al
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
import torchvision.utils as vutils
from collections import OrderedDict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def strip_orig_mod(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
        new_state_dict[new_key] = v
    return new_state_dict

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

def load_models(config, device):
    """Load and initialize all required models"""
    
    # Initialize models
    autoencoder_model_config = config['autoencoder_params']
    dataset_config = config['dataset_params_input']
    
    
    # Initialize encoder
    encoder = Encoder(im_channels=dataset_config['im_channels']).to(device)
    
    # Initialize Unet model
    model = Unet(im_channels=autoencoder_model_config['z_channels']).to(device)
    
    # Initialize VAE for decoding
    vae = VAE(latent_dim=16).to(device)
    vae.eval()
    
    # Load VAE weights
    vae_path ="checkpoints\vae_model.pth"
    if os.path.exists(vae_path):
        print(f'Loading VAE checkpoint from {vae_path}')
        checkpoint_vae = torch.load(vae_path, weights_only=False, map_location=device)
        vae.load_state_dict(checkpoint_vae['model_state_dict'])
        print('VAE loaded successfully')
    else:
        print(f'Warning: VAE checkpoint not found at {vae_path}')
        return None, None, None
    
    # Freeze VAE parameters
    for param in vae.parameters():
        param.requires_grad = False
    
    return model, encoder, vae

def load_trained_weights(model, encoder, checkpoint_path, device):
    """Load trained model weights"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return False
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(strip_orig_mod(checkpoint['model_state_dict']))
        encoder.load_state_dict(strip_orig_mod(checkpoint['encoder_state_dict']))
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # If it's just state dict
        model.load_state_dict(strip_orig_mod(checkpoint))
        print("Loaded model weights only")
    
    return True

def create_inference_datasets(config):
    """Create datasets for inference"""
    dataset_config = config['dataset_params_input']
    dataset_albedo = config['albedo_params']
    train_config = config['train_params']
    latent_path = train_config['vae_latent_dir_name']
    
    # Create test/validation datasets
    test_dataset_encoder = ImageDatasetwv_h5(
        im_path=dataset_config['im_path'],
        im_size=dataset_config['im_size'],
        split='val',  # or 'test' if you have a test split
        latent_path=latent_path
    )
    
    test_albedo_dataset = ImageDatasetwv_h5_al(
        im_path=dataset_albedo['im_path'],
        im_size=dataset_config['im_size'], 
        file_suffix='.diffuse_reflectance.exr',
        split='val',  # or 'test'
        latent_path=latent_path
    )
    
    return test_dataset_encoder, test_albedo_dataset

def run_inference(model, encoder, vae, data_loader, config, output_dir, num_inference_steps=20):
    """Run inference on the dataset"""
    
    model.eval()
    encoder.eval()
    vae.eval()
    
    train_config = config['train_params']
    autoencoder_model_config = config['autoencoder_params']
    
    # Create wrapped model for sampling
    wrapped_model = WrappedModel(model, encoder)
    solver = ODESolver(velocity_model=wrapped_model)
    
    # Get time discretization for ODE solver
    
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (cond_img, albedo_img) in enumerate(tqdm(data_loader, desc="Running Inference")):
            cond_img = cond_img.float().to(device)
            albedo_img = albedo_img.float().to(device)
            
            current_batch_size = cond_img.size(0)
            latent_channels = autoencoder_model_config['z_channels']
            latent_size = train_config['im_size_lt']
            
            # Generate initial noise
            x_init = torch.randn(current_batch_size, latent_channels, latent_size, latent_size).to(device)
            
            # Process condition images through encoder
            cond_encoded = encoder(cond_img)
            
            # Set condition for the wrapped model
            wrapped_model.set_condition(cond_encoded)
            
            # Sample from the model
            samples = solver.sample(
                x_init=x_init,
                method='euler',
                step_size=0.5,  # Smaller step size for better quality
                return_intermediates=False
            )
            
            # Decode samples using VAE decoder
            samples = samples.float()
            decoded_output = vae.decoder(samples)
            
            # Save results
            save_inference_results(
                decoded_output, cond_img, albedo_img, 
                output_dir, batch_idx, current_batch_size
            )

def save_inference_results(generated, condition, ground_truth, output_dir, batch_idx, batch_size):
    """Save inference results"""
    
    # Save individual images
    for i in range(batch_size):
        # Create a comparison collage for each sample
        sample_collage = torch.cat([
            generated[i:i+1].cpu(),
            condition[i:i+1].cpu(), 
            ground_truth[i:i+1].cpu()
        ], dim=0)
        
        sample_path = os.path.join(output_dir, f"sample_{batch_idx:04d}_{i:02d}.png")
        vutils.save_image(sample_collage, sample_path, nrow=1, normalize=True, padding=2)
        
        # Save individual components
        vutils.save_image(generated[i:i+1].cpu(), 
                         os.path.join(output_dir, f"generated_{batch_idx:04d}_{i:02d}.png"), 
                         normalize=True)
    
    # Save batch collage
    batch_collage = torch.cat([generated.cpu(), condition.cpu(), ground_truth.cpu()], dim=0)
    batch_path = os.path.join(output_dir, f"batch_{batch_idx:04d}_collage.png")
    vutils.save_image(batch_collage, batch_path, nrow=batch_size, normalize=True, padding=2)

def run_single_image_inference(model, encoder, vae, condition_image_path, config, output_path, num_inference_steps=20):
    """Run inference on a single image"""
    from PIL import Image
    import torchvision.transforms as transforms
    
    model.eval()
    encoder.eval()
    vae.eval()
    
    # Load and preprocess the condition image
    transform = transforms.Compose([
        transforms.Resize((config['dataset_params_input']['im_size'], config['dataset_params_input']['im_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust normalization as needed
    ])
    
    condition_img = Image.open(condition_image_path).convert('RGB')
    condition_tensor = transform(condition_img).unsqueeze(0).to(device)
    
    train_config = config['train_params']
    autoencoder_model_config = config['autoencoder_params']
    
    # Create wrapped model for sampling
    wrapped_model = WrappedModel(model, encoder)
    solver = ODESolver(velocity_model=wrapped_model)
    
    
    with torch.no_grad():
        # Generate initial noise
        latent_channels = autoencoder_model_config['z_channels']
        latent_size = train_config['im_size_lt']
        x_init = torch.randn(1, latent_channels, latent_size, latent_size).to(device)
        
        # Process condition image through encoder
        cond_encoded = encoder(condition_tensor)
        
        # Set condition for the wrapped model
        wrapped_model.set_condition(cond_encoded)
        
        # Sample from the model
        samples = solver.sample(
            x_init=x_init,
            method='euler',
            step_size=0.5,
            return_intermediates=False
        )
        
        # Decode samples
        samples = samples.float()
        decoded_output = vae.decoder(samples)
        
        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        vutils.save_image(decoded_output, output_path, normalize=True)
        print(f"Saved inference result to {output_path}")

def inference_main(args):
    """Main inference function"""
    
    # Load configuration
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    print("Loading models...")
    model, encoder, vae = load_models(config, device)
    if model is None:
        print("Failed to load models")
        return
    
    # Load trained weights
    checkpoint_path = args.checkpoint_path
    if not load_trained_weights(model, encoder, checkpoint_path, device):
        print("Failed to load trained weights")
        return
    
    print("Models loaded successfully!")
    
    if args.single_image:
        # Single image inference
        print(f"Running single image inference on {args.input_image}")
        run_single_image_inference(
            model, encoder, vae, 
            args.input_image, config, args.output_path,
            num_inference_steps=args.num_steps
        )
    else:
        # Batch inference on dataset
        print("Creating inference datasets...")
        test_dataset_encoder, test_albedo_dataset = create_inference_datasets(config)
        
        # Create simple data loader (assuming datasets are aligned)
        # Note: You might need to create a proper combined dataset like in training
        test_loader = DataLoader(
            list(zip(test_dataset_encoder, test_albedo_dataset)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        print(f"Running batch inference on {len(test_dataset_encoder)} samples...")
        run_inference(
            model, encoder, vae, test_loader, config, 
            args.output_dir, num_inference_steps=args.num_steps
        )
    
    print("Inference completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flow Matching Model Inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/unet_hyperism.yaml', type=str,
                        help='Path to config file')
    parser.add_argument('--checkpoint', dest='checkpoint_path',
                        required=True, type=str,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', default='inference_results', type=str,
                        help='Directory to save inference results')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size for inference')
    parser.add_argument('--num_steps', default=20, type=int,
                        help='Number of inference steps for ODE solver')
    parser.add_argument('--single_image', action='store_true',
                        help='Run inference on a single image')
    parser.add_argument('--input_image', type=str,
                        help='Path to input image (for single image inference)')
    parser.add_argument('--output_path', type=str,
                        help='Output path for single image inference')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.single_image and (not args.input_image or not args.output_path):
        print("For single image inference, both --input_image and --output_path are required")
        exit(1)
    
    inference_main(args)