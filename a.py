import torch
import os
import logging
from pathlib import Path

def strip_orig_mod(state_dict):
    """Remove '_orig_mod.' prefix from state dict keys if present (for compiled models)"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[10:]  # Remove '_orig_mod.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def extract_vae_checkpoint(checkpoint_path, output_dir="inference_models"):
    """
    Extract VAE model and discriminator state dicts from training checkpoint
    
    Args:
        checkpoint_path: Path to the training checkpoint
        output_dir: Directory to save the extracted models
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        model_state = strip_orig_mod(checkpoint['model_state_dict'])
        model_output_path = os.path.join(output_dir, "vae_model.pth")
        torch.save(model_state, model_output_path)
        print(f"VAE model saved to: {model_output_path}")
    
    # Extract discriminator state dict
    if 'discriminator_state_dict' in checkpoint:
        discriminator_state = strip_orig_mod(checkpoint['discriminator_state_dict'])
        discriminator_output_path = os.path.join(output_dir, "discriminator_model.pth")
        torch.save(discriminator_state, discriminator_output_path)
        print(f"Discriminator model saved to: {discriminator_output_path}")
    
    # Save metadata
    metadata = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'best_val_loss': checkpoint.get('best_val_loss', 'unknown'),
        'step_count': checkpoint.get('step_count', 'unknown'),
    }
    metadata_path = os.path.join(output_dir, "vae_metadata.txt")
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    print(f"Metadata saved to: {metadata_path}")

def extract_unet_encoder_checkpoint(checkpoint_path, output_dir="inference_models"):
    """
    Extract UNet and Encoder state dicts from training checkpoint and save as combined inference checkpoint
    
    Args:
        checkpoint_path: Path to the training checkpoint
        output_dir: Directory to save the extracted models
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Check if it's a dict with state_dict keys or just the model weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Create combined inference checkpoint
        inference_checkpoint = {}
        
        # Extract and clean model state dict
        model_state = strip_orig_mod(checkpoint['model_state_dict'])
        inference_checkpoint['model_state_dict'] = model_state
        
        # Extract and clean encoder state dict if present
        if 'encoder_state_dict' in checkpoint:
            encoder_state = strip_orig_mod(checkpoint['encoder_state_dict'])
            inference_checkpoint['encoder_state_dict'] = encoder_state
        
        # Add metadata (optional)
        inference_checkpoint['epoch'] = checkpoint.get('epoch', 'unknown')
        inference_checkpoint['best_val_loss'] = checkpoint.get('best_val_loss', 'unknown')
        inference_checkpoint['step_count'] = checkpoint.get('step_count', 'unknown')
        
        # Save combined checkpoint
        combined_output_path = os.path.join(output_dir, "unet_encoder_inference.pth")
        torch.save(inference_checkpoint, combined_output_path)
        print(f"Combined UNet+Encoder inference checkpoint saved to: {combined_output_path}")
        
    else:
        # Checkpoint contains only model weights - save as is but wrap in dict format
        inference_checkpoint = {
            'model_state_dict': strip_orig_mod(checkpoint)
        }
        combined_output_path = os.path.join(output_dir, "unet_encoder_inference.pth")
        torch.save(inference_checkpoint, combined_output_path)
        print(f"UNet model (weights only) saved as inference checkpoint to: {combined_output_path}")

def create_inference_loader(model_path, model_class, device='cpu'):
    """
    Create a function to load model for inference
    
    Args:
        model_path: Path to the saved model state dict
        model_class: The model class to instantiate
        device: Device to load the model on
    
    Returns:
        Loaded model ready for inference
    """
    def load_model():
        model = model_class()  # You'll need to pass appropriate args here
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    return load_model

# Usage example
if __name__ == "__main__":
    # Extract VAE checkpoint
    vae_checkpoint_path = "D:\\mtech\\checkpoints\\epoch_41_best_autoencoder_model_checkpoint.pth"
    if os.path.exists(vae_checkpoint_path):
        extract_vae_checkpoint(vae_checkpoint_path, "inference_models/vae")
    
    # Extract UNet+Encoder checkpoint
    unet_checkpoint_path = "D:\\mtech\\checkpoints\\epoch_191_flow_model_ckpt.pth"
    if os.path.exists(unet_checkpoint_path):
        extract_unet_encoder_checkpoint(unet_checkpoint_path, "inference_models/unet_encoder")
    
    print("\nExtraction complete! Your inference models are ready.")
    print("Directory structure:")
    print("inference_models/")
    print("├── vae/")
    print("│   ├── vae_model.pth")
    print("│   ├── discriminator_model.pth")
    print("│   └── vae_metadata.txt")
    print("└── unet_encoder/")
    print("    └── unet_encoder_inference.pth  # Combined checkpoint")