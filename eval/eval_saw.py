import numpy as np
import torch
from scipy.ndimage import maximum_filter
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import yaml
import argparse
import torch
import os
import numpy as np
from collections import OrderedDict
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
import logging
from pathlib import Path
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Setup device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name(device)} (CUDA:{torch.cuda.current_device()})")
else:
    print("Using device: CPU")


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
    
def strip_orig_mod(state_dict):
    """Remove '_orig_mod.' prefix from state dict keys"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
        new_state_dict[new_key] = v
    return new_state_dict


class ShadingAPEvaluator:
    """
    AP evaluator specifically for shading prediction accuracy
    """
    def __init__(self, model, encoder, vae, device='cuda:1'):
        self.model = model
        self.encoder = encoder
        self.vae = vae
        self.device = device
        
    def switch_to_eval(self):
        """Switch models to evaluation mode"""
        self.model.eval()
        self.encoder.eval()
        self.vae.eval()
        
    def switch_to_train(self):
        """Switch models to training mode"""
        self.model.train()
        self.encoder.train() 
        self.vae.train()

    def compute_pr_shading(self, triplet_loader, thres_count=400, bl_filter_size=10):
        """
        Compute Precision-Recall specifically for shading prediction quality
        
        Args:
            triplet_loader: DataLoader with HDR/Albedo/Shading triplets
            thres_count: Number of thresholds for PR curve
            bl_filter_size: Filter size for gradient magnitude filtering
            
        Returns:
            Average Precision (AP) score for shading prediction
        """
        thres_list = self.gen_pr_thres_list(thres_count)
        
        print(f"Computing Shading PR with {len(thres_list)} thresholds")
        
        # Compute PR using shading prediction error
        rdic_list = self.get_precision_recall_list_shading(
            triplet_loader=triplet_loader,
            thres_list=thres_list,
            bl_filter_size=bl_filter_size
        )
        
        if not rdic_list:
            print("Warning: No valid results for Shading PR computation")
            return 0.0
        
        # Build PR curve
        plot_arr = np.empty((len(rdic_list) + 2, 2))
        
        # Extrapolate starting point
        plot_arr[0, 0] = 0.0
        plot_arr[0, 1] = rdic_list[0]['overall_prec'] if rdic_list else 0.5
        
        # Fill PR points
        for i, rdic in enumerate(rdic_list):
            plot_arr[i+1, 0] = rdic['overall_recall']
            plot_arr[i+1, 1] = rdic['overall_prec']
        
        # Extrapolate end point
        plot_arr[-1, 0] = 1.0
        plot_arr[-1, 1] = 0.5
        
        # Calculate AP using trapezoidal rule
        AP = np.trapezoid(plot_arr[:, 1], plot_arr[:, 0])
        
        return AP

    def get_precision_recall_list_shading(self, triplet_loader, thres_list, bl_filter_size):
        """
        Get precision-recall list based on shading prediction accuracy
        """
        output_count = len(thres_list)
        overall_conf_mx_list = [
            np.zeros((3, 2), dtype=int)  # Following SAW's 3-class structure: [NS-ND, NS-SB, S] x [smooth, non-smooth]
            for _ in range(output_count)
        ]
        
        count = 0
        total_processed = 0
        
        with torch.no_grad():
            
            wrapped_model = WrappedModel(self.model, self.encoder)
            solver = ODESolver(velocity_model=wrapped_model)
                
            for batch_idx, batch_data in enumerate(triplet_loader):
                try:
                    # Extract data
                    hdr_im = batch_data['hdr'].float().to(self.device)
                    albedo_im = batch_data['albedo'].float().to(self.device)
                    shading_im = batch_data['shading'].float().to(self.device)
                    scene_names = batch_data['scene_name']
                    
                    batch_size = hdr_im.size(0)
                    
                    # Generate albedo predictions using your existing model pipeline
                    #cond_encoded = self.encoder(hdr_im)
                    
                    # Sample from model (you'll need to replace this with your actual sampling code)
                    latent_size = 32  # Adjust based on your model
                    latent_channels = 16  # Adjust based on your model
                    x_init = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=self.device)
                    
                    with torch.amp.autocast(device_type='cuda:1'):
                            cond_encoded = self.encoder(hdr_im)
                            
                            wrapped_model.set_condition(cond_encoded)
                            
                            # Sample from the model
                            samples = solver.sample(
                                x_init=x_init,
                                method="midpoint",
                                step_size=0.001,
                                return_intermediates=False
                            )
                            
                            # Decode to get predicted albedo
                            predicted_albedo = self.vae.decoder(samples.float())
                        
                    predicted_albedo_01 = (predicted_albedo + 1.0) / 2.0
                    predicted_albedo_01 = torch.clamp(predicted_albedo_01, 0.0, 1.0)
                    
                    # Process each image in batch
                    for b in range(batch_size):
                        print(f"Processing shading evaluation for image {count + 1}, scene: {scene_names[b]}")
                        
                        # Get single image data
                        single_hdr = hdr_im[b:b+1]
                        single_albedo_gt = albedo_im[b:b+1]
                        single_shading_gt = shading_im[b:b+1]
                        single_albedo_pred = predicted_albedo_01[b:b+1]
                        
                        # Calculate predicted shading from HDR and predicted albedo
                        predicted_shading = self.calculate_shading_from_hdr_albedo(
                            single_hdr, single_albedo_pred
                        )
                        predicted_shading = torch.clamp(predicted_shading, 0.0, 1.0)
                        
                        # Convert to numpy for evaluation
                        pred_shading_np = predicted_shading[0, 0].cpu().numpy()
                        gt_shading_np = single_shading_gt[0, 0].cpu().numpy()
                        
                        # Compute confusion matrices for shading evaluation
                        conf_mx_list = self.eval_shading_prediction(
                            pred_shading=pred_shading_np,
                            gt_shading=gt_shading_np,
                            thres_list=thres_list,
                            bl_filter_size=bl_filter_size,
                            scene_name=scene_names[b]
                        )
                        
                        # Accumulate confusion matrices
                        for i, conf_mx in enumerate(conf_mx_list):
                            if conf_mx is not None:
                                overall_conf_mx_list[i] += conf_mx
                        
                        count += 1
                        
                    total_processed += 1
                    
                    # Early stopping for debugging (remove in production)
                    # if total_processed >= 50:  # Process only first 50 batches
                    #     print(f"Early stopping after {total_processed} batches for debugging")
                    #     break
                        
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        # Convert confusion matrices to precision/recall with class weights
        class_weights = [1, 1, 2]  # Following SAW's class weighting
        ret = []
        for i in range(output_count):
            overall_prec, overall_recall = self.get_pr_from_conf_mx(
                conf_mx=overall_conf_mx_list[i],
                class_weights=class_weights
            )
            
            ret.append(dict(
                overall_prec=overall_prec,
                overall_recall=overall_recall,
                overall_conf_mx=overall_conf_mx_list[i],
            ))
        
        return ret

    def eval_shading_prediction(self, pred_shading, gt_shading, thres_list, 
                               bl_filter_size, scene_name):
        """
        Evaluate shading prediction quality using gradient magnitude analysis
        This follows SAW's approach but applied to shading prediction vs GT shading
        """
        # Ensure positive values and convert to log space for gradient computation
        pred_shading_safe = np.maximum(pred_shading, 1e-4)
        gt_shading_safe = np.maximum(gt_shading, 1e-4)
        
        pred_shading_log = np.log(pred_shading_safe)
        gt_shading_log = np.log(gt_shading_safe)
        
        # Compute gradient magnitudes for predicted shading
        pred_gradmag = self.compute_gradmag(pred_shading_log)
        pred_gradmag = np.abs(pred_gradmag)
        
        # Apply maximum filter if specified (following SAW)
        if bl_filter_size > 0:
            pred_gradmag_max = maximum_filter(pred_gradmag, size=bl_filter_size)
        else:
            pred_gradmag_max = pred_gradmag
        
        # Create ground truth labels based on GT shading characteristics
        # This is the key part - we need to classify regions as smooth vs non-smooth
        # based on the ground truth shading
        y_true = self.create_shading_labels(gt_shading, gt_shading_log)
        
        # Create mask for valid pixels (avoiding very dark/bright regions)
        valid_mask = (gt_shading > 0.005) & (gt_shading < 0.995) & (pred_shading > 0.005)
        
        y_true_flat = np.ravel(y_true)
        valid_mask_flat = np.ravel(valid_mask)
        
        # If no valid pixels, return None for all thresholds
        if not np.any(valid_mask_flat):
            return [None] * len(thres_list)
        
        ret = []
        for thres in thres_list:
            # Predicted labels: smooth if gradient magnitude < threshold
            y_pred = (pred_gradmag < thres).astype(int)
            y_pred_max = (pred_gradmag_max < thres).astype(int)
            
            y_pred_flat = np.ravel(y_pred)
            y_pred_max_flat = np.ravel(y_pred_max)
            
            assert y_pred_flat.shape == y_true_flat.shape
            
            # Compute confusion matrix only on valid pixels
            if np.any(valid_mask_flat):
                confusion_matrix = self.grouped_weighted_confusion_matrix(
                    y_true_flat[valid_mask_flat], 
                    y_pred_flat[valid_mask_flat], 
                    y_pred_max_flat[valid_mask_flat],
                    gt_shading  # Use for weighting
                )
            else:
                confusion_matrix = None
                
            ret.append(confusion_matrix)
        
        return ret

    def create_shading_labels(self, gt_shading, gt_shading_log):
        """
        Create ground truth labels for shading regions
        0: Normal/depth discontinuity non-smooth shading (NS-ND)
        1: Shadow boundary non-smooth shading (NS-SB)  
        2: Smooth shading (S)
        """
        # Compute GT shading gradients
        gt_gradmag = self.compute_gradmag(gt_shading_log)
        gt_gradmag = np.abs(gt_gradmag)
        
        # Find smooth regions (low gradient magnitude)
        smooth_threshold = np.percentile(gt_gradmag, 70)  # Top 30% are non-smooth
        shadow_threshold = np.percentile(gt_shading, 20)   # Bottom 20% might be shadow regions
        
        labels = np.zeros_like(gt_shading, dtype=int)
        
        # Label smooth regions (class 2)
        smooth_mask = gt_gradmag < smooth_threshold
        labels[smooth_mask] = 2
        
        # Label potential shadow boundaries (class 1) - areas with low intensity but high gradient
        shadow_boundary_mask = (gt_shading < shadow_threshold) & (gt_gradmag >= smooth_threshold)
        labels[shadow_boundary_mask] = 1
        
        # Everything else is normal/depth discontinuity (class 0) - default
        
        return labels

    def grouped_weighted_confusion_matrix(self, y_true, y_pred, y_pred_max, weights):
        """
        Compute weighted confusion matrix following SAW's approach
        """
        # Create 3x2 confusion matrix: [3 classes] x [2 predictions: smooth/non-smooth]
        conf_mx = np.zeros((3, 2), dtype=int)
        
        # For each true class
        for true_class in range(3):
            class_mask = (y_true == true_class)
            if not np.any(class_mask):
                continue
                
            # Count smooth predictions (y_pred == 1 means smooth)
            smooth_pred_count = np.sum(y_pred[class_mask] == 1)
            nonsmooth_pred_count = np.sum(y_pred[class_mask] == 0)
            
            conf_mx[true_class, 1] = smooth_pred_count      # Predicted as smooth
            conf_mx[true_class, 0] = nonsmooth_pred_count   # Predicted as non-smooth
        
        return conf_mx

    def get_pr_from_conf_mx(self, conf_mx, class_weights):
        """
        Calculate precision and recall from confusion matrix with class weights
        Following SAW's weighted approach
        """
        if conf_mx is None:
            return 0.0, 0.0
        
        # conf_mx shape: (3, 2) = [3 classes] x [predicted smooth/non-smooth]
        # class_weights = [1, 1, 2] for [NS-ND, NS-SB, S]
        
        weighted_tp = 0
        weighted_fp = 0  
        weighted_fn = 0
        weighted_tn = 0
        
        for class_idx in range(3):
            weight = class_weights[class_idx]
            
            # For smooth shading (class 2), being predicted as smooth (column 1) is correct
            if class_idx == 2:  # Smooth class
                tp = conf_mx[class_idx, 1] * weight  # Correctly predicted as smooth
                fn = conf_mx[class_idx, 0] * weight  # Incorrectly predicted as non-smooth
            else:  # Non-smooth classes (0, 1)
                fp = conf_mx[class_idx, 1] * weight  # Incorrectly predicted as smooth
                tn = conf_mx[class_idx, 0] * weight  # Correctly predicted as non-smooth
                
            weighted_tp += tp if class_idx == 2 else 0
            weighted_fp += fp if class_idx != 2 else 0
            weighted_fn += fn if class_idx == 2 else 0
            weighted_tn += tn if class_idx != 2 else 0
        
        # Calculate precision and recall
        if weighted_tp + weighted_fp > 0:
            precision = weighted_tp / (weighted_tp + weighted_fp)
        else:
            precision = 0.0
            
        if weighted_tp + weighted_fn > 0:
            recall = weighted_tp / (weighted_tp + weighted_fn)
        else:
            recall = 0.0
        
        return precision, recall

    def gen_pr_thres_list(self, thres_count):
        """Generate threshold list for PR curve"""
        # Generate logarithmically spaced thresholds for gradient magnitude
        min_thres = 1e-4
        max_thres = 2.0  # Higher max for gradient magnitudes
        thres_list = np.logspace(np.log10(min_thres), np.log10(max_thres), thres_count)
        return thres_list.tolist()

    def compute_gradmag(self, image_arr):
        """
        Compute gradient magnitude of image (following SAW's approach)
        """
        # Compute gradients using numpy
        grad_y, grad_x = np.gradient(image_arr)
        
        # Compute magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        return grad_mag

    def calculate_shading_from_hdr_albedo(self, hdr_image, albedo_image, epsilon=1e-6):
        """
        Calculate shading from HDR image and albedo
        """
        # Convert HDR from [-1, 1] to [0, 1]
        hdr_01 = (hdr_image + 1.0) / 2.0
        hdr_01 = torch.clamp(hdr_01, min=0.0, max=1.0)
        albedo_image = torch.clamp(albedo_image, min=0.0, max=1.0)
        
        # Calculate shading: I = A * S => S = (I · A) / (A · A)
        numerator = torch.sum(hdr_01 * albedo_image, dim=1, keepdim=True)
        denominator = torch.sum(albedo_image ** 2, dim=1, keepdim=True) + epsilon
        
        shading = numerator / denominator
        return shading


def test_shading_AP(model, encoder, vae, triplet_loader):
    """
    Test function for computing Average Precision specifically for shading prediction
    
    Args:
        model: Flow matching model
        encoder: Encoder model  
        vae: VAE model
        triplet_loader: DataLoader with HDR/Albedo/Shading triplets
        
    Returns:
        Average Precision (AP) score for shading prediction quality
    """
    print("============================= Validation ON SHADING PREDICTION ============================")
    
    # Initialize evaluator
    evaluator = ShadingAPEvaluator(model, encoder, vae)
    
    # Switch to evaluation mode
    evaluator.switch_to_eval()
    
    # Compute AP for shading prediction
    AP = evaluator.compute_pr_shading(
        triplet_loader=triplet_loader,
        thres_count=400,  # Same as original SAW
        bl_filter_size=10  # Same as original SAW
    )
    
    print(f"Current Shading Prediction AP: {AP:.6f}")
    
    # Switch back to training mode
    evaluator.switch_to_train()
    
    return AP


class ProcessedImagesTripletDataset(torch.utils.data.Dataset):
    """
    Dataset that loads HDR, albedo, and shading images from scene-based folder structure
    Structure: processed_images/scene_id/scene_id_type.png
    """
    def __init__(self, base_folder, transform=None):
        self.base_folder = Path(base_folder)
        self.transform = transform
        
        # Find all scene folders and create triplets
        self.triplets = self._create_triplets()
        
        print(f"Found {len(self.triplets)} valid triplets")
        
        if len(self.triplets) == 0:
            print("WARNING: No valid triplets found!")
            self._debug_folder_structure()
    
    def _debug_folder_structure(self):
        """Debug function to show folder structure"""
        print("Checking folder structure:")
        print(f"Base folder: {self.base_folder}")
        print(f"Base folder exists: {self.base_folder.exists()}")
        
        if self.base_folder.exists():
            subfolders = [d for d in self.base_folder.iterdir() if d.is_dir()]
            print(f"Found {len(subfolders)} subfolders")
            
            for i, subfolder in enumerate(subfolders[:5]):  # Show first 5
                print(f"  Subfolder {i+1}: {subfolder.name}")
                files = list(subfolder.glob("*.png"))
                print(f"    PNG files: {[f.name for f in files]}")
    
    def _create_triplets(self):
        """
        Create triplets by scanning scene folders
        """
        triplets = []
        
        if not self.base_folder.exists():
            print(f"Error: Base folder {self.base_folder} does not exist!")
            return triplets
        
        # Get all scene folders (should be numeric folders like 54, 63, etc.)
        scene_folders = [d for d in self.base_folder.iterdir() 
                        if d.is_dir() and d.name.replace('_', '').replace('-', '').isdigit()]
        scene_folders = sorted(scene_folders, key=lambda x: int(x.name.replace('_', '').replace('-', '')))
        
        print(f"Found {len(scene_folders)} scene folders")
        
        for scene_folder in scene_folders:
            scene_name = scene_folder.name
            
            # Look for the required files in this scene folder
            ldr_file = self._find_file(scene_folder, scene_name, ['_reconstructed_ldr'], ['.png'])
            albedo_file = self._find_file(scene_folder, scene_name, ['_albedo'], ['.png'])
            shading_file = self._find_file(scene_folder, scene_name, ['_shading'], ['.png'])
            
            if ldr_file and albedo_file and shading_file:
                triplets.append({
                    'scene_name': scene_name,
                    'ldr_path': ldr_file,
                    'albedo_path': albedo_file,
                    'shading_path': shading_file
                })
            else:
                missing = []
                if not ldr_file: missing.append("LDR")
                if not albedo_file: missing.append("Albedo") 
                if not shading_file: missing.append("Shading")
                print(f"Warning: Missing {', '.join(missing)} for scene '{scene_name}'")
                
        return triplets
    
    def _find_file(self, folder, scene_name, suffixes, extensions):
        """
        Find a file in the folder that matches the pattern: scene_name + suffix + extension
        """
        for suffix in suffixes:
            for ext in extensions:
                candidate = folder / f"{scene_name}{suffix}{ext}"
                if candidate.exists():
                    return candidate
        return None
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        
        # Load LDR image (equivalent to HDR in your original code)
        ldr_image = self._load_image(triplet['ldr_path'])
        if ldr_image is None:
            ldr_image = torch.zeros((3, 256, 256), dtype=torch.float32)
            print(f"Failed to load LDR image: {triplet['ldr_path']}")
        
        # Convert LDR from [0, 1] to [-1, 1] to match your HDR format
        ldr_image = ldr_image * 2.0 - 1.0
        
        # Load albedo image
        albedo_image = self._load_image(triplet['albedo_path'])
        if albedo_image is None:
            albedo_image = torch.zeros((3, 256, 256), dtype=torch.float32)
            print(f"Failed to load albedo image: {triplet['albedo_path']}")
        
        # Ensure albedo is in [0, 1] range
        if albedo_image.max() > 1.0:
            albedo_image = albedo_image / 255.0
        
        # Load shading image
        shading_image = self._load_image(triplet['shading_path'])
        if shading_image is None:
            shading_image = torch.zeros((1, 256, 256), dtype=torch.float32)
            print(f"Failed to load shading image: {triplet['shading_path']}")
        
        # Ensure shading is single channel and in [0, 1] range
        if shading_image.shape[0] > 1:
            shading_image = shading_image.mean(dim=0, keepdim=True)
        
        if shading_image.max() > 1.0:
            shading_image = shading_image / 255.0
        
        # Apply transforms if provided
        if self.transform:
            ldr_image = self.transform(ldr_image)
            albedo_image = self.transform(albedo_image)
            shading_image = self.transform(shading_image)
        
        return {
            'hdr': ldr_image,  # Using LDR as HDR equivalent
            'albedo': albedo_image,
            'shading': shading_image,
            'scene_name': triplet['scene_name']
        }
    
    def _load_image(self, file_path):
        """Load image and convert to tensor"""
        try:
            # Load image with PIL
            image = Image.open(file_path)
            
            # Handle grayscale images (like shading)
            if len(image.getbands()) == 1:
                image = image.convert('L')
                transform = transforms.Compose([
                    transforms.ToTensor()
                ])
                tensor = transform(image)  # Shape: [1, H, W]
            else:
                # RGB images
                image = image.convert('RGB')
                transform = transforms.ToTensor()
                tensor = transform(image)  # Shape: [3, H, W]
            
            return tensor
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None


def create_processed_images_loader(base_folder, batch_size=32, num_workers=4, shuffle=False):
    """
    Create DataLoader for the processed images dataset
    
    Args:
        base_folder: Path to processed_images folder
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader instance
    """
    dataset = ProcessedImagesTripletDataset(base_folder)
    
    if len(dataset) == 0:
        print("ERROR: No valid triplets found in dataset!")
        return None
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    return loader


# Modified main function to use the new dataset loader
def main_with_processed_images(args):
    """Modified main function that uses processed images dataset"""
    
    # Read configuration
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            return
    
    # Extract configuration sections
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Set random seeds for reproducibility
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if device == 'cuda:1':
        torch.cuda.manual_seed_all(seed)
    
    print("Loading models...")
    
    # Initialize models (same as your existing code)
    from models.vae import VAE
    from models.unet import Unet, Encoder
    
    vae = VAE(latent_dim=16).to(device)
    vae.eval()
    
    encoder = Encoder(im_channels=3).to(device)
    encoder.eval()
    
    model = Unet(im_channels=autoencoder_config['z_channels']).to(device)
    model.eval()
    
    # Load checkpoints (same as your existing code)
    vae_path = "checkpoints/epoch_41_best_autoencoder_model_checkpoint.pth"
    if os.path.exists(vae_path):
        print(f'Loading VAE checkpoint from {vae_path}')
        checkpoint_vae = torch.load(vae_path, weights_only=False, map_location=device)
        vae.load_state_dict(checkpoint_vae['model_state_dict'])
        print('VAE loaded successfully')
    else:
        print(f'VAE checkpoint not found at {vae_path}')
        return
    
    checkpoint_path = "checkpoints/epoch_191_flow_model_ckpt.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            from collections import OrderedDict
            def strip_orig_mod(state_dict):
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_key = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
                    new_state_dict[new_key] = v
                return new_state_dict
            
            model.load_state_dict(strip_orig_mod(checkpoint['model_state_dict']))
            encoder.load_state_dict(strip_orig_mod(checkpoint['encoder_state_dict']))
            print("Model and encoder loaded successfully")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    
    # Freeze parameters
    for param in vae.parameters():
        param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = False
    for param in encoder.parameters():
        param.requires_grad = False
    
    print("Creating processed images dataset...")
    
    # Create dataset loader using the new structure
    triplet_loader = create_processed_images_loader(
        base_folder=args.processed_folder,  # New argument
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False
    )
    
    if triplet_loader is None:
        print("Failed to create dataset loader!")
        return
    
    print(f"Dataset size: {len(triplet_loader.dataset)}")
    print(f"Number of batches: {len(triplet_loader)}")
    
    # Print some sample triplets for verification
    print("Sample triplets:")
    for i in range(min(5, len(triplet_loader.dataset.triplets))):
        triplet = triplet_loader.dataset.triplets[i]
        print(f"  {i+1}. Scene: {triplet['scene_name']}")
        print(f"     LDR: {triplet['ldr_path'].name}")
        print(f"     Albedo: {triplet['albedo_path'].name}")
        print(f"     Shading: {triplet['shading_path'].name}")
    
    # Run your existing evaluation or shading AP evaluation
    print("Computing Shading AP...")
    shading_ap = test_shading_AP(model, encoder, vae, triplet_loader)
    print(f"Shading Prediction AP: {shading_ap:.6f}")
    
    return shading_ap


# Updated argument parser
if __name__ == '__main__':
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Evaluate flow matching model on processed images dataset')
    parser.add_argument('--config', dest='config_path',
                       default='config/fine.yaml', type=str,
                       help='Path to configuration file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    parser.add_argument('--processed_folder', type=str, 
                       default="/mnt/zone/B/mithlesh/dataset/saw/processed_images",
                       help='Path to processed_images folder containing scene subfolders')
    
    # Handle Jupyter/IPython arguments
    import sys
    args, unknown = parser.parse_known_args(sys.argv[1:])
    
    try:
        main_with_processed_images(args)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Evaluation failed with error: {str(e)}")
        raise