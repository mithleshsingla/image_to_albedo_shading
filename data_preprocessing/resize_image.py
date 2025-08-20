import os
import glob
import torch
import OpenEXR
import Imath
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

class EXRImageResizer:
    def __init__(self, source_dir, target_dir, target_size=256, use_gpu=True):
        """
        Initialize the EXR Image Resizer
        
        Args:
            source_dir (str): Source directory containing the original images
            target_dir (str): Target directory for resized images
            target_size (int): Target size for the final square images (default: 256)
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.im_size = target_size
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
    
    def read_exr_image(self, file_path):
        """
        Read EXR image file and convert to tensor
        
        Args:
            file_path (str): Path to the EXR file
            
        Returns:
            torch.Tensor: Image tensor in CHW format
        """
        try:
            import array
            
            # Open the EXR file
            exr_file = OpenEXR.InputFile(str(file_path))
            header = exr_file.header()
            
            # Get image dimensions
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
            
            # Stack channels to create HWC image then convert to CHW
            image = np.stack([r, g, b], axis=0)  # CHW format directly
            
            # Convert to tensor
            img_tensor = torch.from_numpy(image).float()
            
            exr_file.close()
            return img_tensor
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    
    def save_exr_image(self, img_tensor, file_path):
        """
        Save tensor as EXR image
        
        Args:
            img_tensor (torch.Tensor): Image tensor in CHW format
            file_path (str): Output file path
        """
        try:
            import array
            
            # Convert tensor to numpy (keep CHW format)
            if img_tensor.is_cuda:
                img_tensor = img_tensor.cpu()
            
            img_np = img_tensor.numpy().astype(np.float32)
            channels, height, width = img_np.shape
            
            # Create EXR header
            header = OpenEXR.Header(width, height)
            header['channels'] = {
                'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            
            # Create output file
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            exr_file = OpenEXR.OutputFile(str(file_path), header)
            
            # Extract and write channels
            r_channel = img_np[0, :, :].flatten()
            g_channel = img_np[1, :, :].flatten() if channels > 1 else np.zeros_like(r_channel)
            b_channel = img_np[2, :, :].flatten() if channels > 2 else np.zeros_like(r_channel)
            
            # Convert to the format expected by OpenEXR
            r_str = array.array('f', r_channel).tobytes()
            g_str = array.array('f', g_channel).tobytes()
            b_str = array.array('f', b_channel).tobytes()
            
            # Write channels
            exr_file.writePixels({'R': r_str, 'G': g_str, 'B': b_str})
            exr_file.close()
            
        except Exception as e:
            print(f"Error saving {file_path}: {e}")
    
    def resize_and_crop_image(self, img_tensor):
        """
        Resize image preserving aspect ratio and center crop to target size
        
        Args:
            img_tensor (torch.Tensor): Input image tensor in CHW format
            
        Returns:
            torch.Tensor: Resized and cropped image tensor
        """
        # Move to GPU if available
        img_tensor = img_tensor.to(self.device)
        
        # Get original dimensions
        c, h, w = img_tensor.shape
        
        # Calculate new dimensions preserving aspect ratio
        aspect_ratio = w / h
        if aspect_ratio > 1:  # Width > Height
            new_w = max(self.im_size, int(self.im_size * aspect_ratio))
            new_h = self.im_size
        else:  # Height >= Width
            new_h = max(self.im_size, int(self.im_size / aspect_ratio))
            new_w = self.im_size
        
        # Resize the image preserving aspect ratio
        img_tensor = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0),  # Add batch dimension
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Remove batch dimension
        
        # Center crop to get target size
        _, h, w = img_tensor.shape
        h_start = max(0, (h - self.im_size) // 2)
        w_start = max(0, (w - self.im_size) // 2)
        img_tensor = img_tensor[:, h_start:h_start+self.im_size, w_start:w_start+self.im_size]
        
        return img_tensor
    
    def process_images(self):
        """
        Process all EXR images in the source directory
        """
        # Find all EXR files matching the pattern
        pattern = str(self.source_dir / "**" / "*.exr")
        exr_files = glob.glob(pattern, recursive=True)
        
        if not exr_files:
            print(f"No EXR files found in {self.source_dir}")
            return
        
        print(f"Found {len(exr_files)} EXR files to process")
        
        # Process each file
        for file_path in tqdm(exr_files, desc="Processing images"):
            try:
                # Read the image
                img_tensor = self.read_exr_image(file_path)
                if img_tensor is None:
                    continue
                
                # Resize and crop
                processed_tensor = self.resize_and_crop_image(img_tensor)
                
                # Create output path maintaining directory structure
                relative_path = Path(file_path).relative_to(self.source_dir)
                output_path = self.target_dir / relative_path
                
                # Save the processed image
                self.save_exr_image(processed_tensor, output_path)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        print(f"Processing complete! Resized images saved to {self.target_dir}")

def main():
    parser = argparse.ArgumentParser(description="Resize EXR images while maintaining folder structure")
    parser.add_argument("source_dir", help="Source directory containing EXR images")
    parser.add_argument("target_dir", help="Target directory for resized images")
    parser.add_argument("--size", type=int, default=256, help="Target image size (default: 256)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    
    args = parser.parse_args()
    
    # Initialize and run the resizer
    resizer = EXRImageResizer(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        target_size=args.size,
        use_gpu=not args.no_gpu
    )
    
    resizer.process_images()

if __name__ == "__main__":
    main()