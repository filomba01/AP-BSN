import os
import random
from PIL import Image
import numpy as np
import torch

def test_pytorch():
    print("PyTorch version:", torch.__version__)
    print("Is CUDA available?", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA current device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("Running on CPU.")

    # Test a simple tensor operation
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    z = x + y
    print("Tensor addition result:", z)

def add_noise(image_rgb, peak=255.0, noise_boost=10, photon_scale=1000):
    
    # Normalize to [0, 1] range (assuming 8-bit images)
    image_rgb_float = image_rgb.float() / 255.0

    # Sentinel-2 band-specific SNRs (R, G, B = B04, B03, B02)
    snrs = torch.tensor([230, 249, 214], dtype=torch.float32)

    # Add Poisson noise
    poisson = torch.poisson(image_rgb_float * photon_scale) / photon_scale

    # Add Gaussian noise for each band based on the avg pixel value for that band
    avg_per_band = torch.mean(image_rgb_float, dim=(1, 2))  # Average across spatial dimensions
    std_dev = avg_per_band / snrs
    
    # Expand std_dev to match image dimensions for broadcasting
    std_dev_expanded = std_dev.view(-1, 1, 1) * noise_boost
    gaussian = torch.normal(0, std_dev_expanded.expand_as(image_rgb_float))
    
    # Combine noises
    noisy = poisson + gaussian
    
    # Scale back to [0, 255] range
    synthesized_img = noisy * 255.0
    
    # Set nlf as the average standard deviation across bands (scaled to [0,255])
    nlf = torch.mean(std_dev * noise_boost).item() * 255.0
    return torch.clamp(synthesized_img, 0, 255).to(torch.uint8), gaussian, poisson, nlf


if __name__ == "__main__":
    # Paths
    input_folder = "dataset/prep/SatelliteImages_s512_o128/test/CL/"
    output_folder = "dataset/prep/SatelliteImages_s512_o128/test/noisy/"
    os.makedirs(output_folder, exist_ok=True)

    # List image files
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.tif'))]
    if not image_files:
        print("No images found in", input_folder)
        exit(1)

    # inject noise into images 
    for i, img_name in enumerate(image_files):
        # Load image and convert to RGB
        img_path = os.path.join(input_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        # Convert to torch tensor (C, H, W)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        # Apply noise
        noisy_img, gaussian, poisson, nlf = add_noise(img_tensor, noise_boost=10, photon_scale=1000)

        # Save noisy image
        noisy_img_np = noisy_img.permute(1, 2, 0).cpu().numpy()
        noisy_img_pil = Image.fromarray(noisy_img_np.astype(np.uint8))
        noisy_img_name = f"{img_name}_N.png"
        noisy_img_pil.save(os.path.join(output_folder, noisy_img_name))
        print(f"Noisy image saved to {os.path.join(output_folder, noisy_img_name)}")
