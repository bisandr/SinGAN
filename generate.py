import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import argparse
from SinGAN.functions import post_config, denorm, load_trained_pyramid
from SinGAN.models import *
import os

def outpaint_image(opt, Gs, Zs, reals, NoiseAmp):
    """
    Generates an outpainted image using a trained SinGAN model.
    
    Args:
        opt: SinGAN options.
        Gs: List of trained generators at each scale.
        Zs: List of noise maps.
        reals: List of real images at each scale.
        NoiseAmp: List of noise amplitudes at each scale.
    
    Returns:
        Expanded and outpainted image.
    """
    in_scale = len(Gs) - 1  # Use the finest scale for final output

    # Load input image and define new expanded canvas
    img = cv2.imread(opt.input_image, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (opt.n_size, opt.n_size))  # Ensure input size matches training
    img = np.transpose(img, (2, 0, 1)) / 255.0  # Normalize
    img = torch.tensor(img).unsqueeze(0).float().to(opt.device)

    # Define expanded output size
    new_size = opt.m_size  # Set the outpainted size
    expanded_img = torch.zeros(1, 3, new_size, new_size).to(opt.device)
    
    # Place input image in the center of the expanded image
    offset = (new_size - opt.n_size) // 2
    expanded_img[:, :, offset:offset + opt.n_size, offset:offset + opt.n_size] = img

    # Generate outpainted image using SinGAN
    for scale in range(in_scale + 1):
        z_in = expanded_img if scale == in_scale else torch.randn_like(expanded_img).to(opt.device)
        expanded_img = Gs[scale](z_in)
    
    return expanded_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help="Path to input image")
    parser.add_argument('--n_size', type=int, default=128, help="Size of the input image")
    parser.add_argument('--m_size', type=int, default=256, help="Size of the outpainted output")
    parser.add_argument('--model_dir', type=str, required=True, help="Path to trained SinGAN model")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    opt = parser.parse_args()
    post_config(opt)

    # Load trained SinGAN model
    Gs, Zs, reals, NoiseAmp = load_trained_pyramid(opt.model_dir, opt.device)

    # Generate outpainted image
    outpainted_img = outpaint_image(opt, Gs, Zs, reals, NoiseAmp)

    # Convert tensor to image and save
    outpainted_img = denorm(outpainted_img).detach().cpu().numpy()[0].transpose(1, 2, 0) * 255
    outpainted_img = np.clip(outpainted_img, 0, 255).astype(np.uint8)
    cv2.imwrite("outpainted_result.png", outpainted_img)
    
    print("✅ Outpainted image saved as outpainted_result.png")
