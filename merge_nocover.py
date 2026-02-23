import torch as th
from dataclasses import dataclass
import numpy as np
import cv2
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt

inverse_transparency = False

def get_video_frames(video_path, indices):
    """
    Load frames from image directory instead of video file
    video_path: path relative to data_root/images/
    """
    base_path = "/home/wsy/SAM3-GUI/data_root/images"
    full_path = f"{base_path}/{video_path}"

    frames = []
    for idx in indices:
        img_path = f"{full_path}/{str(idx).zfill(5)}.jpg"
        frame = cv2.imread(img_path)
        if frame is not None:
            frames.append(frame)
        else:
            print(f"Warning: Could not read {img_path}")

    print(f"Loaded {len(frames)} frames from {full_path}")
    return frames

def get_masks(name, indices):
    """
    Load masks from data_root/masks directory
    name: path relative to data_root/masks/
    """
    base_path = "/home/wsy/SAM3-GUI/data_root/masks"
    masks = []
    for i in indices:
        mask = np.load(f"{base_path}/{name}/{str(i).zfill(5)}.npy")
        masks.append(mask)

    print(f"Loaded {len(masks)} masks from {base_path}/{name}")
    return masks


@dataclass
class Config:
    start: int = 0
    interval: int = 1
    end: int = 10
    name: str = ""
    background: str = None

def merge(cfg: Config):
    # Create output directory
    os.makedirs("output", exist_ok=True)

    indices = np.arange(cfg.start, cfg.end, cfg.interval, dtype=int)
    frames = get_video_frames(f"{cfg.name}", indices)
    masks = get_masks(cfg.name, indices=indices)

    # Load background if specified, otherwise use None
    background = None
    if cfg.background:
        background = get_background(cfg.name, cfg.background)

    img = merge_masked_images(frames, masks, background=background)

    # Save output
    name = cfg.name.split("/")[-1]
    output_path = f"output/{name}_merged_nocover.png"
    cv2.imwrite(output_path, img)
    print(f"Saved merged image to {output_path}")

def get_background(name, back):
    if back is None:
        return None
    # Try loading from images directory
    img = cv2.imread(f"/home/wsy/SAM3-GUI/data_root/images/{name}/{back}")
    if img is None:
        print(f"Warning: Could not load background {back}")
    return img
    
    
def merge_masked_images(images, masks, background=None):
    """
    Merge masked parts of n images into the last image with transparency
    to illustrate movement (e.g., of a drone).

    Args:
        images (list of PIL.Image): List of n images in sequence.
        masks (list of PIL.Image): List of n binary masks (same size as images),
                                   where white (255) indicates the region to extract.

    Returns:
        PIL.Image: Combined image showing transparent masked areas.
    """
    """
    Merge masked parts of n images (NumPy arrays) into the last image, 
    making masked areas transparent, to illustrate movement (e.g., of a drone).

    Args:
        images (list of np.ndarray): List of n images as NumPy arrays with shape (H, W, 3 or 4).
        masks (list of np.ndarray): List of n masks as NumPy arrays with shape (H, W), 
                                    where values are 0 or 255 (binary mask).

    Returns:
        np.ndarray: Combined image as a NumPy array with shape (H, W, 4) (RGBA).
    """
    if len(images) != len(masks):
        raise ValueError("Number of images and masks must be the same")

    # Convert images to RGBA if they are RGB
    # inverse_transparency = True
    images = [np.dstack((img, np.full(img.shape[:2], 255))) if img.shape[-1] == 3 else img for img in images]
    if inverse_transparency:
        images = list(reversed(images))
        masks = list(reversed(masks))
    # Use the last image as the base (convert to float for blending)
    combined_image = images[-1].astype(np.float32)
    darkening_factor = 1.0
    if background is None:
        background = images[-1].astype(np.float32)
    else:
        background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA).astype(np.float32)
    # background[..., :3] *= darkening_factor  # Darken the background
    # background = (background * darkening_factor).astype(np.float32)
    combined_image = background
    # make the background darker
    # background[..., 0] *= 0.5
    
    transparent_factor_max = 1.0

    transparent_factors = np.linspace(0, transparent_factor_max, num=len(images))
    mix_factor = 1
    
    for idx in (range(len(images))):  # TODO if transparency is not inversed
        
        img = images[idx].astype(np.float32)
        mask = np.clip(masks[idx], None ,1) # Normalize mask to [0, 1]
        
        mask = np.expand_dims(mask, axis=-1)  # Add a channel dimension to match (H, W, 1)
        mask = 1-((background * mask-img*mask)<=10).all(axis=-1, keepdims=True).astype(np.int8)
        mask = np.tile(mask, (1,1,4))
        # Make the masked region transparent by reducing alpha channel
        # transparent_layer = img * mask
        # transparent_layer[..., 3] *= (idx + 1) / len(images)  # Gradual transparency
        trans = transparent_factors[idx]
        # if inverse_transparency:
        #     # trans = (1-(idx + 1) / len(images)) ** transparent_factor_max
        #     trans = transparent_factors[len(images) - 1 - idx]
        new = mask * mix_factor
        remain = 1 - new
        # Blend the transparent masked area onto the combined image
        # combined_image = combined_image * remain + img * new * trans + background * new * (1-trans)
        combined_image = combined_image * remain + img * new * trans + background * new * (1-trans)

    # Clip values to valid range and convert back to uint8
    combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
    return combined_image

def save_frames(cfg):
    indices = np.arange(cfg.start, cfg.end, cfg.interval, dtype=int)
    frames = get_video_frames(f"{cfg.name}_frames", indices)
    # save frames in a folder
    os.makedirs(f"videos/{cfg.name}_frames", exist_ok=True)
    # remove all images end with png in the folder
    for f in os.listdir(f"videos/{cfg.name}_frames"):
        if f.endswith(".png"):
            os.remove(os.path.join(f"videos/{cfg.name}_frames", f))
    for i, frame in zip(indices, frames):
        cv2.imwrite(f"videos/{cfg.name}_frames/{str(i).zfill(5)}.png", frame)
    print(f"Saved frames to videos/{cfg.name}_frames/")
    return frames
    

def main():
    global inverse_transparency
    inverse_transparency = False

    # Configure for Cam1_color data (nocover version)
    h = Config(start=1, interval=3, end=150, name="Cam1_color")

    print(f"Processing frames {h.start} to {h.end} with interval {h.interval}")
    merge(h)

if __name__ == "__main__":
    main()