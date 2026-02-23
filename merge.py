import torch as th
from dataclasses import dataclass
import numpy as np
import cv2
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt

inverse_transparency = False

transparent_factor_max = 1.0
transparent_factor_min = 0.5

start_color = None
end_color = None

def get_mask_centers(img, mask):
    # only one mask contrains any value from 0 to 255
    # get all the pixels by mask
    # masked_pixels = img[np.tile(mask,(1,1,4))>0].reshape(-1,4)
    labels = np.unique(mask)
    centers = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    starts, ends =[], []
    
    # 如果标签数少于3，说明没有足够的目标
    if len(labels) < 3:
        return []
        
    # 假设最大的两个轮廓是我们的目标
    if len(contours) >= 2:
        areas = [cv2.contourArea(cnt) for cnt in contours]
        # 获取最大的两个轮廓索引
        sorted_indices = np.argsort(areas)[::-1][:2]
        main_contours = [contours[i] for i in sorted_indices]
        
        for cnt in main_contours:
            # 获取轮廓的边界框
            x, y, w, h = cv2.boundingRect(cnt)
            # 提取轮廓内的像素
            mask_single = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(mask_single, [cnt], 255)
            masked_pixels = img[mask_single > 0]
            
            # 计算重心
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
            centers.append((cX, cY))
        
    return centers

def _hex_to_bgr_array(hex_code: str):
    """hex_code: 'F5EEF4' or '#F5EEF4' -> np.array([B,G,R], float32)"""
    h = hex_code.lstrip('#')
    if len(h) != 6:
        raise ValueError(f"Bad hex color: {hex_code}")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return np.array([b, g, r], dtype=np.float32)



def draw_lines(img,mask, points):
    
    # BGR 起点亮粉色 (255, 0, 255), 终点黄色 (0, 255, 255)
    if not points:
        return
    (x1, y1), (x2, y2) = points
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

    # 线长度 -> 作为步数(至少 1)
    length = np.hypot(x2 - x1, y2 - y1)
    steps = max(1, int(length))  # 每像素一个小段，平滑
    thickness = 2

    # 处理通道 (BGR 或 BGRA)
    has_alpha = (img.shape[2] == 4)
    alpha_val = 255

    for i in range(steps):
        t0 = i / steps
        t1 = (i + 1) / steps
        # 两端点
        sx = int(round(x1 + (x2 - x1) * t0))
        sy = int(round(y1 + (y2 - y1) * t0))
        ex = int(round(x1 + (x2 - x1) * t1))
        ey = int(round(y1 + (y2 - y1) * t1))
        # 颜色插值
        c = start_color * (1 - t0) + end_color * t0
        if has_alpha:
            color = (int(c[0]), int(c[1]), int(c[2]), alpha_val)
        else:
            color = (int(c[0]), int(c[1]), int(c[2]))
        cv2.line(img, (sx, sy), (ex, ey), color, thickness, lineType=cv2.LINE_AA)
        # mask 仍然画成 1
        cv2.line(mask, (sx, sy), (ex, ey), (1.0), thickness, lineType=cv2.LINE_AA)

    return img


def compare_pixel_colors(masked_pixels, start_color, end_color, method='euclidean', stat_method='mode'):
    """
    比较提取像素区域的统计特征与起始/结束颜色的相似度
    
    Args:
        masked_pixels: (N, 3或4) 提取的像素值
        start_color: (3,) BGR起始颜色
        end_color: (3,) BGR结束颜色  
        method: 'euclidean', 'manhattan', 'cosine'
        stat_method: 'mean', 'median', 'mode' - 统计方法
    
    Returns:
        closer_to_start: bool, True表示更接近start_color
        distance_start: float, 到start_color的距离
        distance_end: float, 到end_color的距离
    """
    if len(masked_pixels) == 0:
        return False, float('inf'), float('inf')
        
    # 只取BGR通道
    pixels_bgr = masked_pixels[:, :3].astype(np.float32)
    start_bgr = start_color.astype(np.float32) 
    end_bgr = end_color.astype(np.float32)
    
    # 计算统计特征
    if stat_method == 'mean':
        representative_color = np.mean(pixels_bgr, axis=0)
    elif stat_method == 'median':
        representative_color = np.median(pixels_bgr, axis=0)
    elif stat_method == 'mode':
        # 对每个通道分别计算众数（量化到最近整数）
        representative_color = np.zeros(3)
        for i in range(3):
            channel_values = np.round(pixels_bgr[:, i]).astype(int)
            unique_vals, counts = np.unique(channel_values, return_counts=True)
            mode_idx = np.argmax(counts)
            representative_color[i] = unique_vals[mode_idx]
    else:
        raise ValueError(f"Unknown stat_method: {stat_method}")
    
    # 计算代表性颜色与目标颜色的距离
    if method == 'euclidean':
        dist_start = np.sqrt(np.sum((representative_color - start_bgr)**2))
        dist_end = np.sqrt(np.sum((representative_color - end_bgr)**2))
    elif method == 'manhattan':
        dist_start = np.sum(np.abs(representative_color - start_bgr))
        dist_end = np.sum(np.abs(representative_color - end_bgr))
    elif method == 'cosine':
        def cosine_dist(a, b):
            dot = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            return 1 - dot / (norm_a * norm_b + 1e-8)
        dist_start = cosine_dist(representative_color, start_bgr)
        dist_end = cosine_dist(representative_color, end_bgr)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    closer_to_start = dist_start < dist_end
    return closer_to_start, dist_start, dist_end

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
    output_path = f"output/{name}_merged.png"
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

    transparent_factors = np.linspace(transparent_factor_min, transparent_factor_max, num=len(images))
    mix_factor = 1
    
    r_id = [81, 82, 83, 84 ,88]
    
    for idx in (range(len(images))):  # TODO if transparency is not inversed
        
        img = images[idx].astype(np.float32)
        mask = masks[idx] #np.clip(masks[idx], None ,1) # Normalize mask to [0, 1]
        # centers = get_mask_centers(masks[idx])
        centers = get_mask_centers(img,mask)

        mask = np.expand_dims(mask, axis=-1)  # Add a channel dimension to match (H, W, 1)

        # Make the masked region transparent by reducing alpha channel
        # transparent_layer = img * mask
        # transparent_layer[..., 3] *= (idx + 1) / len(images)  # Gradual transparency
        trans = transparent_factors[idx]
        # if inverse_transparency:
        #     # trans = (1-(idx + 1) / len(images)) ** transparent_factor_max
        #     trans = transparent_factors[len(images) - 1 - idx]
        print(f"Frame {idx}: transparency factor {trans}")
        if len(centers)>= 2 and idx % 1 == 0:
            if idx >= len(images) - 4 or idx in r_id:
                centers = list(reversed(centers))
            draw_lines(img, mask,centers)
        
        mask = np.clip(mask, None ,1) # Normalize mask to [0, 1]
        new = mask * mix_factor
        remain = 1 - new
        
        # each part
        new_part = img * new
        remain_part = background * new
        
        # Blend the transparent masked area onto the combined image
        combined_image = combined_image * remain + new_part * trans + remain_part * (1-trans)

    # Clip values to valid range and convert back to uint8
    combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
    return combined_image

def save_frames(cfg):
    indices = np.arange(cfg.start, cfg.end, cfg.interval, dtype=int)
    frames = get_video_frames(f"{cfg.name}_frames", indices)
    # save frames in a folder
    os.makedirs(f"videos/{cfg.name}_frames", exist_ok=True)
    for i, frame in zip(indices, frames):
        cv2.imwrite(f"videos/{cfg.name}_frames/{str(i).zfill(5)}.png", frame)
    return frames
    

def main():
    global inverse_transparency, transparent_factor_min, transparent_factor_max, start_color, end_color

    # Set colors for trajectory lines
    end_color = _hex_to_bgr_array("D0BA72")
    start_color = _hex_to_bgr_array("F5EEF4")

    inverse_transparency = False
    transparent_factor_min = 0.5
    transparent_factor_max = 1.0

    # Configure for Cam1_color data
    # You can adjust start, interval, end as needed
    h = Config(start=1, interval=3, end=150, name="Cam1_color")

    print(f"Processing frames {h.start} to {h.end} with interval {h.interval}")
    merge(h)

if __name__ == "__main__":
    main()