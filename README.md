# GUI for SAM3

A GUI tool for **SAM3** (Segment Anything with Concepts) video and image segmentation with **open-vocabulary text prompting** support.

## Key Features

- **Text Prompting**: Segment objects using natural language (e.g., "person", "car", "red shoe")
- **Point Clicking**: Interactive refinement with positive/negative points
- **Box Prompts**: Draw bounding boxes to segment objects
- **Video Tracking**: Multi-object tracking across video frames with propagation directions
- **Multi-Object Management**: Track multiple objects independently with "Add New Mask"
- **Open-Vocabulary**: Detect 4M+ different object types
- **Frame-Specific Points**: Points only appear on their designated frames
- **Auto-Download**: SAM3 model automatically downloads from HuggingFace

## Installation

### 1. Install SAM3

First, install [SAM3](https://github.com/facebookresearch/sam3):

```bash
# Install PyTorch with CUDA support
pip install torch>=2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126

# Install SAM3
cd /path/to/sam3
pip install -e .
```

### 2. Install GUI Dependencies

```bash
cd SAM3-GUI
pip install -r requirements.txt
```

**Note:** SAM3-GUI requires SAM3 to be installed first.

### 3. HuggingFace Authentication (Optional)

If you need to access private checkpoints, authenticate with HuggingFace:

```bash
huggingface-cli login
```

## Usage

### Starting the GUI

```bash
python cli.py --root_dir [data_root]
```

Optional arguments:
- `--port`: Port number (default: 8890)
- `--server_name`: Server address (default: 127.0.0.1; use 0.0.0.0 for external access)
- `--vid_name`: Video subdirectory name (default: "videos")
- `--img_name`: Image subdirectory name (default: "images")
- `--mask_name`: Mask subdirectory name (default: "masks")

### Data Organization

Organize your data in the following structure:

```
data_root/
├── videos/          # For MP4 files to extract frames
│   ├── seq1.mp4
│   └── seq2.mp4
├── images/          # For pre-extracted frame sequences
│   ├── seq1/
│   │   ├── frame_00000.png
│   │   ├── frame_00001.png
│   │   └── ...
│   └── seq2/
│       └── ...
└── masks/           # Output directory for saved masks
    ├── seq1/
    │   ├── frame_00000.png
    │   ├── frame_00000.npy
    │   └── ...
    └── seq2/
```

## Video Mode Workflow

### 1. Load Frames

- **Option A**: Select a video file and extract frames
  - Choose video from dropdown
  - Set start/end time, FPS, and height
  - Click "Extract Frames"

- **Option B**: Load pre-extracted frames
  - Select a frame folder from the dropdown
  - Click "Load Selected Frames"

### 2. Add Prompts

Choose from three prompt types:

#### **Text Prompts**
1. Enter a text description (e.g., "person", "car", "red shirt")
2. Click "Detect with Text"
3. Use "Add New Mask" to segment additional objects

#### **Point Prompts**
1. Click "+ Positive" for inclusion points (green)
2. Click "- Negative" for exclusion points (red)
3. Click on the **Output Image** to place points
4. Points are frame-specific - switch frames to add points on other frames
5. View the "Added Points" table to see all points across frames

#### **Box Prompts**
1. Click "Segment Box"
2. Click two corners on the frame to draw a box
3. The object inside will be segmented

### 3. Manage Objects

- **View Tracked Objects**: Dropdown shows all detected objects (0, 1, 2, ...)
- **Remove Objects**: Select an object and click "Remove Selected Object"

### 4. Track Through Video

1. Select propagation direction:
   - **Forward**: Propagate from current frame to end
   - **Backward**: Propagate from current frame to start
   - **Both**: Propagate in both directions (default)

2. Click "Track All Frames"
3. View the tracked video output
4. Use frame slider to review results on individual frames

### 5. Save Masks

- Mask save path is auto-generated: `{root_dir}/masks/{sequence_name}/`
- Click "Save Masks" to export masks as PNG and NPZ files

## Image Mode Workflow

Single image segmentation with three modes:

### **Find All Mode**
1. Enter a text prompt (e.g., "shoe", "person", "car")
2. Adjust confidence threshold (0.0-1.0)
3. Click "Find All" to detect all matching objects

### **Box Mode**
1. Click "Segment Box"
2. Draw a box by clicking two corners on the image
3. The object inside will be segmented

### **Point Mode**
1. Click "+ Positive" or "- Negative"
2. Click on the image to place points
3. Use "Remove Point by Index" to delete specific points

## Tips for Best Results

### Text Prompting
- **Be specific**: "person in red shirt" works better than "person"
- **Use simple descriptions**: Common nouns work best (person, car, dog)
- **Try variations**: If "car" doesn't work, try "vehicle" or "automobile"

### Point Clicking
- Use positive points (green) on the object to segment
- Use negative points (red) on background or other objects
- Start with 1-3 positive points, add negatives as needed
- **Important**: Click on the **Output Image** (right side) when adding multiple points

### Multi-Object Tracking
1. Segment your first object (text/point/box)
2. Click "Add New Mask" to increment mask index
3. Segment the second object
4. Click "Track All Frames" to track all objects together

## Troubleshooting

### Model Download Issues

SAM3 auto-downloads from HuggingFace. If you encounter issues:

```bash
# Check HuggingFace connection
huggingface-cli whoami

# For private checkpoints, ensure you have access
# and are logged in with huggingface-cli login
```

### CUDA Out of Memory

If you run out of GPU memory:

```bash
# Use a smaller batch size or process fewer frames at once
# Consider extracting fewer frames from your video
```

### Text Prompt Not Working

1. Ensure frames are loaded first
2. Try simpler or more common descriptions
3. Check that the object is clearly visible in the current frame
4. Combine with point clicks for difficult cases

### Points Not Remembering

- Points are frame-specific - each frame has its own set of points
- Use the "Added Points" table to see all points across all frames
- When switching frames, only points for that frame are displayed

### Dtype Mismatch Error

If you see "mat1 and mat2 must have the same dtype" errors:
- Ensure you're using PyTorch with CUDA support
- The code now handles BFloat16/Float32 dtype mismatches automatically

## Acknowledgments

The app is modified based on [shape-of-motion](https://github.com/vye16/shape-of-motion/), upgraded from SAM2 to SAM3 with text prompting support.

![gradio interface](asset/gradio_interface.png)
