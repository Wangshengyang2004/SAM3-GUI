# GUI for SAM3

**English** | [简体中文](README_CN.md)

A GUI tool for **SAM3** (Segment Anything with Concepts) video and image segmentation with **open-vocabulary text prompting** support.

## Key Features

- **Native SAM3 Support**: Built on SAM3 with full API compatibility
- **Text Prompting**: Segment objects using natural language (e.g., "person", "car", "red shoe")
- **Point Clicking**: Interactive refinement with positive/negative points
- **Box Prompts**: Draw bounding boxes to segment objects
- **Video Tracking**: Multi-object tracking across video frames with propagation directions
- **Multi-Object Management**: Track multiple objects independently with "Add New Mask"

## Installation

### Prerequisites

- Python 3.12 or higher
- PyTorch 2.7 or higher
- CUDA-compatible GPU with CUDA 12.6 or higher

### 1. Install SAM3

First, install [SAM3](https://github.com/facebookresearch/sam3):

```bash
# Create a new Conda environment
conda create -n sam3 python=3.12
conda activate sam3

# Install PyTorch with CUDA support
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

**Note for Blackwell RTX 50X0 GPUs:** These GPUs require torchvision to be compiled from source as pre-built wheels are not yet available.

# Clone the repository and install the package
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# Optional: Install additional dependencies for example notebooks or development
# For running example notebooks
pip install -e ".[notebooks]"

# For development
pip install -e ".[train,dev]"
```

### 2. Install GUI Dependencies

```bash
cd SAM3-GUI
pip install -r requirements.txt
```

**Note:** SAM3-GUI requires SAM3 to be installed first.

### 3. Download SAM3 Model

Download the SAM3 model checkpoint using ModelScope:

```bash
# Install modelscope if not already installed
pip install modelscope

# Download the model (will be saved to ~/sam3/model/sam3.pt by default)
python download_model.py

# Or specify a custom output directory
python download_model.py --output_dir /path/to/model/dir
```

The model will be downloaded from [ModelScope](https://www.modelscope.cn/models/facebook/sam3) and saved to `~/sam3/model/sam3.pt` by default.

**Alternative: HuggingFace Authentication (Optional)**

If you prefer to use HuggingFace or need to access private checkpoints:

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

## Acknowledgments

The app is modified based on [shape-of-motion](https://github.com/vye16/shape-of-motion/), upgraded from SAM2 to SAM3 with text prompting support.

![SAM3 GUI Video Mode](asset/sam3_1.png)

![SAM3 GUI Image Mode](asset/sam3_2.png)
