# GUI for SAM3

A GUI tool for **SAM3** (Segment Anything with Concepts) video and image segmentation with **open-vocabulary text prompting** support.

## Key Features

- **Text Prompting (NEW!)**: Segment objects using natural language (e.g., "person", "car", "red shoe")
- **Point Clicking**: Interactive refinement with positive/negative points
- **Video Tracking**: Multi-object tracking across video frames
- **Open-Vocabulary**: Detect 4M+ different object types
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

**Note:** SAM3-GUI requires SAM3 to be installed first. The GUI requirements include:
- Gradio 4.37.2
- imageio, imageio-ffmpeg
- loguru, fastapi

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
│   │   ├── 00000.png
│   │   ├── 00001.png
│   │   └── ...
│   └── seq2/
│       └── ...
└── masks/           # Output directory for saved masks
    ├── seq1/
    └── seq2/
```

### Text Prompting (NEW!)

1. Load a video or image sequence
2. Click **"Get SAM features"** to initialize SAM3
3. **Enter a text description** (e.g., "person walking", "car", "dog")
4. Click **"Segment with Text"** to detect all matching objects
5. Click **"Submit mask for tracking"** to propagate through video

**Text Prompt Examples:**
- "person"
- "car"
- "red shirt"
- "player in white"
- "basketball hoop"

### Point-Based Refinement

Traditional SAM-style interaction is still supported:

1. Click **"Toggle positive"** to add inclusion points (green)
2. Click **"Toggle negative"** to add exclusion points (red)
3. Click on the image to place points
4. Click **"Submit mask for tracking"** when satisfied

### Video Processing Workflow

1. **Select or extract frames:**
   - Choose an MP4 from the video list and extract frames
   - OR select a pre-extracted image directory

2. **Initialize SAM3:**
   - Click "Get SAM features" to initialize the model

3. **Add prompts:**
   - Use text prompts to segment objects by description
   - OR use point clicks for manual refinement

4. **Track through video:**
   - Click "Submit mask for tracking" to propagate masks
   - Review the tracked video output

5. **Save masks:**
   - Click "Save masks" to export masks to the output directory

## How Text Prompting Works

SAM3 introduces open-vocabulary segmentation powered by vision-language models:

1. **Text Encoder**: Your prompt is encoded into a semantic embedding
2. **Detector**: SAM3's detector finds objects matching the text description
3. **Tracker**: The detected objects are tracked through the video

**Advantages over point clicking:**
- No manual annotation required
- Can find all instances of a concept at once
- Works with 4M+ different object descriptions
- Can combine text with points for refinement

## Tips for Best Results

### Text Prompting
- **Be specific**: "person in red shirt" works better than "person"
- **Use simple descriptions**: Common nouns work best (person, car, dog)
- **Combine with points**: Use text for initial detection, then refine with points
- **Try variations**: If "car" doesn't work, try "vehicle" or "automobile"

### Point Clicking
- Use positive points (green) on the object to segment
- Use negative points (red) on background or other objects
- Start with 1-3 positive points, add negatives as needed
- Points can be added to multiple objects for multi-object tracking

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

1. Ensure SAM3 features are initialized (click "Get SAM features")
2. Try simpler or more common descriptions
3. Check that the object is clearly visible in the current frame
4. Combine with point clicks for difficult cases

## Acknowledge

The app is modified based on [shape-of-motion](https://github.com/vye16/shape-of-motion/), upgraded from SAM2 to SAM3 with text prompting support.

![gradio interface](asset/gradio_interface.png)
