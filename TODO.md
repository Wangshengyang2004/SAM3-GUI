# SAM3-GUI Feature TODO

## Current Implementation Status

### Video Mode
- [x] Text prompts (natural language object detection)
- [x] Point prompts (positive/negative clicks)
- [x] Multi-object tracking with unique IDs
- [x] Video propagation (forward)
- [x] Frame extraction from video files
- [x] Mask export (per-frame .npy and colored images)
- [x] Tracked video output (.mp4)

### Image Mode
- [x] Text prompts ("Find All" mode)
- [x] Point prompts (positive/negative clicks)
- [x] Box prompts (bounding box selection)
- [x] Multi-mask output with score ranking
- [x] Mask export (.npy)

---

## Missing Features

### High Priority (Common Use Cases)

- [ ] **Video Mode: Box Prompt**
  - SAM3 supports drawing boxes on frames to select objects
  - API: `add_prompt` with box coordinates

- [ ] **Confidence Threshold UI Control**
  - Currently hardcoded to 0.3 in image mode
  - SAM3 API: `set_confidence_threshold(threshold, state)`
  - Add slider in both Video and Image modes

- [ ] **Remove Tracked Object**
  - Delete specific object from tracking
  - SAM3 API: `handle_request(type="remove_object", obj_id=...)`

### Medium Priority (Advanced Features)

- [ ] **Multi-Frame Prompt Support**
  - Add prompts on any frame, not just current frame
  - Currently limited to prompting only the displayed frame
  - SAM3 supports `frame_index` parameter for any frame

- [ ] **Reverse Propagation**
  - Propagate masks backward in time
  - SAM3 API: `propagate_in_video(..., reverse=True)`
  - Add UI toggle for propagation direction (forward/backward/both)

- [ ] **Image Mode: Negative Box**
  - Exclude regions with negative bounding boxes
  - SAM3 API: `add_geometric_prompt(box, label=False, state)`

### Low Priority (Specialized Features)

- [ ] **Mask Prompt Input**
  - Use existing mask as input to guide segmentation
  - Useful for refinement workflows
  - SAM3 supports mask inputs in both image and video modes

- [ ] **Batch Image Inference**
  - Process multiple images at once
  - See `sam3_image_batched_inference.ipynb` for reference

- [ ] **Agent Mode Integration**
  - AI Agent integration for complex workflows
  - See `sam3_agent.ipynb` for reference

---

## Bug Fixes Completed

- [x] **"No point prompt was applied" error** (2024-02)
  - Root cause: Missing `propagate_preflight=True` in `propagate_in_video` call
  - Fix: Added parameter to consolidate temp outputs before propagation

- [x] **Segmentation Preview 404** (2024-02)
  - Root cause: `queue=False` incompatible with Gradio 6.x
  - Fix: Removed `queue=False` from select event handlers

- [x] **Port binding error on startup** (2024-02)
  - Root cause: Fixed port with no fallback
  - Fix: Auto-increment port if occupied (8890 → 8891 → ...)

---

## API Reference

### Video Mode Key Methods
```python
# Start session
handle_request(type="start_session", resource_path=video_path)

# Add prompts
handle_request(type="add_prompt", session_id=..., frame_index=...,
               text="object" | points=[[x,y]] | point_labels=[1])

# Remove object
handle_request(type="remove_object", session_id=..., obj_id=...)

# Propagate
handle_stream_request(type="propagate_in_video", session_id=...,
                      propagation_direction="both"|"forward"|"backward")

# Tracker methods
tracker.add_new_points(inference_state, frame_idx, obj_id, points, labels)
tracker.propagate_in_video(state, start_frame_idx, max_frame_num, reverse, propagate_preflight)
```

### Image Mode Key Methods
```python
# Initialize
processor = Sam3Processor(model, confidence_threshold=0.3)
state = processor.set_image(image)

# Prompts
processor.set_text_prompt(prompt, state)
processor.add_geometric_prompt(box, label=True|False, state)

# Point-based (via model directly)
model.predict_inst(point_coords, point_labels, multimask_output=True)

# Configuration
processor.set_confidence_threshold(threshold, state)
processor.reset_all_prompts(state)
```
