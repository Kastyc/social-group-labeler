# social-group-labeler
A local web tool for labeling social groups in pedestrian video footage, built for the UT Austin Social World Model research project.

## Workflow

### 1. Install dependencies
```bash
pip install -r requirements.txt
pip install rosbags  # for rosbag frame extraction only
```

### 2. Extract frames from a rosbag
```bash
python extract_frames.py path/to/recording.bag frames/ --fps 5
```
Frames are saved as PNGs in `frames/`. Increase `--fps` to get more frames (e.g. `--fps 30` for all frames).

### 3. Run the labeling web app
```bash
FRAMES_DIR=frames uvicorn main:app --reload
```
Open `http://localhost:8000` in your browser. On first run, it auto-detects people in all frames using YOLO and caches results to `frames/detections_cache.json`.

In the UI, click people to assign them to groups, then save annotations. Annotations are stored in `frames/annotations.json`.

### 4. Extract features from your annotations
```bash
python extract_features.py
```
Outputs `features.csv` — one row per person pair per frame, with distance and group label.

### 5. Train the classifier
```bash
python train_and_visualize.py --train
```
Trains a Random Forest on `features.csv` and saves the model to `group_classifier.pkl`.

### 6. Run predictions
```bash
# Single frame
python train_and_visualize.py --predict frames/my_frame.png

# All frames
python train_and_visualize.py --predict-all predictions/
```
Outputs images with colored bounding boxes per predicted group.

When training try to compare with a different rosbag than what you've used to train initially.
