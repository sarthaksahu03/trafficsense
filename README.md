# TrafficSense: ALPR & Violation Detection System

TrafficSense is a fully integrated computer vision pipeline operating on raw CCTV footage. It delivers modular subsystems for Speed Estimation and Red Light Violation mapping wrapped inside an asynchronous GUI application. 
Powered by YOLOv8 for object localization and EasyOCR for License Plate parsing.

## Table of Contents
1. [Features](#features)
2. [Operating System Support](#operating-system-support)
3. [Installation](#installation)
4. [Usage Instructions](#usage-instructions)
5. [Calibration Workflows](#calibration-workflows)
6. [Generating Synthetic Datasets](#generating-synthetic-datasets)

## Features
- **Red-Light Detection**: Mathematically evaluates vehicle crossings relative to an active traffic signal array without requiring distinct trained YOLO classification models for state mappings.
- **Speed Estimation**: Derives accurate 2D projection speeds computing physical displacement within a standardized four-point perspective warp.
- **Dynamic Dashboard**: Isolates violation instances and captures localized bounding boxes for immediate inspection tracking via the interactive CustomTkinter interface.
- **Evidentiary CSV Logging**: Preserves timestamped violation data sequentially safely mitigating overlapping OCR deduplication requests.

## Operating System Support
This software acts purely asynchronously via standard OpenCV handling; however, GPU execution vastly accelerates DeepSORT and YOLO throughput speeds.

### Windows
- Recommended Environment: Anaconda/Miniconda 
- Prerequisite: CUDA Toolkit 11.8+ installed natively if utilizing NVIDIA GPUs (otherwise defaults to CPU bottleneck).
- Tkinter works natively out of the box.

### macOS
- Recommended Environment: Homebrew & Miniforge
- Prerequisite: Apple Silicon (M1/M2/M3) automatically leverages MPS (Metal Performance Shaders) backend on native torch installs, giving large speedups over CPU.
- X11 or Wayland is not necessary.

### Linux (Ubuntu/Fedora/Debian)
- Recommended Environment: Python Virtual Environments (`venv`) or standard Conda deployments.
- Note for Wayland Users: PyQt bindings can conflict with default Wayland variables forcing `OpenCV` to fail drawing windows. You may either install `qtwayland5` or fallback OpenCV outputs setting: `export QT_QPA_PLATFORM=xcb`.
- NVIDIA proprietary drivers and `cudatoolkit` are required for accelerated throughputs matching Windows limits.

## Installation

1. **Clone the Directory**
   ```bash
   git clone https://github.com/sarthaksahu03/trafficsense.git
   cd trafficsense
   ```

2. **Create Virtual Environment**
   ```bash
   conda create -n tsense python=3.10 -y
   conda activate tsense
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > **Note on PyTorch:** For GPU support on Windows/Linux, you MUST fetch the specific Compute build explicitly instead of standard pip defaults. Visit [pytorch.org](https://pytorch.org/get-started/locally/) to execute the exact environment flags corresponding to your hardware. 

## Usage Instructions

Initialize the application dashboard:
```bash
python gui.py
```

1. Navigate using the **Sidebar Setup** controls.
2. Toggle your execution goal between **"Speed"** constraints or **"Red Light"** constraints.
3. Import your CCTV video `.mp4` using **Select Input Video**.
4. Configure limits safely dragging the thresholds sliders.
5. Calibrate the logic constraints clicking the orange `Calibrate` buttons sequentially.

### Calibration Workflows
Depending on your operational mode, OpenCV prompts coordinate mappings locking to your source video scaling.

* **Speed ROI (4 Clicks)**: Form a quadrilateral mapping the observable road. Points *must* anchor clockwise (Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left).
* **Stop Line (2 Clicks)**: Define a virtual barrier perpendicular to approaching lanes (Left side point -> Right side point).
* **Traffic Light ROI (2 Clicks)**: Tightly encase the rectangular bounds of the traffic light cluster (Top-Left -> Bottom-Right corner drops).

## Generating Synthetic Datasets
We ship an external utility designed purely for data synthesis over blank footage feeds, automating a custom 5s -> 2s -> 5s traffic light state loop.
```bash
python generate_tl_dataset.py --input raw_camera.mp4 --output labeled_dataset.mp4
```
It leverages existing standard 2-Click initialization bounds ensuring workflow parity preventing calibration confusion.
