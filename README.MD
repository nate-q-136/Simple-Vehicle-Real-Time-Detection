
# Vehicles Real-time Detection

This project is a real-time vehicle detection application using YOLOv8 and OpenCV. It can detect and annotate vehicles in a video feed, such as cars, motorcycles, buses, and trucks, with bounding boxes and confidence scores displayed for each detection.

## Features

- Detects specific types of vehicles: cars, motorcycles, buses, and trucks.
- Draws bounding boxes and confidence scores around detected vehicles.
- Runs in real-time using YOLOv8 and OpenCV.
- Configurable to work on CPU or GPU for faster processing.

## Prerequisites

Make sure you have the following dependencies installed:

- Python 3.10
- OpenCV
- Ultralytics YOLOv8
- Torch
- Supervision (for frame handling and annotation)
- Numpy
- Matplotlib

You should create a virtual environment for running python
```bash
python3.10 -m venv venv
```
Then, access to created virtual environment
```bash
source venv/bin/activate
```
To install the required packages, you can use:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository and navigate to the project directory.
2. Download the YOLOv8 model weights file (`yolov8s.pt`) and place it in the same directory.
3. Place your video file in the directory and update the `SOURCE_VIDEO_PATH` in the script.

## Usage

Run the script with the following command:

```bash
python main.py
```

Press `q` to exit the real-time detection window.

### Demo
I am using CPU so it's quite slow. To speed up the inference, your device should be equipped GPU
![Demo GIF](demo/demo.gif)


## Code Overview

- **Model and Device Setup**: Loads the YOLOv8 model and configures it to use GPU if available, falling back to CPU otherwise.
- **Selected Classes**: Specifies which vehicle classes to detect (car, motorcycle, bus, truck).
- **Real-time Detection**: Reads video frames in real-time, performs detections, and annotates frames with bounding boxes and confidence scores.
- **Display**: Shows each annotated frame in a window until `q` is pressed.

## Example

After running the code, the application will detect vehicles in the specified video and display bounding boxes with confidence scores in real-time.

## License

This project is licensed under the MIT License.
