
# Vehicle Tracking Application

This application uses the YOLOv8 model for real-time vehicle detection and tracking in video footage. It counts unique cars crossing a specified line in the video and displays various statistics.

## Features

- Real-time vehicle detection using YOLOv8.
- Unique vehicle tracking and counting.
- Display of FPS (Frames Per Second) and the total count of detected vehicles.
- User interaction via mouse movement to get pixel coordinates.

## Requirements

Before running the application, make sure you have the following installed:

- Python 3.x
- OpenCV
- Pandas
- Ultralytics YOLOv8
- A suitable tracker module (implement or provide details)

### Installation

You can install the required packages using pip:

```bash
pip install opencv-python pandas ultralytics
```

## Files Needed

- `yolov8s.pt`: The YOLOv8 model weights. You can download it from the [Ultralytics YOLO repository](https://github.com/ultralytics/yolov8).
- `coco.txt`: A text file containing the COCO class names used by YOLO.
- `veh2.mp4`: A sample video file for vehicle tracking.
- `tracker.py`: A Python module containing the implementation of the Tracker class (ensure this is included in your project).

## How to Run

1. Clone or download the repository.
2. Ensure all required files are in the same directory as your main script.
3. Run the script:

```bash
python main.py
```

4. The application will open a window showing the video with detected vehicles. Press 'q' to exit.

## Usage

- Move the mouse over the video window to print the BGR color values of the pixels (optional).
- The green line indicates where the vehicles are counted.
- The statistics for total cars and cars that crossed the counting line are displayed in real-time.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/yolov8) for providing the YOLOv8 model.
- [OpenCV](https://opencv.org/) for image and video processing.
- [Pandas](https://pandas.pydata.org/) for data manipulation.

```

### Notes:
- Modify any section as necessary to better fit your project.
- Ensure that all external files and dependencies mentioned are available to the user.
- If there are specific instructions or configurations for the `tracker.py` file, consider adding a section for that as well.