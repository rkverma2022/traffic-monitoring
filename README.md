# Traffic Monitoring System

An AI-powered traffic monitoring and analysis system that uses computer vision to detect, track, and analyze vehicles in video footage. The system provides real-time metrics including vehicle counts, speeds, and traffic flow patterns.

## Features

- **Vehicle Detection & Classification**: Identifies and categorizes vehicles as cars, buses, trucks, and motorcycles
- **Speed Monitoring**: Calculates and displays vehicle speeds in km/h
- **Speed Violation Detection**: Identifies vehicles exceeding the configured speed limit
- **Traffic Flow Analysis**: Tracks vehicle counts in customizable time periods
- **Period-based Traffic Metrics**: Monitors vehicles passing within defined time intervals
- **Data Logging**: Records comprehensive traffic data to CSV for further analysis
- **Performance Metrics**: Tracks system performance including processing speed and detection accuracy
- **Video Output**: Generates annotated video output with real-time analytics overlay

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Pandas
- Ultralytics YOLO
- Additional dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/traffic-monitoring-system.git
   cd traffic-monitoring-system
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the YOLO model:
   ```
   # The system uses YOLOv8s by default which will be downloaded automatically on first run
   # Alternatively, you can manually download it from the Ultralytics repository
   ```

4. Ensure you have the required files:
   - `tracker.py` - Contains tracking algorithm implementation
   - `coco.txt` - Contains class names for YOLO model

## Usage

1. Place your traffic video in the `test-videos` directory
2. Update the input video path in the script if necessary:
   ```python
   input_video_path = 'test-videos/your-video.mp4'
   ```

3. Run the script:
   ```
   python traffic_monitor.py
   ```

4. Results will be saved to the `output` directory:
   - `output_video.mp4`: Annotated video with analytics overlay
   - `traffic_data.csv`: Detailed traffic metrics
   - `performance_metrics.csv`: System performance data

## Configuration Options

The system offers several configurable parameters:

### Speed Detection
- `speed_limit`: Maximum allowed speed (default: 120 km/h)
- `dist`: Distance between detection regions (default: 12 units)

### Time Periods
- `time_period_to_count`: General counting interval (default: 5 seconds)
- `period_duration`: Traffic flow analysis period (default: 10 seconds)
- `csv_logging_interval`: Data logging frequency (default: 30 seconds)

### Detection Regions
- `area`: First detection region (can be adjusted for different camera angles)
- `area2`: Second detection region (can be adjusted for different camera angles)

## Output Metrics

### Real-time Display
- Total vehicle count by type (cars, buses, trucks, motorcycles)
- Vehicles detected in the latest time period
- Average speed
- Speed violations
- Detection accuracy
- Vehicles passed in the current period
- Average vehicles per period

### CSV Data
The `traffic_data.csv` file includes:
- Timestamp
- Total vehicles by type
- Vehicles per time period
- Average speed
- Speed violations
- Detection accuracy by vehicle type
- Vehicles passing in configured time periods
- Average vehicles per period

## Performance Considerations

- Processing speed depends on hardware specifications and video resolution
- For optimal performance, consider:
  - Using a GPU-enabled system
  - Adjusting video resolution if necessary
  - Running on hardware with adequate memory for YOLO detection

## Troubleshooting

Common issues:
- **CUDA errors**: Make sure your GPU drivers are up to date
- **Missing dependencies**: Check that all required packages are installed
- **Video format issues**: Ensure the video file is in a compatible format
- **Low accuracy**: Consider adjusting the detection regions for your specific camera angle

## License

[Insert your license information here]

## Acknowledgments

- This system uses Ultralytics YOLOv8 for object detection
- Vehicle tracking algorithm based on [provide reference if applicable]