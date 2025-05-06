import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
from math import dist
from tracker import*
import os
import csv
from datetime import datetime
from collections import deque

# get the mouse coordinates 
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

# Function to initialize CSV file with headers if it doesn't exist
def initialize_csv(csv_path):
    headers = [
        'timestamp', 
        'total_vehicles', 
        'cars', 
        'buses', 
        'trucks', 
        'bikes',
        'vehicles_per_time_period',
        'avg_speed',
        'speed_violations',
        'avg_accuracy_car',    # Added accuracy metrics
        'avg_accuracy_bus',
        'avg_accuracy_truck',
        'avg_accuracy_bike',
        'overall_accuracy',
        'vehicles_passing_in_period',  # New column
        'avg_vehicles_per_period'      # New column
    ]
    
    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Create file with headers if it doesn't exist
    if not file_exists:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
    
    return file_exists

# Function to log data to CSV
def log_to_csv(csv_path, data):
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

cv2.namedWindow('TMS')
cv2.setMouseCallback('TMS', RGB)

# load pretrained model
model = YOLO('yolov8s.pt') 

input_video_path = 'test-videos/v6.mp4'
# read Video
cap = cv2.VideoCapture(input_video_path)

# load class
my_file = open("coco.txt", "r") 
data = my_file.read()
class_list = data.split("\n")

# Setup output paths
output_directory = 'output'
os.makedirs(output_directory, exist_ok=True)
output_video_path = os.path.join(output_directory, 'output_video.mp4')
csv_path = os.path.join(output_directory, 'traffic_data.csv')

# Initialize CSV file
initialize_csv(csv_path)

# original video size and fps
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (1020, 500))  # Adjusted to match resized frame

count = 0

# Co-ordinates of the desired region (Region of Interest or ROI)
area = [(170, 219), (1018, 219), (1003, 286), (110, 286)]
area2 = [(85, 316), (1014, 316), (978, 392), (1, 392)]

area_c = set()  # Initialize empty Set

tracker = Tracker()  # Initialize the Tracker object

speed_limit = 120

vehicles_entering = {}  # Initialize empty dictionary
vehicles_elapsed_time = {}  # Initialize empty dictionary
vehicles_entering_backward = {}  # Initialize empty dictionary
vehicles_speeds = []  # To track all vehicle speeds
speed_violations = 0  # Counter for speed violations

p_time = time.time()
time_period_to_count = 5  # seconds
number_of_vehicle_detected_in_last_t_sec = 0

set_of_vehicle_detected = set()

# New variables for the vehicle passing feature
period_duration = 10  # Time period in seconds (configurable)
period_start_time = time.time()
vehicles_in_current_period = set()
vehicles_passed_in_periods = deque(maxlen=10)  # Store last 10 periods
total_periods = 0

car_count = 0
bike_count = 0
truck_count = 0
bus_count = 0

# Accuracy tracking dictionaries
car_confidences = []
bus_confidences = []
truck_confidences = []
bike_confidences = []
overall_confidences = []

# Performance metrics
start_processing_time = time.time()
total_frames_processed = 0
csv_logging_interval = 30  # Log data every 30 seconds

# Set to track vehicles that have passed in each period
vehicles_passed_set = set()

while True:
    curr_time = time.time()
    
    # Reset vehicle count every time_period_to_count seconds
    if curr_time - p_time >= time_period_to_count:
        print(f"Vehicles counted in {time_period_to_count} seconds: {len(set_of_vehicle_detected)}")
        number_of_vehicle_detected_in_last_t_sec = len(set_of_vehicle_detected)
        # clear the set
        set_of_vehicle_detected.clear()
        p_time = curr_time
    
    # Check if the period has elapsed for vehicle passing count
    if curr_time - period_start_time >= period_duration:
        # Increment total periods
        total_periods += 1
        
        # Get number of vehicles that passed in this period
        vehicles_passed_count = len(vehicles_in_current_period)
        
        # Add to the deque
        vehicles_passed_in_periods.append(vehicles_passed_count)
        
        # Log the data
        print(f"Vehicles passed in last {period_duration}s: {vehicles_passed_count}")
        
        # Reset for next period
        vehicles_in_current_period.clear()
        period_start_time = curr_time
    
    # Log data to CSV every csv_logging_interval seconds
    elapsed_since_start = curr_time - start_processing_time
    if elapsed_since_start > 0 and elapsed_since_start % csv_logging_interval < 0.1:  # Check if we're close to the interval
        # Calculate average speed
        avg_speed = 0
        if vehicles_speeds:
            avg_speed = sum(vehicles_speeds) / len(vehicles_speeds)
        
        # Calculate average accuracies
        avg_accuracy_car = sum(car_confidences) / len(car_confidences) * 100 if car_confidences else 0
        avg_accuracy_bus = sum(bus_confidences) / len(bus_confidences) * 100 if bus_confidences else 0
        avg_accuracy_truck = sum(truck_confidences) / len(truck_confidences) * 100 if truck_confidences else 0
        avg_accuracy_bike = sum(bike_confidences) / len(bike_confidences) * 100 if bike_confidences else 0
        overall_accuracy = sum(overall_confidences) / len(overall_confidences) * 100 if overall_confidences else 0
        
        # Calculate average vehicles per period
        avg_vehicles_per_period = sum(vehicles_passed_in_periods) / len(vehicles_passed_in_periods) if vehicles_passed_in_periods else 0
        
        # Get the most recent count of vehicles passed in the last completed period
        recent_vehicles_passing = vehicles_passed_in_periods[-1] if vehicles_passed_in_periods else 0
        
        # Prepare data for CSV
        total_vehicles = car_count + bus_count + truck_count + bike_count
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        csv_data = [
            timestamp,
            total_vehicles,
            car_count,
            bus_count,
            truck_count,
            bike_count,
            number_of_vehicle_detected_in_last_t_sec,
            round(avg_speed, 2) if avg_speed else 0,
            speed_violations,
            round(avg_accuracy_car, 2),
            round(avg_accuracy_bus, 2),
            round(avg_accuracy_truck, 2),
            round(avg_accuracy_bike, 2),
            round(overall_accuracy, 2),
            recent_vehicles_passing,
            round(avg_vehicles_per_period, 2)
        ]
        
        # Log to CSV
        log_to_csv(csv_path, csv_data)
        print(f"Data logged to CSV at {timestamp}")
        
        # Reset certain metrics after logging
        vehicles_speeds = []
        speed_violations = 0
        # Keep the confidence scores for cumulative accuracy tracking

    try:
        ret, frame = cap.read()
        
        if not ret:
            break
            
        count += 1
        total_frames_processed += 1
        
        frame = cv2.resize(frame, (1020, 500))

        # Detect objects using YOLO
        results = model.predict(frame)
        
        a = results[0].boxes
        detected_vehicle_data_in_a_frame = results[0].boxes.data
        
        # show predicted result
        px = pd.DataFrame(detected_vehicle_data_in_a_frame).astype("float")

        list = []  # Initialize empty List

        # Reset vehicle counts for each frame
        car_count = 0
        bike_count = 0
        truck_count = 0
        bus_count = 0

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            confidence = float(row[4])  # Extract confidence score
            d = int(row[5])
            c = class_list[d]
            
            # Add confidence to overall metrics
            overall_confidences.append(confidence)
            
            if 'car' in c:
                list.append([x1, y1, x2, y2])
                car_count += 1
                car_confidences.append(confidence)
            if 'motorcycle' in c:
                list.append([x1, y1, x2, y2])
                bike_count += 1
                bike_confidences.append(confidence)
            elif 'truck' in c:
                list.append([x1, y1, x2, y2])
                truck_count += 1
                truck_confidences.append(confidence)
            elif 'bus' in c:
                list.append([x1, y1, x2, y2])
                bus_count += 1
                bus_confidences.append(confidence)

        # returns a list of bounding boxes with the IDs
        bbox_id = tracker.update(list)

        # each box
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            set_of_vehicle_detected.add(id)
            cx = int(x3+x4)//2
            cy = int(y3+y4)//2

            results = cv2.pointPolygonTest(
                np.array(area, np.int32), ((cx, cy)), False)   

            results2 = cv2.pointPolygonTest(
                np.array(area2, np.int32), ((cx, cy)), False) 
            
            # Track vehicles passing through either area for the period counting
            if results >= 0 or results2 >= 0:
                vehicles_in_current_period.add(id)
            
            # Area-1 (forward moving vehicles enter here first)
            if results >= 0:
                # forward vehicles
                if id not in vehicles_entering_backward:
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                    # start timer
                    Init_time = time.time()
                    if id not in vehicles_entering:
                        vehicles_entering[id] = Init_time
                    else:
                        Init_time = vehicles_entering[id]
                # backward vehicles
                else:
                    try:
                        elapsed_time = time.time() - vehicles_entering_backward[id]
                    except KeyError:
                        pass
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX,
                                    0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

                    if id not in vehicles_entering_backward:
                        vehicles_entering_backward[id] = elapsed_time
                    else:
                        try:
                            # Speed -> distance/elapsed time
                            elapsed_time = vehicles_entering_backward[id]
                            dist = 12  # Distance between two region
                            speed_KH = (dist/elapsed_time)*3.6

                            # Store speed for analytics
                            vehicles_speeds.append(speed_KH)

                            cv2.putText(frame, str(int(speed_KH))+'Km/h', (x4, y4),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 139), 2, cv2.LINE_AA)

                        except ZeroDivisionError:
                            pass

                        if speed_KH >= speed_limit:
                            # Increment speed violation counter
                            speed_violations += 1
                            
                            # Display a warning message
                            cv2.waitKey(500)
                            cv2.putText(frame, f"Speed limit violated!", (440, 112),
                                        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 139), 2, cv2.LINE_AA)
                            cv2.putText(frame, f"Vehicle ID = {id}", (580, 50),
                                cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 139), 2, cv2.LINE_AA)
                            cv2.putText(frame, 'Detected', (cx, cy),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                            cv2.waitKey(500)

            # Area-2 | Main Area (backward moving vehicles enter here first)
            if results2 >= 0:
                # backward vehicles
                if id not in vehicles_entering:
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    area_c.add(id)  # Vehicle-counter
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                    Init_time = time.time()

                    if id not in vehicles_entering_backward:
                        vehicles_entering_backward[id] = Init_time
                    else:
                        Init_time = vehicles_entering_backward[id]

                # forward vehicles
                else:
                    try:
                        elapsed_time = time.time() - vehicles_entering[id]
                    except KeyError:
                        pass

                    if id not in vehicles_elapsed_time:
                        vehicles_elapsed_time[id] = elapsed_time
                    else:
                        try:
                            elapsed_time = vehicles_elapsed_time[id]
                            dist = 12  # Distance between two region
                            speed_KH = (dist/elapsed_time)*3.6
                            
                            # Store speed for analytics
                            vehicles_speeds.append(speed_KH)

                            cv2.putText(frame, str(int(speed_KH))+'Km/h', (x4, y4),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 139), 2, cv2.LINE_AA)

                        except ZeroDivisionError:
                            pass

                        if speed_KH >= speed_limit:
                            # Increment speed violation counter
                            speed_violations += 1
                            
                            # Display a warning message
                            cv2.waitKey(500)
                            cv2.putText(frame, f"Speed limit violated!", (580, 106),
                                        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 139), 2, cv2.LINE_AA)
                            cv2.putText(frame, f"Vehicle ID = {id}", (580, 50),
                                        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 139), 2, cv2.LINE_AA)
                            
                            cv2.putText(frame, 'Detected', (cx, cy),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                            cv2.waitKey(500)

        # Draw ROI areas
        cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)  # Area-1
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 2)  # Area-2 | Main Area

        cnt = len(area_c)
        
        # Calculate current average speed
        current_avg_speed = 0
        if vehicles_speeds:
            current_avg_speed = sum(vehicles_speeds) / len(vehicles_speeds)
        
        # Calculate current accuracy metrics for display
        current_accuracy_car = sum(car_confidences[-100:]) / len(car_confidences[-100:]) * 100 if car_confidences else 0
        current_accuracy_overall = sum(overall_confidences[-100:]) / len(overall_confidences[-100:]) * 100 if overall_confidences else 0
        
        # Calculate average vehicles per period for display
        avg_vehicles_per_period = sum(vehicles_passed_in_periods) / len(vehicles_passed_in_periods) if vehicles_passed_in_periods else 0
            
        # Display information on the frame
        display_texts = [
            f"Vehicle Count: {car_count + bus_count + truck_count + bike_count}",
            f"Cars: {car_count}",
            f"Buses: {bus_count}",
            f"Trucks: {truck_count}",
            f"Bikes: {bike_count}",
            f"Vehicles ({int(curr_time - p_time)}s): {number_of_vehicle_detected_in_last_t_sec}",
            f"Avg Speed: {int(current_avg_speed) if current_avg_speed else 0} Km/h",
            f"Speed Violations: {speed_violations}",
            f"Car Accuracy: {int(current_accuracy_car)}%",
            f"Overall Accuracy: {int(current_accuracy_overall)}%",
            f"Vehicles in current {period_duration}s: {len(vehicles_in_current_period)}",
            f"Avg vehicles per {period_duration}s: {int(avg_vehicles_per_period)}"
        ]
        
        # Starting position for the text
        x, y = 50, 50  # (x, y) coordinates for the first label
        line_spacing = 40  # Spacing between each line
        
        # Overlay the text on the frame
        for index, text in enumerate(display_texts):
            cv2.putText(
                frame,
                text,
                (x, y + index * line_spacing),  # Adjust vertical position for each label
                cv2.FONT_HERSHEY_TRIPLEX,
                1,  # Font scale
                (0, 0, 0),  # Text color
                2,  # Thickness
                cv2.LINE_AA  # Anti-aliased line
            )

        # Show and write the frame
        cv2.imshow("TMS", frame)
        
        # Write frame to output video
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    except Exception as e:
        print(f"Error in frame processing: {e}")
        continue

# Final log entry before closing
try:
    # Calculate average speed
    avg_speed = 0
    if vehicles_speeds:
        avg_speed = sum(vehicles_speeds) / len(vehicles_speeds)
    
    # Calculate final accuracy metrics
    final_accuracy_car = sum(car_confidences) / len(car_confidences) * 100 if car_confidences else 0
    final_accuracy_bus = sum(bus_confidences) / len(bus_confidences) * 100 if bus_confidences else 0
    final_accuracy_truck = sum(truck_confidences) / len(truck_confidences) * 100 if truck_confidences else 0
    final_accuracy_bike = sum(bike_confidences) / len(bike_confidences) * 100 if bike_confidences else 0
    final_accuracy_overall = sum(overall_confidences) / len(overall_confidences) * 100 if overall_confidences else 0
    
    # Calculate average vehicles per period for the final report
    avg_vehicles_per_period = sum(vehicles_passed_in_periods) / len(vehicles_passed_in_periods) if vehicles_passed_in_periods else 0
    recent_vehicles_passing = vehicles_passed_in_periods[-1] if vehicles_passed_in_periods else 0
    
    # Prepare final data for CSV
    total_vehicles = car_count + bus_count + truck_count + bike_count
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    csv_data = [
        timestamp + " (Final)",
        total_vehicles,
        car_count,
        bus_count,
        truck_count,
        bike_count,
        number_of_vehicle_detected_in_last_t_sec,
        round(avg_speed, 2) if avg_speed else 0,
        speed_violations,
        round(final_accuracy_car, 2),
        round(final_accuracy_bus, 2),
        round(final_accuracy_truck, 2),
        round(final_accuracy_bike, 2),
        round(final_accuracy_overall, 2),
        recent_vehicles_passing,
        round(avg_vehicles_per_period, 2)
    ]
    
    # Log final data to CSV
    log_to_csv(csv_path, csv_data)
    
    # Also log performance metrics
    total_processing_time = time.time() - start_processing_time
    fps_processing = total_frames_processed / total_processing_time if total_processing_time > 0 else 0
    
    with open(os.path.join(output_directory, 'performance_metrics.csv'), 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(['total_frames_processed', 'total_processing_time_seconds', 'average_fps', 'total_vehicles_detected', 'total_speed_violations', 'avg_accuracy_car', 'avg_accuracy_bus', 'avg_accuracy_truck', 'avg_accuracy_bike', 'overall_accuracy', 'vid', 'vehicles_passing_in_time_period', 'period', 'avg_vehicles_per_period'])
        writer.writerow([
            total_frames_processed,  
            round(total_processing_time, 2), 
            round(fps_processing, 2),  
            len(area_c), 
            speed_violations,
            round(final_accuracy_car, 2),
            round(final_accuracy_bus, 2),
            round(final_accuracy_truck, 2),
            round(final_accuracy_bike, 2),
            round(final_accuracy_overall, 2),
            input_video_path,
            recent_vehicles_passing,
            period_duration,
            round(avg_vehicles_per_period, 2)
        ])
except Exception as e:
    print(f"Error in final logging: {e}")

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Results saved to {csv_path}")
print(f"Performance metrics saved to {os.path.join(output_directory, 'performance_metrics.csv')}")
print(f"Output video saved to {output_video_path}")