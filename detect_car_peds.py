import cv2
import torch
import numpy as np
import argparse
import time
import sys
import os
from scipy.spatial import distance
import math
from collections import deque
"""
Traffic Footage Vehicle and Pedestrian Detection with Additional RL Features (No Speed Computation)

This script reads an input video (e.g., traffic footage), uses a YOLOv5 model
to detect persons and cars, and then writes out an annotated video. In addition
to the usual bounding boxes, we compute extra features to help a Reinforcement
Learning (RL) agent make decisions for a "smart traffic light" system.
"""

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Detect cars and pedestrians in traffic footage."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input file (image or video).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Path to save the annotated output (video or image).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the output in a window while processing.",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="yolov5x",
        choices=["yolov5s", "yolov5m", "yolov5l", "yolov5x"],
        help=(
            "YOLOv5 model variant to use. "
            "Choose 'yolov5s' (fastest) to 'yolov5x' (most accurate). Default is 'yolov5x'."
        ),
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5).",
    )
    parser.add_argument(
        "--calibration_factor",
        type=float,
        default=8.8,
        help="Calibration factor, in pixels per meter (default: 8.8 px/m).",
    )
    parser.add_argument(
        "--speed_estimation_frames",
        type=int,
        default=5,
        help="Number of frames to use for speed estimation (default: 5).",
    )
    parser.add_argument(
        "--roi_line",
        type=str,
        default=None,
        help="Region of interest line coordinates 'x1,y1,x2,y2' for speed calculation reference (optional).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="real",
        choices=["real", "simulation"],
        help="Processing mode: 'real' for real-world footage, 'simulation' for simulated environments (default: real).",
    )
    return parser.parse_args()


def load_yolo_model(model_variant: str, conf_threshold: float, device: str):
    """
    Loads the specified YOLOv5 model variant from PyTorch Hub with pretrained weights.
    Sets the confidence threshold and moves the model to the desired device.
    """
    print(f"Loading YOLOv5 model variant '{model_variant}' (this may take a few seconds)...")
    try:
        model = torch.hub.load("ultralytics/yolov5", model_variant, pretrained=True)
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        sys.exit(1)
    model.conf = conf_threshold
    model.to(device)
    print("Model loaded successfully!")
    return model


class VehicleTracker:
    """
    A tracker that maintains vehicle tracks and calculates speeds based on
    positional changes over time.
    """
    def __init__(self, max_disappeared=10, max_distance=80, max_history=30):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.max_history = max_history
        
        self.vehicle_history = {}
        self.vehicle_speeds = {}
        self.vehicle_boxes = {}
        self.vehicle_directions = {}
        self.frame_count = 0
        self.speed_buffer = {}
        self.object_classes = {}
        self.vehicle_features = {}  # Store features for each vehicle

    def register(self, centroid, bbox, timestamp, class_label, features=None):
        """Register a new object with a new ID"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.vehicle_history[self.next_object_id] = deque(maxlen=self.max_history)
        self.vehicle_history[self.next_object_id].append((centroid, timestamp))
        self.vehicle_speeds[self.next_object_id] = 0.0
        self.vehicle_boxes[self.next_object_id] = bbox
        self.vehicle_directions[self.next_object_id] = (0, 0)
        self.speed_buffer[self.next_object_id] = deque(maxlen=10)
        self.object_classes[self.next_object_id] = class_label
        
        # Store vehicle features
        if features is not None:
            self.vehicle_features[self.next_object_id] = [features]  # Start a list of features for this vehicle
        else:
            self.vehicle_features[self.next_object_id] = []
            
        self.next_object_id += 1

    def deregister(self, object_id):
        """Deregister an object that has disappeared for too long"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.vehicle_history[object_id]
        del self.vehicle_speeds[object_id]
        del self.vehicle_boxes[object_id]
        del self.vehicle_directions[object_id]
        del self.speed_buffer[object_id]
        del self.object_classes[object_id]
        
        # Also remove stored features
        if object_id in self.vehicle_features:
            del self.vehicle_features[object_id]

    def update(self, detections, timestamp):
        """
        Update the tracker with new detections.
        detections: List of [bbox, centroid, class_label, features] quadruplets
        timestamp: Current timestamp for this frame
        """
        self.frame_count += 1

        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for det in detections:
                if len(det) == 4:  # With features
                    bbox, centroid, class_label, features = det
                    self.register(centroid, bbox, timestamp, class_label, features)
                else:  # Without features (backward compatibility)
                    bbox, centroid, class_label = det
                    self.register(centroid, bbox, timestamp, class_label)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            distances_matrix = distance.cdist(object_centroids, [d[1] for d in detections])

            rows = distances_matrix.min(axis=1).argsort()
            cols = distances_matrix.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if distances_matrix[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]

                new_centroid = detections[col][1]
                old_centroid = self.objects[object_id]

                dx = new_centroid[0] - old_centroid[0]
                dy = new_centroid[1] - old_centroid[1]

                self.objects[object_id] = new_centroid
                self.vehicle_history[object_id].append((new_centroid, timestamp))
                self.vehicle_boxes[object_id] = detections[col][0]
                self.vehicle_directions[object_id] = (dx, dy)
                self.disappeared[object_id] = 0
                
                # Update features if available
                if len(detections[col]) >= 4 and detections[col][3] is not None:
                    if object_id in self.vehicle_features:
                        self.vehicle_features[object_id].append(detections[col][3])
                        # Keep only the most recent features (maximum 30)
                        if len(self.vehicle_features[object_id]) > 30:
                            self.vehicle_features[object_id] = self.vehicle_features[object_id][-30:]

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(distances_matrix.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            unused_cols = set(range(distances_matrix.shape[1])).difference(used_cols)
            for col in unused_cols:
                det = detections[col]
                if len(det) >= 4:  # With features
                    self.register(det[1], det[0], timestamp, det[2], det[3])
                else:  # Without features
                    self.register(det[1], det[0], timestamp, det[2])

        return self.objects

    def calculate_speed(self, calibration_factor, estimation_frames, frame_rate):
        """
        Calculate the speed for all tracked vehicles using a short history window.
        """
        for object_id, history in self.vehicle_history.items():
            if len(history) < 2:
                continue

            if len(history) >= estimation_frames:
                recent_points = list(history)[-estimation_frames:]
                
                total_pixels = 0.0
                total_time = 0.0

                for i in range(len(recent_points) - 1):
                    (pos1, t1) = recent_points[i]
                    (pos2, t2) = recent_points[i + 1]
                    dist_px = distance.euclidean(pos1, pos2)
                    dt = t2 - t1
                    
                    if dt > 0:
                        total_pixels += dist_px
                        total_time += dt

                if total_time < 0.000001:
                    continue

                distance_meters = total_pixels / calibration_factor
                speed_mps = distance_meters / total_time
                speed_kmh = speed_mps * 3.6

                if total_pixels < 2.0:
                    speed_kmh = 0.0

                self.speed_buffer[object_id].append(speed_kmh)
                median_speed = np.median(self.speed_buffer[object_id])

                if self.vehicle_speeds[object_id] == 0:
                    self.vehicle_speeds[object_id] = median_speed
                else:
                    alpha = 0.5
                    self.vehicle_speeds[object_id] = (
                        alpha * median_speed + (1 - alpha) * self.vehicle_speeds[object_id]
                    )


def extract_image_features(frame, mode='real', roi=None):
    """
    Extract various features from an input image/frame or region of interest.
    
    Parameters:
    frame (numpy.ndarray): Input image in BGR format (OpenCV default)
    mode (str): 'real' for real-world footage, 'simulation' for simulated environments
    roi (tuple): Optional region of interest (x1, y1, x2, y2) to extract features from
    
    Returns:
    dict: Dictionary containing various image features and metrics
    """
    import cv2
    import numpy as np
    
    # If ROI is provided, extract that region from the frame
    if roi is not None:
        x1, y1, x2, y2 = roi
        # Ensure ROI is within frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        # Check if ROI has valid dimensions
        if x2 <= x1 or y2 <= y1 or (x2-x1)*(y2-y1) < 100:  # Minimum size check
            return None
            
        frame = frame[y1:y2, x1:x2]
    
    features = {}
    
    # Check if the frame is valid
    if frame.size == 0:
        return None
        
    height, width, channels = frame.shape
    features['dimensions'] = {
        'height': height,
        'width': width,
        'channels': channels,
        'aspect_ratio': width / height if width > 0 and height > 0 else 0
    }
    
    # Basic features for both real and simulation modes
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        features['brightness'] = {
            'mean': np.mean(gray),
            'std': np.std(gray),
        }
        
        # Calculate edge features if the frame is large enough
        if width >= 10 and height >= 10:
            edges = cv2.Canny(gray, 100, 200)
            edge_pixel_count = np.count_nonzero(edges)
            features['edges'] = {
                'edge_density': edge_pixel_count / (width * height),
                'edge_count': edge_pixel_count
            }
        else:
            features['edges'] = {
                'edge_density': 0,
                'edge_count': 0
            }
            
        # Calculate vehicle-specific features
        if roi is not None:
            # Calculate aspect ratio (width/height) - useful for vehicle type
            features['shape'] = {
                'aspect_ratio': width / height if height > 0 else 0,
                'area': width * height,
                'perimeter': 2 * (width + height)
            }
            
            # Calculate distribution of pixels (histogram) for shape analysis
            hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            features['histogram'] = {
                'distribution': hist.flatten().tolist(),
                'variance': np.var(hist)
            }
        
        # Real-world specific features
        if mode == 'real':
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            features['brightness'].update({
                'min': np.min(gray),
                'max': np.max(gray),
            })
            
            features['color'] = {
                'mean_bgr': np.mean(frame, axis=(0, 1)).tolist(),
                'std_bgr': np.std(frame, axis=(0, 1)).tolist(),
                'mean_hsv': np.mean(hsv, axis=(0, 1)).tolist(),
                'std_hsv': np.std(hsv, axis=(0, 1)).tolist(),
            }
            
            # Color distribution (dominant colors)
            if roi is not None:
                # Calculate dominant color using k-means clustering
                Z = frame.reshape((-1, 3))
                Z = np.float32(Z)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                K = 1  # Number of clusters (dominant colors)
                ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                features['color']['dominant'] = center[0].tolist()
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            features['blur'] = {
                'laplacian_variance': laplacian.var(),
                'is_blurry': laplacian.var() < 100
            }
            
            # Only include time of day for full frame, not ROIs
            if roi is None:
                avg_brightness = features['brightness']['mean']
                avg_color_temp = features['color']['mean_bgr'][2] / max(features['color']['mean_bgr'][0], 1)
                
                if avg_brightness < 50:
                    time_of_day = 'night'
                elif avg_brightness < 100:
                    time_of_day = 'evening/dawn' if avg_color_temp > 1.2 else 'dusk/overcast'
                else:
                    time_of_day = 'day'
                    
                features['time_of_day'] = {
                    'estimated': time_of_day,
                    'confidence': min(100, max(0, (avg_brightness / 255) * 100))
                }
    
    except Exception as e:
        # If any error occurs during feature extraction, return basic features
        print(f"Error extracting features: {e}")
        return {'error': str(e), 'dimensions': features.get('dimensions', {})}
        
    return features


def process_image(model, image_path, output_path, mode='real'):
    """Processes an image and saves the annotated result."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open image '{image_path}'.")
        sys.exit(1)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(rgb_image)
    detections = results.xyxy[0].cpu().numpy()

    target_labels = {"person", "car"}
    COLORS = {"person": (0, 255, 0), "car": (0, 0, 255)}
    
    features = extract_image_features(image, mode)
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = model.names[int(cls)]
        if label not in target_labels:
            continue
        
        color = COLORS.get(label, (255, 255, 255))
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(image, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    height, width = image.shape[:2]
    
    # Add time of day annotation only in real mode
    if mode == 'real' and 'time_of_day' in features:
        cv2.putText(
            image,
            f"Time: {features['time_of_day']['estimated']} ({features['time_of_day']['confidence']:.1f}%)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
    
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved as '{output_path}'.")


def process_video(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_yolo_model(args.model_variant, args.conf_threshold, device)

    target_labels = {"car", "person", "truck", "bus", "motorcycle"}  # Extended vehicle types
    COLORS = {"person": (0, 255, 0), "car": (0, 0, 255), "truck": (255, 0, 0), 
              "bus": (255, 255, 0), "motorcycle": (255, 0, 255)}

    roi_line = None
    if args.roi_line:
        try:
            roi_coords = [int(x) for x in args.roi_line.split(',')]
            if len(roi_coords) == 4:
                roi_line = roi_coords
                print(f"Using ROI line: {roi_line}")
        except ValueError:
            print("Invalid ROI line format. Using full frame instead.")

    cap = cv2.Video


def main():
    args = parse_arguments()
    if os.path.isfile(args.input):
        if args.input.lower().endswith(('.png', '.jpg', '.jpeg')):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = load_yolo_model(args.model_variant, args.conf_threshold, device)
            process_image(model, args.input, args.output)
        else:
            process_video(args)
    else:
        print("Error: Input file not found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
