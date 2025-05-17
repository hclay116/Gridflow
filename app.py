# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import time
import os
import torch
from detect_car_peds import extract_image_features 

app = Flask(__name__)
CORS(app)

# --- Global YOLO Model Setup ---
MODEL_VARIANT = "yolov5x"
CONF_THRESHOLD = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading YOLOv5 model ({MODEL_VARIANT}) on {device}...")
model = torch.hub.load("ultralytics/yolov5", MODEL_VARIANT, pretrained=True)
model.conf = CONF_THRESHOLD
model.to(device)
TARGET_LABELS = {"person", "car"}
COLORS = {"person": (0, 255, 0), "car": (0, 0, 255)}

class TrafficQLearning:
    def __init__(self):
        """
        Simple safe state:
          - Only two safe states: NS-green or EW-green
          - No all-green to avoid collisions
          - Wait times are tracked for each approach.
        """
        self.vehicle_presence = {
            'N': {'present': False, 'wait_time': 0},
            'S': {'present': False, 'wait_time': 0},
            'E': {'present': False, 'wait_time': 0},
            'W': {'present': False, 'wait_time': 0}
        }
        self.min_phase_time = 5
        self.max_phase_time = 30
        self.last_switch_time = time.time()
        self.current_safe = "NS"
        self.current_state = {'N': 'green', 'S': 'green', 'E': 'red', 'W': 'red'}
        self.last_features = {}
        self.image_counter = 0

    def detect_vehicles(self, image_array):
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        zones = {
            'N': gray[h//2 - 40 : h//2 - 5,  w//2 - 20 : w//2 + 20],
            'S': gray[h//2 + 5  : h//2 + 40, w//2 - 20 : w//2 + 20],
            'E': gray[h//2 - 20 : h//2 + 20, w//2 + 5  : w//2 + 40],
            'W': gray[h//2 - 20 : h//2 + 20, w//2 - 40 : w//2 - 5]
        }
        approaching = set()
        for direction, zone in zones.items():
            is_present = np.mean(zone) > 40
            self.vehicle_presence[direction]['present'] = is_present
            if is_present:
                approaching.add(direction)
                self.vehicle_presence[direction]['wait_time'] += 1
            else:
                self.vehicle_presence[direction]['wait_time'] = 0

        print(f"[Detect] Approaching: {approaching}, Wait times: {self.vehicle_presence}")
        return approaching

    def resolve_conflicts(self, approaching):
        now = time.time()
        elapsed = now - self.last_switch_time
        ns_wait = self.vehicle_presence['N']['wait_time'] + self.vehicle_presence['S']['wait_time']
        ew_wait = self.vehicle_presence['E']['wait_time'] + self.vehicle_presence['W']['wait_time']
        print(f"[Resolve] Elapsed: {elapsed:.2f} sec, NS_wait: {ns_wait}, EW_wait: {ew_wait}")

        if elapsed < self.min_phase_time:
            print("[Resolve] Holding state due to min_phase_time.")
            return self.current_state

        if elapsed >= self.max_phase_time:
            if self.current_safe == "NS":
                self.set_EW_green()
            else:
                self.set_NS_green()
            self.last_switch_time = now
            print("[Resolve] Forced toggle due to max_phase_time.")
            return self.current_state

        if ns_wait > ew_wait:
            if self.current_safe != "NS":
                self.set_NS_green()
                self.last_switch_time = now
                print("[Resolve] Switching to NS-green due to higher wait.")
        elif ew_wait > ns_wait:
            if self.current_safe != "EW":
                self.set_EW_green()
                self.last_switch_time = now
                print("[Resolve] Switching to EW-green due to higher wait.")
        else:
            print("[Resolve] Tie in wait times; holding state.")

        return self.current_state

    def set_NS_green(self):
        self.current_safe = "NS"
        self.current_state = {'N': 'green', 'S': 'green', 'E': 'red', 'W': 'red'}

    def set_EW_green(self):
        self.current_safe = "EW"
        self.current_state = {'N': 'red', 'S': 'red', 'E': 'green', 'W': 'green'}

    def save_image(self, image_array):
        self.image_counter += 1
        filename = f"frame_{self.image_counter}.jpg"
        cv2.imwrite(filename, image_array)
        print(f"[Save] Image saved as {filename}")

controller = TrafficQLearning()

@app.route('/detect/bbox', methods=['POST'])
def detect_bbox():
    data = request.json
    image_b64 = data['image']  # base64 portion
    # decode the base64 to an image array
    image_bytes = base64.b64decode(image_b64)
    image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # run YOLO detection
    results = model(image_array)  # depends on your YOLO loading code
    detections = []
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det
        label = model.names[int(cls)]
        detections.append({
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2),
            "label": label,
            "confidence": float(conf)
        })
    return jsonify({"bboxes": detections})


@app.route('/policy/rl', methods=['POST'])
def get_traffic_action():
    try:
        image_data = base64.b64decode(request.json['image'])
        image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        if image_array is None:
            print("[Error] Decoded image is None.")
            return jsonify({"error": "Decoded image is None"}), 500

        print(f"[Debug] Image shape: {image_array.shape}")

        # Annotate image with YOLOv5 detections.
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        results = model(rgb_image)
        detections = results.xyxy[0].cpu().numpy()
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            label = model.names[int(cls)]
            if label not in TARGET_LABELS:
                continue
            color = COLORS.get(label, (255, 255, 255))
            cv2.rectangle(image_array, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(image_array, text, (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Extract features.
        features = extract_image_features(image_array, mode='simulation')
        controller.last_features = features
        print(f"[Features] {features}")

        # Detect vehicles.
        approaching = controller.detect_vehicles(image_array)
        # Resolve safe state.
        new_state = controller.resolve_conflicts(approaching)
        controller.current_state = new_state
        response_state = "NS-green" if new_state['N'] == 'green' else "EW-green"
        print(f"[Response] {response_state}")
        return jsonify({"state": response_state})
    except Exception as e:
        print(f"[Error] {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs("frames", exist_ok=True)
    # Optionally change working directory:
    # os.chdir("frames")
    print("Traffic controller running on port 5001...")
    app.run(host='0.0.0.0', port=5001)
