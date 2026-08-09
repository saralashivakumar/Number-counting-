from flask import Flask, render_template, Response, jsonify, request
import cv2
import json
import imutils
import numpy as np
import dlib
import datetime
import threading
import time
from imutils.video import VideoStream, FPS
from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from utils.mailer import Mailer
from utils import thread

app = Flask(__name__)

# Global state for stats and video stream generator
stats = {
    "total_enter": 0,
    "total_exit": 0,
    "inside": 0,
    "status": "Stopped",
    "fps": 0,
    "is_running": False
}

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

PROTOTXT = "detector/MobileNetSSD_deploy.prototxt"
MODEL = "detector/MobileNetSSD_deploy.caffemodel"
CONFIDENCE = 0.3
SKIP_FRAMES = 5

def generate_frames():
    global stats
    with open("utils/config.json", "r") as file:
        config = json.load(file)

    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

    if config["Thread"]:
        vs = thread.ThreadingClass(config["url"])
    else:
        vs = VideoStream(config["url"]).start()
        time.sleep(2.0)

    ct = CentroidTracker(maxDisappeared=40, maxDistance=90)
    trackers = []
    trackableObjects = {}

    totalFrames = 0
    totalDown = 0  # Enter (UP in modified logic)
    totalUp = 0    # Exit (DOWN in modified logic)
    move_in = []
    move_out = []
    
    fps = FPS().start()
    stats["is_running"] = True

    while stats["is_running"]:
        if config["Thread"]:
            frame = vs.read()
        else:
            frame = vs.read()

        if frame is None:
            time.sleep(0.01)
            continue

        frame = imutils.resize(frame, width=640)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        (H, W) = frame.shape[:2]

        rects = []
        status = "Waiting"

        if totalFrames % SKIP_FRAMES == 0:
            status = "Detecting"
            trackers = []
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            boxes = []
            confidences = []

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > CONFIDENCE:
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")
                    boxes.append([startX, startY, endX - startX, endY - startY])
                    confidences.append(float(confidence))

            # Apply Non-Maximum Suppression (NMS) to separate overlapping people in groups
            indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, 0.3)
            if len(indices) > 0:
                for i in indices.flatten():
                    (x, y, w, h) = boxes[i]
                    startX, startY, endX, endY = x, y, x + w, y + h
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)
        else:
            for tracker in trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                rects.append((startX, startY, endX, endY))

        # Draw counting boundary line
        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 230, 255), 2)
        cv2.putText(frame, "DETECTION BOUNDARY", (10, H // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1)

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                # Calculate direction relative to movement history
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                
                # Check if centroid has crossed the center line boundary
                # Or if the centroid trajectory spanned across H // 2
                min_y = min(y)
                max_y = max(y)
                to.centroids.append(centroid)

                if not to.counted:
                    prev_y = to.centroids[-2][1] if len(to.centroids) > 1 else centroid[1]
                    # UP direction -> Enter: started at/below line (prev_y >= H // 2) and crossed UP to above line (centroid[1] < H // 2)
                    if direction < 0 and prev_y >= (H // 2) and centroid[1] < (H // 2):
                        totalDown += 1
                        move_in.append(totalDown)
                        to.counted = True
                    # DOWN direction -> Exit: started at/above line (prev_y <= H // 2) and crossed DOWN to below line (centroid[1] > H // 2)
                    elif direction > 0 and prev_y <= (H // 2) and centroid[1] > (H // 2):
                        totalUp += 1
                        move_out.append(totalUp)
                        to.counted = True

            trackableObjects[objectID] = to
            cv2.putText(frame, f"ID {objectID}", (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 120), 1)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 120), -1)

        total_inside = max(0, len(move_in) - len(move_out))

        stats["total_enter"] = len(move_in)
        stats["total_exit"] = len(move_out)
        stats["inside"] = total_inside
        stats["status"] = status
        
        totalFrames += 1
        fps.update()
        if totalFrames % 10 == 0:
            fps.stop()
            stats["fps"] = round(fps.fps(), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    if config["Thread"]:
        vs.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    return jsonify(stats)

@app.route('/api/config', methods=['GET', 'POST'])
def manage_config():
    if request.method == 'POST':
        new_cfg = request.json
        with open("utils/config.json", "w") as f:
            json.dump(new_cfg, f, indent=4)
        return jsonify({"status": "success"})
    with open("utils/config.json", "r") as f:
        cfg = json.load(f)
    return jsonify(cfg)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
