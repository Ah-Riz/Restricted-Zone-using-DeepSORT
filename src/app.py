from pytubefix import YouTube
from pytubefix.cli import on_progress
import cv2
import torch
import os
import time
import logging
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort

logging.basicConfig(
    filename="pytubefix.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

TEMP_VIDEO_PATH = "temp_video.mp4"
OUTPUT_VIDEO_PATH = "output_video.mp4"
RESTRICTED_ZONES = [(200, 100), (600, 400)]
ALERT_THRESHOLD = 3
FRAME_SKIP = 2
# DETECTION_CONFIDENCE = 0.5
RESIZE_RATIO = 0.5

def main(yt_url = "https://www.youtube.com/watch?v=GJNjaRJWVP8"):
    video_path = video_downloader(yt_url)
    if video_path is None:
        return
    
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    # model.conf = DETECTION_CONFIDENCE
    model.classes = [0]  # Only detect persons
    model.eval()
    
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    new_width = int(width * RESIZE_RATIO)
    new_height = int(height * RESIZE_RATIO)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    intrusion_times = {}
    frame_count = 0
    processed_frames = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # frame_count += 1
        # if frame_count % (FRAME_SKIP + 1) != 0:
        #     out.write(frame)
        #     continue

        processed_frames += 1

        small_frame = cv2.resize(frame, (new_width, new_height))

        with torch.no_grad():
            results = model(small_frame)
        
        detections = []

        for *xyxy, conf, cls in results.xyxy[0]:
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, [coord / RESIZE_RATIO for coord in xyxy])
                bbox = (x1, y1, x2 - x1, y2 - y1)
                detections.append((bbox, conf.item(), 'person'))
        
        tracks = tracker.update_tracks(detections, frame=frame)

        cv2.rectangle(frame, RESTRICTED_ZONES[0], RESTRICTED_ZONES[1], (0, 0, 255), 2)
        cv2.putText(frame, "Restricted Zone", (RESTRICTED_ZONES[0][0], RESTRICTED_ZONES[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            track_id = track.track_id
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            in_zone = (RESTRICTED_ZONES[0][0] <= center_x <= RESTRICTED_ZONES[1][0] and
                       RESTRICTED_ZONES[0][1] <= center_y <= RESTRICTED_ZONES[1][1])
            
            if in_zone:
                if track_id not in intrusion_times:
                    intrusion_times[track_id] = time.time()
                else:
                    duration = time.time() - intrusion_times[track_id]
                    if duration > ALERT_THRESHOLD:
                        logging.warning(f"Intrusion detected by track ID {track_id} at {datetime.now()}")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Alert: Track ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                if track_id in intrusion_times:
                    del intrusion_times[track_id]
        
        out.write(frame)
    
    cap.release()
    out.release()

    processing_time = time.time() - start_time
    total_frames = frame_count
    logging.info(f"Processing complete")
    logging.info(f"Total frames: {total_frames}")
    logging.info(f"Processed frames: {processed_frames}")
    logging.info(f"Processing time: {processing_time:.2f} seconds")
    logging.info(f"Average FPS: {total_frames / processing_time:.2f}")

    if os.path.exists(TEMP_VIDEO_PATH):
        os.remove(TEMP_VIDEO_PATH)

def video_downloader(yt_url):
    try:
        yt = YouTube(yt_url, on_progress_callback = on_progress)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
        if not stream:
            logging.error("No suitable stream found for the video.")
            return None
        stream.download(filename=TEMP_VIDEO_PATH)
        logging.info(f"Video downloaded successfully: {TEMP_VIDEO_PATH}")
        return TEMP_VIDEO_PATH
    except Exception as e:
        logging.error(f"Error downloading video: {e}")
        return None

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=GJNjaRJWVP8"
    main(url)