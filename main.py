import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import supervision as sv
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOURCE_VIDEO_PATH = "videos/video-vehicle1.mp4"
MODEL = "yolov8s.pt"
model = YOLO(MODEL).to(device)
model.fuse()
     
# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names

# class_ids of interest - car, motorcycle, bus and truck
selected_classes = [2, 3, 5, 7]

def main():
    generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
    while True:
        iterator = iter(generator)
        frame = next(iterator)
        results = model(frame, verbose=False, imgsz=640)[0]
        drawing_frame = frame.copy()
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.confidence > 0.7]
        detections = detections[np.isin(detections.class_id, selected_classes)]
        labels = [
            f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for confidence, class_id in zip(detections.confidence, detections.class_id)
        ]
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        annotated_frame = box_annotator.annotate(scene=drawing_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imshow("frame", annotated_frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()