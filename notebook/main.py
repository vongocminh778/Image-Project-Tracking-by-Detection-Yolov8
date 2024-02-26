import cv2
import numpy as np
from lib.yolov8_interface import YOLOv8
from lib.deepsort_interface import DeepSORT
from lib.detection_utils import draw_detection

def main():
    yolo_model_path = 'yolov8_mot_det.pt'
    video_path = 'CityRoam.mp4'

    detector = YOLOv8(yolo_model_path)
    tracker = DeepSORT()
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    all_tracking_results = []
    tracked_ids = np.array([], dtype=np.int32)
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        detector_results = detector.detect(frame)
        bboxes, scores, class_ids = detector_results

        tracker_pred = tracker.tracking(
            origin_frame=frame,
            bboxes=bboxes,
            scores=scores,
            class_ids=class_ids
        )
        if tracker_pred.size > 0:
            bboxes = tracker_pred[:, :4]

            class_ids = tracker_pred[:, 4].astype(int)
            conf_scores = tracker_pred[:, 5]
            tracking_ids = tracker_pred[:, 6].astype(int)

            new_ids = np.setdiff1d(tracking_ids, tracked_ids)
            tracked_ids = np.concatenate((tracked_ids, new_ids))

            frame = draw_detection(
                img=frame,
                bboxes=bboxes,
                scores=conf_scores,
                class_ids=class_ids,
                ids=tracking_ids
            )

        all_tracking_results.append(tracker_pred)
        
        cv2.imshow('yolov8', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
