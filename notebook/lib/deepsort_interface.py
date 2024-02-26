from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet
import numpy as np

class DeepSORT:
    def __init__(
        self,
        model_path='resources/networks/mars-small128.pb',
        max_cosine_distance=0.7,
        nn_budget=None,
        classes=['objects']
    ):
        self.encoder = gdet.create_box_encoder(model_path, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
        self.tracker = Tracker(self.metric)

        key_list = []
        val_list = []
        for ID, class_name in enumerate(classes):
            key_list.append(ID)
            val_list.append(class_name)
        self.key_list = key_list
        self.val_list = val_list

    def tracking(
        self,
        origin_frame,
        bboxes,
        scores,
        class_ids
    ):
        features = self.encoder(origin_frame, bboxes)

        detections = [Detection(bbox, score, class_id, feature)
            for bbox, score, class_id, feature in zip(bboxes, scores, class_ids, features)]

        self.tracker.predict()
        self.tracker.update(detections)

        tracked_bboxes = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue
            bbox = track.to_tlbr()
            class_id = track.get_class()
            conf_score = track.get_conf_score()
            tracking_id = track.track_id
            tracked_bboxes.append(
                bbox.tolist() + [class_id, conf_score, tracking_id]
            )

        tracked_bboxes = np.array(tracked_bboxes)

        return tracked_bboxes
