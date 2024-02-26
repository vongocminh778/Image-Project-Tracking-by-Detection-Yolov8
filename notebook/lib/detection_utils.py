import cv2
import numpy as np

def draw_detection(img, bboxes, scores, class_ids, ids, classes=['objects'], mask_alpha=0.3):
        height, width = img.shape[:2]
        np.random.seed(0)
        rng = np.random.default_rng(3)
        colors = rng.uniform(0, 255, size=(len(classes), 3))

        mask_img = img.copy()
        det_img = img.copy()

        size = min([height, width]) * 0.0006
        text_thickness = int(min([height, width]) * 0.001)

        for bbox, score, class_id, id_ in zip(bboxes, scores, class_ids, ids):
            color = colors[class_id]

            x1, y1, x2, y2 = bbox.astype(int)

            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

            label = classes[class_id]
            caption = f'{label} {int(score * 100)}% ID: {id_}'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(det_img, (x1, y1),
                        (x1 + tw, y1 - th), color, -1)
            cv2.rectangle(mask_img, (x1, y1),
                        (x1 + tw, y1 - th), color, -1)
            cv2.putText(det_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

            cv2.putText(mask_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)
