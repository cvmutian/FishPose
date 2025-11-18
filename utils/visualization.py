import cv2
import numpy as np

KP_COLOR = (0, 255, 0)
PRED_KP_COLOR = (255, 0, 0)
BOX_COLOR = (0, 255, 0)
PRED_BOX_COLOR = (255, 0, 0)

def draw_keypoints_and_boxes(
    image,
    pred_keypoints,
    pred_boxes,
    gt_keypoints,
    gt_boxes,
    keypoint_names=None
):
    vis_image = image.copy()
    
    if pred_keypoints:
        for instance_kps in pred_keypoints:
            for i, (x, y, v) in enumerate(instance_kps):
                if v > 0:
                    cv2.circle(vis_image, (int(x), int(y)), 3, PRED_KP_COLOR, -1)
                    cv2.putText(vis_image, str(i), (int(x) + 2, int(y) - 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRED_KP_COLOR, 1)

    if gt_keypoints is not None:
        for instance_kps in gt_keypoints:
            for i, (x, y, v) in enumerate(instance_kps):
                if v > 0:
                    cv2.circle(vis_image, (int(x), int(y)), 3, KP_COLOR, -1)
                    cv2.putText(vis_image, str(i), (int(x) + 2, int(y) - 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, KP_COLOR, 1)
    
    if gt_boxes is not None:
        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), BOX_COLOR, 2)

    if pred_boxes is not None:
        for box in pred_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), PRED_BOX_COLOR, 2)
            
    legend_text_gt = "Green: Ground Truth"
    legend_text_pred = "Blue: Prediction"
    cv2.putText(vis_image, legend_text_gt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, KP_COLOR, 2)
    cv2.putText(vis_image, legend_text_pred, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, PRED_KP_COLOR, 2)

    return vis_image
