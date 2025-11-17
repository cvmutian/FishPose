import cv2
import numpy as np

# Define a color palette for keypoints and boxes for better visualization
KP_COLOR = (0, 255, 0)  # Green for Ground Truth Keypoints
PRED_KP_COLOR = (255, 0, 0) # Blue for Predicted Keypoints
BOX_COLOR = (0, 255, 0) # Green for Ground Truth Boxes
PRED_BOX_COLOR = (255, 0, 0) # Blue for Predicted Boxes

def draw_keypoints_and_boxes(
    image,
    pred_keypoints,
    pred_boxes,
    gt_keypoints,
    gt_boxes,
    keypoint_names=None
):
    """
    Draws predicted and ground truth keypoints and bounding boxes on an image.

    Args:
        image (np.ndarray): The image to draw on (in BGR format).
        pred_keypoints (list or np.ndarray): Predicted keypoints for instances. 
            Shape: (num_instances, num_keypoints, 3) where each keypoint is (x, y, score/visibility).
        pred_boxes (list or np.ndarray): Predicted bounding boxes. Shape: (num_instances, 4) as (x1, y1, x2, y2).
        gt_keypoints (list or np.ndarray): Ground truth keypoints. Shape similar to pred_keypoints.
        gt_boxes (list or np.ndarray): Ground truth bounding boxes. Shape similar to pred_boxes.
        keypoint_names (list of str, optional): Names for each keypoint index.

    Returns:
        np.ndarray: The image with annotations.
    """
    vis_image = image.copy()
    
    # Draw predicted keypoints
    if pred_keypoints:
        for instance_kps in pred_keypoints:
            for i, (x, y, v) in enumerate(instance_kps):
                if v > 0:
                    cv2.circle(vis_image, (int(x), int(y)), 3, PRED_KP_COLOR, -1)
                    cv2.putText(vis_image, str(i), (int(x) + 2, int(y) - 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRED_KP_COLOR, 1)

    # Draw ground truth keypoints
    if gt_keypoints is not None:
        for instance_kps in gt_keypoints:
            for i, (x, y, v) in enumerate(instance_kps):
                if v > 0:
                    cv2.circle(vis_image, (int(x), int(y)), 3, KP_COLOR, -1)
                    cv2.putText(vis_image, str(i), (int(x) + 2, int(y) - 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, KP_COLOR, 1)
    
    # Draw Ground Truth Bounding Boxes (Green)
    if gt_boxes is not None:
        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), BOX_COLOR, 2)

    # Draw Predicted Bounding Boxes (Blue)
    if pred_boxes is not None:
        for box in pred_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), PRED_BOX_COLOR, 2)
            
    # Add a legend
    legend_text_gt = "Green: Ground Truth"
    legend_text_pred = "Blue: Prediction"
    cv2.putText(vis_image, legend_text_gt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, KP_COLOR, 2)
    cv2.putText(vis_image, legend_text_pred, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, PRED_KP_COLOR, 2)

    return vis_image
