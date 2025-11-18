import numpy as np
from typing import Tuple

def decode_keypoints_from_heatmap(
    heatmaps: np.ndarray, 
    offsets: np.ndarray, 
    img_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
    confidence_threshold: float = 0.1
):
    batch_size, num_keypoints, heatmap_h, heatmap_w = heatmaps.shape
    
    flat_heatmaps = heatmaps.reshape(batch_size, num_keypoints, -1)
    max_indices = np.argmax(flat_heatmaps, axis=2)
    max_scores = np.max(flat_heatmaps, axis=2)
    
    y_coords = max_indices // heatmap_w
    x_coords = max_indices % heatmap_w
    
    decoded_keypoints = np.zeros((batch_size, num_keypoints, 3), dtype=np.float32)
    
    for i in range(batch_size):
        for j in range(num_keypoints):
            hm_y, hm_x = y_coords[i, j], x_coords[i, j]
            
            offset_x = offsets[i, j, 0, hm_y, hm_x]
            offset_y = offsets[i, j, 1, hm_y, hm_x]
            
            final_x = hm_x + offset_x
            final_y = hm_y + offset_y
            
            score = max_scores[i, j]
            if score >= confidence_threshold:
                decoded_keypoints[i, j] = [final_x, final_y, score]
            else:
                decoded_keypoints[i, j] = [0, 0, 0]
                
    scale_x = img_size[1] / heatmap_size[1]
    scale_y = img_size[0] / heatmap_size[0]
    
    decoded_keypoints[:, :, 0] *= scale_x
    decoded_keypoints[:, :, 1] *= scale_y
    
    return decoded_keypoints
