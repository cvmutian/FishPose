import numpy as np
from typing import List, Dict, Any

def compute_oks(pred_kpts: np.ndarray, gt_kpts: np.ndarray, gt_bbox: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    variances = (sigmas * 2)**2
    
    gt_bbox_w = gt_bbox[2]
    gt_bbox_h = gt_bbox[3]
    scale = gt_bbox_w * gt_bbox_h if gt_bbox_w > 0 and gt_bbox_h > 0 else 1.0

    dx = pred_kpts[:, 0] - gt_kpts[:, 0]
    dy = pred_kpts[:, 1] - gt_kpts[:, 1]
    
    visible = gt_kpts[:, 2] > 0
    
    e = (dx**2 + dy**2) / variances / (scale + np.finfo(float).eps) / 2
    
    oks = np.sum(np.exp(-e) * visible) / np.sum(visible) if np.sum(visible) > 0 else 0.0
    return oks

def evaluate_oks_apm(
    all_preds: List[Dict[str, Any]],
    all_gts: List[Dict[str, Any]],
    oks_sigmas: np.ndarray
) -> Dict[str, float]:
    
    oks_thresholds = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    
    num_thresholds = len(oks_thresholds)
    num_gts = len(all_gts)
    
    all_oks = []
    all_scores = []
    
    for pred_info, gt_info in zip(all_preds, all_gts):
        pred_kpts = pred_info['keypoints'][0]
        gt_kpts = gt_info['keypoints'][0]
        gt_bbox = gt_info['bbox'][0]
        score = pred_info['scores'][0]
        
        oks = compute_oks(pred_kpts, gt_kpts, gt_bbox, np.array(oks_sigmas))
        all_oks.append(oks)
        all_scores.append(score)

    all_oks = np.array(all_oks)
    all_scores = np.array(all_scores)
    
    sorted_indices = np.argsort(-all_scores)
    
    true_positives = np.zeros(num_thresholds)
    false_positives = np.zeros(num_thresholds)
    
    precision = np.zeros((len(all_scores), num_thresholds))
    recall = np.zeros((len(all_scores), num_thresholds))
    
    for i, idx in enumerate(sorted_indices):
        for t, threshold in enumerate(oks_thresholds):
            if all_oks[idx] >= threshold:
                true_positives[t] += 1
            else:
                false_positives[t] += 1
            
            precision[i, t] = true_positives[t] / (true_positives[t] + false_positives[t])
            recall[i, t] = true_positives[t] / num_gts if num_gts > 0 else 0.0

    ap = np.zeros(num_thresholds)
    for t in range(num_thresholds):
        if precision[:, t].size > 0:
            ap[t] = np.mean(precision[recall[:, t] > 0.0, t]) if np.sum(recall[:, t] > 0.0) > 0 else 0.0

    mean_ap = np.mean(ap)
    
    final_recall = recall[-1, :] if len(recall) > 0 else np.zeros(num_thresholds)
    mean_ar = np.mean(final_recall)
    
    return {
        'AP': mean_ap,
        'AP@.50': ap[0],
        'AP@.75': ap[5],
        'AR': mean_ar,
    }
