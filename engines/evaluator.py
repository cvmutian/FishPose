import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Any

from utils.metrics import evaluate_oks_apm
from utils.dist import is_main_process
from models.fish_pose_model import FishPoseModel


def get_final_preds(heatmaps: torch.Tensor, coords_offset: torch.Tensor):
    batch_size, num_keypoints, h, w = heatmaps.shape
    
    heatmaps_flat = heatmaps.view(batch_size, num_keypoints, -1)
    max_vals, max_indices = torch.max(heatmaps_flat, dim=2)
    
    y_coords = (max_indices // w).float()
    x_coords = (max_indices % w).float()

    initial_preds = torch.stack([x_coords, y_coords], dim=2)
    final_preds = initial_preds.clone()

    for i in range(batch_size):
        for j in range(num_keypoints):
            px, py = int(x_coords[i, j]), int(y_coords[i, j])
            if 0 <= py < h and 0 <= px < w:
                offset_x = coords_offset[i, j * 2, py, px]
                offset_y = coords_offset[i, j * 2 + 1, py, px]
                final_preds[i, j, 0] += offset_x
                final_preds[i, j, 1] += offset_y
    
    for i in range(batch_size):
        for j in range(num_keypoints):
            px_f, py_f = final_preds[i, j, 0], final_preds[i, j, 1]
            px, py = int(px_f), int(py_f)
            hm = heatmaps[i, j]
            
            if 1 < px < w - 1 and 1 < py < h - 1:
                dx = hm[py, px + 1] - hm[py, px - 1]
                dy = hm[py + 1, px] - hm[py - 1, px]
                final_preds[i, j, 0] += torch.sign(dx) * 0.25
                final_preds[i, j, 1] += torch.sign(dy) * 0.25

    return final_preds, max_vals

class Evaluator:
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], device: torch.device):
        self.model = model
        self.config = config
        self.device = device

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        
        all_preds_list = []
        all_gts_list = []

        for batch in tqdm(data_loader, desc="Evaluating", disable=not is_main_process()):
            images = batch['image'].to(self.device, non_blocking=True)
            keypoints_gt = batch['keypoints']

            if not isinstance(self.model, FishPoseModel):
                 self.model = FishPoseModel(self.config).to(self.device).eval()

            model_outputs = self.model(images)
            final_outputs = model_outputs['final_outputs']
            
            preds_coords, preds_scores = get_final_preds(
                final_outputs['heatmaps'].cpu(),
                final_outputs['coords_offset'].cpu()
            )

            img_h, img_w = self.config['data']['img_size']
            heatmap_h, heatmap_w = self.config['data']['heatmap_size']
            scale_x = img_w / heatmap_w
            scale_y = img_h / heatmap_h
            
            preds_coords[:, :, 0] *= scale_x
            preds_coords[:, :, 1] *= scale_y

            for i in range(images.size(0)):
                pred_kpts = np.zeros((1, self.config['data']['num_keypoints'], 3))
                pred_kpts[0, :, :2] = preds_coords[i].numpy()
                pred_kpts[0, :, 2] = 2
                
                score = preds_scores[i].mean().item()

                all_preds_list.append({'keypoints': pred_kpts, 'scores': np.array([score])})

                gt_kpts = keypoints_gt[i].unsqueeze(0).numpy()
                valid_kpts = gt_kpts[0, gt_kpts[0, :, 2] > 0]
                bbox = [0,0,0,0]
                if len(valid_kpts) > 0:
                    x_min, y_min = valid_kpts[:, :2].min(axis=0)
                    x_max, y_max = valid_kpts[:, :2].max(axis=0)
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                all_gts_list.append({'keypoints': gt_kpts, 'bbox': np.array([bbox])})
        
        if is_main_process():
            return evaluate_oks_apm(
                all_preds_list, all_gts_list, self.config['evaluation']['oks_sigmas']
            )
        
        return {}
