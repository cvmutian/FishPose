import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional, List, Tuple
from torchvision.transforms import functional as F
from pycocotools.coco import COCO

class FishPoseCocoDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        annotations_file: str,
        img_size: tuple,
        heatmap_size: tuple,
        sigma: float,
        num_keypoints: int,
        transform: Optional[Callable] = None,
        output_stride: int = 4,
    ):
        self.root_dir = root_dir
        self.annotations_file = os.path.join(root_dir, annotations_file)
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.num_keypoints = num_keypoints
        self.transform = transform
        self.output_stride = output_stride

        with open(self.annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        self.coco = COCO(self.annotations_file)

        self.images = {img['id']: img for img in coco_data['images']}
        self.annotations_by_img_id = self._group_annotations(coco_data['annotations'])
        
        self.img_ids = sorted(list(self.annotations_by_img_id.keys()))

    def _group_annotations(self, annotations):
        ann_by_img_id = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in ann_by_img_id:
                ann_by_img_id[img_id] = []
            ann_by_img_id[img_id].append(ann)
        return ann_by_img_id

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        
        file_name = img_info['file_name'].replace('\\\\', '/').replace('\\', '/')
        img_path = os.path.join(self.root_dir, file_name)
        
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"\nWarning: Could not read image file, skipping: {img_path}")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]

        annotations = self.annotations_by_img_id[img_id]
        
        keypoints_list = [np.array(ann['keypoints']).reshape(self.num_keypoints, 3) for ann in annotations]
        bboxes_list = [np.array(ann['bbox']) for ann in annotations]
        
        image = np.array(image)
        keypoints_array = np.array(keypoints_list)
        bboxes_array = np.array(bboxes_list)

        if self.transform:
            image, bboxes_array, keypoints_array = self.transform(image, bboxes_array, keypoints_array)

        heatmap_target = self._generate_heatmap_target(keypoints_array.copy())
        offset_targets = self._generate_offset_and_mask_targets(keypoints_array.copy())

        image_tensor = image

        return {
            'image': image_tensor,
            'image_id': img_id,
            'heatmap_target': torch.from_numpy(heatmap_target).float(),
            'offset_target': torch.from_numpy(offset_targets['offset_target']).float(),
            'kpt_mask': torch.from_numpy(offset_targets['kpt_mask']).bool(),
            'keypoints': torch.from_numpy(keypoints_array),
        }

    def _generate_heatmap_target(self, keypoints_list: List[np.ndarray]) -> np.ndarray:
        heatmap = np.zeros((self.num_keypoints, self.heatmap_size[0], self.heatmap_size[1]), dtype=np.float32)
        
        scale_x = self.heatmap_size[1] / self.img_size[1]
        scale_y = self.heatmap_size[0] / self.img_size[0]
        
        for keypoints in keypoints_list:
            for i in range(self.num_keypoints):
                if keypoints[i, 2] > 0:
                    x, y = keypoints[i, :2]
                    
                    hm_x = int(x * scale_x)
                    hm_y = int(y * scale_y)
                    
                    if 0 <= hm_x < self.heatmap_size[1] and 0 <= hm_y < self.heatmap_size[0]:
                        x_range = np.arange(self.heatmap_size[1])
                        y_range = np.arange(self.heatmap_size[0])
                        xx, yy = np.meshgrid(x_range, y_range)
                        
                        dist_sq = (xx - hm_x)**2 + (yy - hm_y)**2
                        exponent = dist_sq / (2 * self.sigma**2)
                        heatmap[i] = np.maximum(heatmap[i], np.exp(-exponent))
        return heatmap

    def _generate_simcc_targets(self, keypoints_list: List[np.ndarray]) -> dict:
        simcc_x = np.zeros((self.num_keypoints, self.img_size[1]), dtype=np.float32)
        simcc_y = np.zeros((self.num_keypoints, self.img_size[0]), dtype=np.float32)

        if not keypoints_list:
            return {"simcc_x": simcc_x, "simcc_y": simcc_y}
            
        keypoints = keypoints_list[0]

        for i in range(self.num_keypoints):
            if keypoints[i, 2] > 0:
                x, y = keypoints[i, :2]
                
                x_range = np.arange(self.img_size[1])
                y_range = np.arange(self.img_size[0])

                ux = int(x)
                uy = int(y)
                
                sigma_1d = self.sigma * 3 

                if 0 <= ux < self.img_size[1]:
                    dist_x = (x_range - ux)**2
                    exponent_x = dist_x / (2 * sigma_1d**2)
                    simcc_x[i] = np.exp(-exponent_x)
                
                if 0 <= uy < self.img_size[0]:
                    dist_y = (y_range - uy)**2
                    exponent_y = dist_y / (2 * sigma_1d**2)
                    simcc_y[i] = np.exp(-exponent_y)
        
        simcc_x /= (simcc_x.sum(axis=1, keepdims=True) + 1e-9)
        simcc_y /= (simcc_y.sum(axis=1, keepdims=True) + 1e-9)

        return {"simcc_x": simcc_x, "simcc_y": simcc_y}


    def _generate_offset_and_mask_targets(self, keypoints_list: List[np.ndarray]) -> dict:
        offset_target = np.zeros((self.num_keypoints * 2, self.heatmap_size[0], self.heatmap_size[1]), dtype=np.float32)
        kpt_mask = np.zeros((self.num_keypoints * 2, self.heatmap_size[0], self.heatmap_size[1]), dtype=bool)
        
        scale_x = self.heatmap_size[1] / self.img_size[1]
        scale_y = self.heatmap_size[0] / self.img_size[0]

        for keypoints in keypoints_list:
            for i in range(self.num_keypoints):
                if keypoints[i, 2] > 0:
                    x, y = keypoints[i, :2]
                    
                    hm_x_float = x * scale_x
                    hm_y_float = y * scale_y
                    
                    hm_x_int = int(hm_x_float)
                    hm_y_int = int(hm_y_float)
                    
                    if 0 <= hm_x_int < self.heatmap_size[1] and 0 <= hm_y_int < self.heatmap_size[0]:
                        offset_x = hm_x_float - hm_x_int
                        offset_y = hm_y_float - hm_y_int
                        
                        offset_target[i, hm_y_int, hm_x_int] = offset_y
                        offset_target[self.num_keypoints + i, hm_y_int, hm_x_int] = offset_x
                        
                        kpt_mask[i, hm_y_int, hm_x_int] = True
                        kpt_mask[self.num_keypoints + i, hm_y_int, hm_x_int] = True
                        
        return {"offset_target": offset_target, "kpt_mask": kpt_mask}


    def _generate_detection_targets(self, bboxes_list: List[np.ndarray]) -> dict:
        feat_h, feat_w = self.img_size[0] // self.output_stride, self.img_size[1] // self.output_stride
        
        cls_target = np.zeros((1, feat_h, feat_w), dtype=np.float32)
        reg_target = np.zeros((4, feat_h, feat_w), dtype=np.float32)
        cen_target = np.zeros((1, feat_h, feat_w), dtype=np.float32)
        reg_mask = np.zeros((1, feat_h, feat_w), dtype=bool)

        for bbox in bboxes_list:
            x, y, w, h = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h

            gx1 = x1 / self.output_stride
            gy1 = y1 / self.output_stride
            gx2 = x2 / self.output_stride
            gy2 = y2 / self.output_stride
            
            cx = int((gx1 + gx2) / 2)
            cy = int((gy1 + gy2) / 2)

            if not (0 <= cx < feat_w and 0 <= cy < feat_h):
                continue

            cls_target[0, cy, cx] = 1
            reg_mask[0, cy, cx] = True

            l = cx - gx1
            t = cy - gy1
            r = gx2 - cx
            b = gy2 - cy
            reg_target[:, cy, cx] = [l, t, r, b]

            l_star, r_star = reg_target[0, cy, cx], reg_target[2, cy, cx]
            t_star, b_star = reg_target[1, cy, cx], reg_target[3, cy, cx]
            centerness = np.sqrt((min(l_star, r_star) / (max(l_star, r_star) + 1e-9)) * \
                               (min(t_star, b_star) / (max(t_star, b_star) + 1e-9)))
            cen_target[0, cy, cx] = centerness
        
        reg_target /= 1.0

        return {
            "cls_target": cls_target,
            "reg_target": reg_target,
            "cen_target": cen_target,
        }
    
    def decode_keypoints_from_heatmap(self, heatmaps, offsets, original_img_size, confidence_threshold=0.1):
        heatmaps = torch.sigmoid(heatmaps)
        B, K, H, W = heatmaps.shape
        
        max_scores, max_indices = torch.max(heatmaps.view(B, K, -1), dim=2)
        
        max_indices = max_indices.view(B, K, 1)
        max_indices_np = max_indices.cpu().numpy()
        max_scores_np = max_scores.cpu().numpy()

        preds = []
        for i in range(B):
            keypoints = np.zeros((K, 3), dtype=np.float32)
            for j in range(K):
                hm_index = max_indices_np[i, j, 0]
                y_coord = hm_index // W
                x_coord = hm_index % W
                
                score = max_scores_np[i, j]

                if score < confidence_threshold:
                    continue

                offset_y = offsets[i, j, y_coord, x_coord].item()
                offset_x = offsets[i, self.num_keypoints + j, y_coord, x_coord].item()

                final_x = x_coord + offset_x
                final_y = y_coord + offset_y

                stride_x = original_img_size[1] / W
                stride_y = original_img_size[0] / H
                
                keypoints[j, 0] = final_x * stride_x
                keypoints[j, 1] = final_y * stride_y
                keypoints[j, 2] = score
            
            preds.append([keypoints])
            
        return preds
