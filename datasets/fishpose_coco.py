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
        
        # Expose the raw COCO object for external use (like in visualization)
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
        
        # --- FIX: Sanitize file_name path for cross-platform compatibility ---
        file_name = img_info['file_name'].replace('\\\\', '/').replace('\\', '/')
        # Construct the full image path by joining the root directory and the file_name from COCO
        img_path = os.path.join(self.root_dir, file_name)
        
        image = cv2.imread(img_path)
        
        # Check if image was loaded correctly
        if image is None:
            print(f"\nWarning: Could not read image file, skipping: {img_path}")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]

        annotations = self.annotations_by_img_id[img_id]
        
        keypoints_list = [np.array(ann['keypoints']).reshape(self.num_keypoints, 3) for ann in annotations]
        bboxes_list = [np.array(ann['bbox']) for ann in annotations]
        
        # Convert to numpy arrays for transforms
        image = np.array(image)
        keypoints_array = np.array(keypoints_list)
        bboxes_array = np.array(bboxes_list)

        # Apply transforms
        if self.transform:
            image, bboxes_array, keypoints_array = self.transform(image, bboxes_array, keypoints_array)

        # --- DEBUG: Print keypoints BEFORE generating targets ---
        # print("\n--- Keypoints BEFORE target generation ---")
        # print(keypoints_array)

        # Generate heatmap and offset targets
        heatmap_target = self._generate_heatmap_target(keypoints_array.copy()) # Use a copy to prevent mutation
        offset_targets = self._generate_offset_and_mask_targets(keypoints_array.copy()) # Use a copy to prevent mutation
        
        # --- DEBUG: Print keypoints AFTER generating targets ---
        # print("\n--- Keypoints AFTER target generation ---")
        # print(keypoints_array)

        # The image should be a tensor after transforms
        image_tensor = image

        return {
            'image': image_tensor,
            'image_id': img_id,
            'heatmap_target': torch.from_numpy(heatmap_target).float(),
            'offset_target': torch.from_numpy(offset_targets['offset_target']).float(),
            'kpt_mask': torch.from_numpy(offset_targets['kpt_mask']).bool(),
            'keypoints': torch.from_numpy(keypoints_array), # Add this line for visualization/evaluation
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
        """Generate SimCC (Soft-argmax Integral Matching for Coordinate Classification) targets."""
        simcc_x = np.zeros((self.num_keypoints, self.img_size[1]), dtype=np.float32)
        simcc_y = np.zeros((self.num_keypoints, self.img_size[0]), dtype=np.float32)

        # We only generate targets for the first instance in the image for simplicity.
        # A more robust implementation might handle multiple instances or choose the largest one.
        if not keypoints_list:
            return {"simcc_x": simcc_x, "simcc_y": simcc_y}
            
        keypoints = keypoints_list[0]

        for i in range(self.num_keypoints):
            if keypoints[i, 2] > 0:
                x, y = keypoints[i, :2]
                
                # Create 1D Gaussian distribution around the target coordinate
                x_range = np.arange(self.img_size[1])
                y_range = np.arange(self.img_size[0])

                ux = int(x)
                uy = int(y)
                
                # Adjust sigma for 1D distribution, can be a tunable parameter
                sigma_1d = self.sigma * 3 

                if 0 <= ux < self.img_size[1]:
                    dist_x = (x_range - ux)**2
                    exponent_x = dist_x / (2 * sigma_1d**2)
                    simcc_x[i] = np.exp(-exponent_x)
                
                if 0 <= uy < self.img_size[0]:
                    dist_y = (y_range - uy)**2
                    exponent_y = dist_y / (2 * sigma_1d**2)
                    simcc_y[i] = np.exp(-exponent_y)
        
        # Normalize to be a probability distribution
        simcc_x /= (simcc_x.sum(axis=1, keepdims=True) + 1e-9)
        simcc_y /= (simcc_y.sum(axis=1, keepdims=True) + 1e-9)

        return {"simcc_x": simcc_x, "simcc_y": simcc_y}


    def _generate_offset_and_mask_targets(self, keypoints_list: List[np.ndarray]) -> dict:
        """Generate keypoint offset and mask targets."""
        # The offset target should match the model's output shape: (K*2, H, W)
        # All Y offsets first, then all X offsets
        offset_target = np.zeros((self.num_keypoints * 2, self.heatmap_size[0], self.heatmap_size[1]), dtype=np.float32)
        # The mask should also be expanded to match the offset target shape
        kpt_mask = np.zeros((self.num_keypoints * 2, self.heatmap_size[0], self.heatmap_size[1]), dtype=bool)
        
        scale_x = self.heatmap_size[1] / self.img_size[1]
        scale_y = self.heatmap_size[0] / self.img_size[0]

        for keypoints in keypoints_list:
            for i in range(self.num_keypoints):
                if keypoints[i, 2] > 0:
                    x, y = keypoints[i, :2]
                    
                    # Scaled coordinates on the heatmap
                    hm_x_float = x * scale_x
                    hm_y_float = y * scale_y
                    
                    # Integer coordinates
                    hm_x_int = int(hm_x_float)
                    hm_y_int = int(hm_y_float)
                    
                    if 0 <= hm_x_int < self.heatmap_size[1] and 0 <= hm_y_int < self.heatmap_size[0]:
                        # Calculate offset
                        offset_x = hm_x_float - hm_x_int
                        offset_y = hm_y_float - hm_y_int
                        
                        # --- REVERT: Populate targets in the original (Y-first, then X-first) format ---
                        offset_target[i, hm_y_int, hm_x_int] = offset_y
                        offset_target[self.num_keypoints + i, hm_y_int, hm_x_int] = offset_x
                        
                        # Set mask for both y and x offset channels
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

    # --- Post-processing and decoding utility ---
    # This function might be used across different scripts for evaluation or visualization
    
    def decode_keypoints_from_heatmap(self, heatmaps, offsets, original_img_size, confidence_threshold=0.1):
        """
        Decodes keypoints from heatmap and offsets.

        Args:
            heatmaps (torch.Tensor): Heatmap tensor of shape (B, K, H, W).
            offsets (torch.Tensor): Offset tensor of shape (B, K*2, H, W).
            original_img_size (list or tuple): The original image size [height, width].
            confidence_threshold (float): Minimum confidence to consider a keypoint valid.

        Returns:
            list: A list of predicted keypoints for each image in the batch.
                  Each element is a list of instances, where each instance is a numpy array
                  of shape (num_keypoints, 3) with (x, y, score).
        """
        heatmaps = torch.sigmoid(heatmaps)
        B, K, H, W = heatmaps.shape
        
        # Find the max value and its index in each heatmap channel
        max_scores, max_indices = torch.max(heatmaps.view(B, K, -1), dim=2)
        
        max_indices = max_indices.view(B, K, 1)
        max_indices_np = max_indices.cpu().numpy()
        max_scores_np = max_scores.cpu().numpy()

        preds = []
        for i in range(B):
            # We assume a single instance per image for this decoding logic
            keypoints = np.zeros((K, 3), dtype=np.float32)
            for j in range(K):
                hm_index = max_indices_np[i, j, 0]
                # Get coordinates from the flattened index
                y_coord = hm_index // W
                x_coord = hm_index % W
                
                score = max_scores_np[i, j]

                # --- Apply confidence threshold ---
                if score < confidence_threshold:
                    continue # Skip this keypoint if confidence is too low

                # Get offsets
                offset_y = offsets[i, j, y_coord, x_coord].item()
                offset_x = offsets[i, self.num_keypoints + j, y_coord, x_coord].item()

                # Calculate final coordinates in heatmap scale
                final_x = x_coord + offset_x
                final_y = y_coord + offset_y

                # Scale coordinates to original image size
                stride_x = original_img_size[1] / W
                stride_y = original_img_size[0] / H
                
                keypoints[j, 0] = final_x * stride_x
                keypoints[j, 1] = final_y * stride_y
                keypoints[j, 2] = score
            
            # For simplicity, we wrap it in a list to represent a single instance
            preds.append([keypoints])
            
        return preds
