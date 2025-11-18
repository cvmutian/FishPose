import argparse
import yaml
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import List, Tuple

from models.fish_pose_model import FishPoseModel
from datasets.fishpose_coco import FishPoseCocoDataset
from datasets import transforms as T
from utils.keypoints import decode_keypoints_from_heatmap

def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    elem = batch[0]
    collated = {}
    for key in elem:
        if key in ['keypoints', 'offset_target', 'kpt_mask', 'heatmap_target']: 
            collated[key] = [d[key] for d in batch]
        else:
            try:
                collated[key] = torch.utils.data.default_collate([d[key] for d in batch])
            except RuntimeError:
                collated[key] = [d[key] for d in batch]
    return collated

def decode_instance(
    heatmaps: torch.Tensor, 
    offsets: torch.Tensor, 
    num_keypoints: int,
    img_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
    confidence_threshold: float = 0.3
) -> np.ndarray:
    B, K, H, W = heatmaps.shape
    
    max_scores, max_indices = torch.max(heatmaps.view(B, K, -1), dim=2)
    max_indices = max_indices.view(B, K, 1)
    
    max_indices_np = max_indices.cpu().numpy()
    max_scores_np = max_scores.cpu().numpy()
    offsets_np = offsets.cpu().numpy()

    pred_keypoints = np.zeros((K, 3), dtype=np.float32)

    for j in range(K):
        hm_index = max_indices_np[0, j, 0]
        y_coord = hm_index // W
        x_coord = hm_index % W
        
        score = max_scores_np[0, j].item()

        if score < confidence_threshold:
            continue

        offset_y = offsets_np[0, j, y_coord, x_coord]
        offset_x = offsets_np[0, num_keypoints + j, y_coord, x_coord]

        final_x = x_coord + offset_x
        final_y = y_coord + offset_y

        stride_x = img_size[1] / W
        stride_y = img_size[0] / H
        
        pred_keypoints[j, 0] = final_x * stride_x
        pred_keypoints[j, 1] = final_y * stride_y
        pred_keypoints[j, 2] = score
        
    return pred_keypoints

def run_evaluation(config: dict, weights_path: str):
    device = torch.device(config['device'])
    
    img_size = config['data']['img_size']
    test_transforms = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    gt_annotations_path = os.path.join(config['data']['root_dir'], config['data']['test_json'])

    test_dataset = FishPoseCocoDataset(
        root_dir=config['data']['root_dir'],
        annotations_file=config['data']['test_json'],
        img_size=config['data']['img_size'],
        heatmap_size=config['data']['heatmap_size'],
        sigma=config['data']['sigma'],
        num_keypoints=config['data']['num_keypoints'],
        output_stride=config['data']['output_stride'],
        transform=test_transforms
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['evaluation']['num_workers'],
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    model = FishPoseModel(config).to(device)
    
    print(f"Loading model weights from: {weights_path}")
    
    try:
        loaded_data = torch.load(weights_path, map_location=device, weights_only=True)
    except (TypeError, AttributeError):
        loaded_data = torch.load(weights_path, map_location=device, weights_only=False)
    
    if isinstance(loaded_data, dict) and 'state_dict' in loaded_data:
        state_dict = loaded_data['state_dict']
    else:
        state_dict = loaded_data
    
    if any('facade_backbone' in key for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('backbone.facade_backbone', 'backbone.ssm_pyramid_backbone')
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    coco_results = []

    coco_gt = COCO(gt_annotations_path)

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating on Test Set")
        for batch in pbar:
            if batch is None:
                continue
                
            images = batch['image'].to(device)
            image_ids = [id_tensor.item() for id_tensor in batch['image_id']]
            
            predictions = model(images)
            pred_heatmap = torch.sigmoid(predictions['pose_heatmap'])
            pred_offsets = predictions['offsets']
            
            for i in range(images.shape[0]):
                img_id = image_ids[i]

                decoded_keypoints = decode_instance(
                    pred_heatmap[i].unsqueeze(0).cpu(),
                    pred_offsets[i].unsqueeze(0).cpu(),
                    num_keypoints=config['data']['num_keypoints'],
                    img_size=config['data']['img_size'],
                    heatmap_size=config['data']['heatmap_size']
                )

                img_info = coco_gt.loadImgs(img_id)[0]
                original_w = img_info['width']
                original_h = img_info['height']
                
                resized_h, resized_w = config['data']['img_size']
                
                scale_x = original_w / resized_w
                scale_y = original_h / resized_h
                
                decoded_keypoints[:, 0] *= scale_x
                decoded_keypoints[:, 1] *= scale_y
                
                coco_kpts = np.zeros(config['data']['num_keypoints'] * 3, dtype=np.float32)
                valid_kpts_mask = decoded_keypoints[:, 2] > 0
                
                coco_kpts[0::3] = decoded_keypoints[:, 0]
                coco_kpts[1::3] = decoded_keypoints[:, 1]
                coco_kpts[2::3][valid_kpts_mask] = 2
                
                avg_score = decoded_keypoints[valid_kpts_mask, 2].mean() if valid_kpts_mask.any() else 0.0

                coco_results.append({
                    'image_id': img_id,
                    'category_id': 1,
                    'keypoints': coco_kpts.tolist(),
                    'score': float(avg_score)
                })

    coco_dt = coco_gt.loadRes(coco_results)
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.params.imgIds = coco_gt.getImgIds()
    coco_eval.params.kpt_oks_sigmas = np.array(config['evaluation']['oks_sigmas'])
    coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2]]
    coco_eval.params.areaRngLbl = ['all']
    
    print("\n--- COCO Keypoint Evaluation Results ---")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a FishPose model.")
    parser.add_argument('--weights', type=str, default='weights/best_model.pth', help='Path to the model weights file.')
    parser.add_argument('--config', type=str, default='configs/FishPose.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    run_evaluation(config, args.weights)
