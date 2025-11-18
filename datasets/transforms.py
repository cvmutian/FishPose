import torch
import random
import cv2
import numpy as np
from torchvision.transforms import functional as F
import torchvision.transforms as transforms_tv
from typing import List, Tuple

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bboxes, keypoints):
        for t in self.transforms:
            image, bboxes, keypoints = t(image, bboxes, keypoints)
        return image, bboxes, keypoints

class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.transform = transforms_tv.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, image, bboxes, keypoints):
        pil_image = F.to_pil_image(image)
        jittered_pil = self.transform(pil_image)
        jittered_image = np.array(jittered_pil)
        return jittered_image, bboxes, keypoints

class ToTensor:
    def __call__(self, image, bboxes, keypoints):
        return F.to_tensor(image), bboxes, keypoints

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, bboxes, keypoints):
        image = F.normalize(image, self.mean, self.std)
        return image, bboxes, keypoints

class RandomAffine:
    def __init__(self, degrees=0, translate=None, scale=None, shear=None):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (list, tuple)) and len(degrees) == 2
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (list, tuple)) and len(translate) == 2
            self.translate = translate
        else:
            self.translate = (0.0, 0.0)

        if scale is not None:
            assert isinstance(scale, (list, tuple)) and len(scale) == 2
            self.scale = scale
        else:
            self.scale = (1.0, 1.0)
            
        if isinstance(shear, (int, float)):
            self.shear = (-shear, shear)
        elif shear is not None:
            assert isinstance(shear, (list, tuple)) and len(shear) == 2
            self.shear = shear
        else:
            self.shear = (0.0, 0.0)

    def __call__(self, image, bboxes, keypoints):
        h, w, _ = image.shape
        center = (w / 2, h / 2)

        angle = random.uniform(self.degrees[0], self.degrees[1])
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        tx = random.uniform(-self.translate[0] * w, self.translate[0] * w)
        ty = random.uniform(-self.translate[1] * h, self.translate[1] * h)
        shear_x = random.uniform(self.shear[0], self.shear[1])
        
        M = cv2.getRotationMatrix2D(center, angle, scale_factor)
        
        M[0, 2] += tx
        M[1, 2] += ty
        
        shear_matrix = np.array([[1, np.tan(np.deg2rad(shear_x)), 0], 
                                 [0, 1, 0]], dtype=np.float32)
        M = np.dot(shear_matrix, np.vstack([M, [0, 0, 1]]))[:2]

        transformed_image = cv2.warpAffine(image, M, (w, h))
        
        if bboxes.shape[0] > 0:
            pass

        if keypoints.shape[0] > 0:
            transformed_keypoints = transform_coords(keypoints.reshape(-1, 3), M).reshape(keypoints.shape)
        else:
            transformed_keypoints = keypoints
            
        return transformed_image, bboxes, transformed_keypoints


def transform_coords(coords, M):
    xy = coords[:, :2]
    v = coords[:, 2:]
    
    xy1 = np.hstack([xy, np.ones((xy.shape[0], 1))])
    
    transformed_xy = np.dot(xy1, M.T)
    
    return np.hstack([transformed_xy, v])


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, bboxes, keypoints):
        if random.random() < self.p:
            h, w, _ = image.shape
            image = cv2.flip(image, 1)

            if bboxes.shape[0] > 0:
                bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
            
            if keypoints.shape[0] > 0:
                keypoints[:, :, 0] = w - keypoints[:, :, 0]

        return image, bboxes, keypoints


class Resize:
    def __init__(self, size: Tuple[int, int]):
        self.target_h, self.target_w = size

    def __call__(self, image, bboxes, keypoints):
        original_h, original_w, _ = image.shape

        resized_image = cv2.resize(image, (self.target_w, self.target_h))

        scale_w = self.target_w / original_w
        scale_h = self.target_h / original_h

        if bboxes.shape[0] > 0:
            bboxes[:, [0, 2]] *= scale_w
            bboxes[:, [1, 3]] *= scale_h

        if keypoints.shape[0] > 0:
            keypoints[:, :, 0] *= scale_w
            keypoints[:, :, 1] *= scale_h
            
        return resized_image, bboxes, keypoints
