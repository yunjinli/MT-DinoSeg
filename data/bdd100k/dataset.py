from torch.utils.data import Dataset, default_collate
import glob
import numpy as np
from PIL import Image
import torch 
import torchvision.transforms as T
from typing import Tuple, List
import cv2 
import os
from typing import Dict

class MultitaskBDD100k(Dataset):
    def __init__(self, image_base: str, 
                        label_base: Dict, 
                        image_transform, 
                        mask_transform, 
                        label_colors_list: Dict, 
                        class_names: Dict, 
                        split, 
                        seg_type,
                        ignore_index,
                        ):
        self.image_paths = glob.glob(f"{image_base}/*")
        # self.label_paths = glob.glob(f"{label_base}/*")
        self.label_paths = {}
        # print(label_base)
        for key, value in label_base.items():
            # print(value)
            self.label_paths[key] = glob.glob(f"{value}/*")
            self.label_paths[key].sort()
            # print(glob.glob(f"{value}/*"))
        # print(self.label_paths)
        self.image_paths.sort()

        self.tasks = label_base.keys()
        # print(self.tasks)
        # for t in self.tasks:
        #     print(t)
        # self.label_paths.sort()

        # for key, value in self.label_paths.items():
        #     for img_path, label_path in zip(self.image_paths, value):
        #         assert os.path.basename(img_path).split('.')[0] == os.path.basename(label_path).split('.')[0], f"{os.path.basename(img_path).split('.')[0]} /= {os.path.basename(label_path).split('.')[0]}"
        
        # self.label_colors_list = label_colors_list
        self.label_colors_list = {}
        for key, value in label_colors_list.items():
            self.label_colors_list[key] = value
        
        # self.class_names = class_names
        self.class_names = {}
        for key, value in class_names.items():
            self.class_names[key] = value

        self.image_transform = image_transform
        self.mask_transform = mask_transform
        # self.class_values = [self.class_names.index(cls.lower()) for cls in self.class_names]
        self.class_values = {}
        for key, value in self.class_names.items():
            self.class_values[key] = [value.index(cls.lower()) for cls in value]

        self.split = split

        self.seg_type = seg_type
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.image_paths)

    def get_label_mask(self, mask, class_values): 
        """
        This function encodes the pixels belonging to the same class
        in the image into the same label
        """
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for value in class_values:
            for ii, label in enumerate(self.label_colors_list):
                if value == self.label_colors_list.index(label):
                    label = np.array(label)
                    label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)

        return label_mask

    def label_mask_to_color_mask(self, label_mask):
        red_map = np.zeros_like(label_mask).astype(np.uint8)
        green_map = np.zeros_like(label_mask).astype(np.uint8)
        blue_map = np.zeros_like(label_mask).astype(np.uint8)
        
        for label_num in range(0, len(self.label_colors_list)):
            if label_num in self.class_values:
                idx = label_mask == label_num
                red_map[idx] = np.array(self.label_colors_list)[label_num, 0]
                green_map[idx] = np.array(self.label_colors_list)[label_num, 1]
                blue_map[idx] = np.array(self.label_colors_list)[label_num, 2]
            
        segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
        return segmented_image
    
    def semantic_to_instances(self, semantic_mask: np.ndarray) -> Tuple[List[int], List[np.ndarray]]:
        labels = []
        masks = []
        
        # print(semantic_mask)
        # print(semantic_mask.shape)
        unique_classes = np.unique(semantic_mask)
        # print(unique_classes)
        for class_idx in unique_classes:
            if class_idx == self.ignore_index:
                continue
            # Create binary mask for this class
            class_mask = (semantic_mask == class_idx).astype(np.uint8)
            labels.append(int(class_idx))
            masks.append(class_mask.astype(np.float32))
        
        return labels, masks
    
    def instances_to_semantic(self, targets: List[dict], ignore_index: int = 255):
        batch_size = len(targets)
        _, H, W = targets[0]['masks'].shape
        masks_list = torch.ones((batch_size, H, W))
        masks_list *= ignore_index

        for i, target in enumerate(targets):
            labels = target['labels']
            masks = target['masks']
            for label, mask in zip(labels, masks):
                masks_list[i][mask.bool()] = label

        return masks_list

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])

        
        # mask = Image.open(self.label_paths[index])
        
        masks = {}
        for key, value in self.label_paths.items():
            masks[key] = Image.open(value[index])
            print(value[index])
        orig_size = image.size

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            for key, value in masks.items():
                masks[key] = self.mask_transform(masks[key])
                masks[key] = np.array(masks[key]).astype(int)

        # mask = np.array(mask).astype(int)

        if self.seg_type == "semantic_segmentation":
            # mask = torch.tensor(mask, dtype=torch.long) 
            for key, value in masks.items():
                masks[key] = torch.tensor(masks[key], dtype=torch.long)

            return image, masks

        elif self.seg_type == "instance_segmentation":
            targets = {}
            for key, value in masks.items():
                labels, instance_masks = self.semantic_to_instances(value)
                # Handle case with no instances
                if len(labels) == 0:
                    # print(f"In task {key}: {self.image_paths[index]} has no labels")
                    labels = torch.zeros(0, dtype=torch.long)
                    binary_masks = torch.zeros(0, semantic_array.shape[0], semantic_array.shape[1], dtype=torch.float32)
                else:
                    labels = torch.tensor(labels, dtype=torch.long)
                    binary_masks = torch.stack([torch.from_numpy(mask) for mask in instance_masks])
                    
                # Create target dictionary
                target = {
                    "labels": labels,
                    "masks": binary_masks,
                    "image_id": index,
                    "orig_size": orig_size,  # (H, W)
                }

                targets[key] = target
            
            # print(target)
            return image, targets
        else:
            raise ValueError(f"Unsupported task_type: {self.seg_type}")
    @property
    def collate_fn(self):
        def func(batch):
            if self.seg_type == "semantic_segmentation":
                imgs = [e[0] for e in batch]
                targets = [e[1] for e in batch]
                imgs = default_collate(imgs)
                # targets = default_collate(targets)
                return imgs, targets
            elif self.seg_type == "instance_segmentation":
                imgs = [e[0] for e in batch]
                # targets = [e[1] for e in batch]
                targets = {}
                for t in self.tasks:
                    targets[t] = []
                for e in batch:
                    for t in self.tasks:
                        targets[t].append(e[1][t])
                imgs = default_collate(imgs)
                return imgs, targets
        return func

class SemsegBDD100k(Dataset):
    def __init__(self, image_base: str, 
                        label_base: str, 
                        image_transform, 
                        mask_transform, 
                        label_colors_list, 
                        class_names, 
                        split, 
                        seg_type,
                        ignore_index,
                        ):
        self.image_paths = glob.glob(f"{image_base}/*")
        self.label_paths = glob.glob(f"{label_base}/*")

        self.image_paths.sort()
        self.label_paths.sort()

        for img_path, label_path in zip(self.image_paths, self.label_paths):
            assert os.path.basename(img_path).split('.')[0] == os.path.basename(label_path).split('.')[0]
        self.label_colors_list = label_colors_list
        self.class_names = class_names

        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.class_values = [self.class_names.index(cls.lower()) for cls in self.class_names]
        self.split = split

        self.seg_type = seg_type
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.image_paths)

    def get_label_mask(self, mask, class_values): 
        """
        This function encodes the pixels belonging to the same class
        in the image into the same label
        """
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for value in class_values:
            for ii, label in enumerate(self.label_colors_list):
                if value == self.label_colors_list.index(label):
                    label = np.array(label)
                    label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)

        return label_mask

    def label_mask_to_color_mask(self, label_mask):
        red_map = np.zeros_like(label_mask).astype(np.uint8)
        green_map = np.zeros_like(label_mask).astype(np.uint8)
        blue_map = np.zeros_like(label_mask).astype(np.uint8)
        
        for label_num in range(0, len(self.label_colors_list)):
            if label_num in self.class_values:
                idx = label_mask == label_num
                red_map[idx] = np.array(self.label_colors_list)[label_num, 0]
                green_map[idx] = np.array(self.label_colors_list)[label_num, 1]
                blue_map[idx] = np.array(self.label_colors_list)[label_num, 2]
            
        segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
        return segmented_image
    
    def semantic_to_instances(self, semantic_mask: np.ndarray) -> Tuple[List[int], List[np.ndarray]]:
        labels = []
        masks = []
        
        # print(semantic_mask)
        # print(semantic_mask.shape)
        unique_classes = np.unique(semantic_mask)
        
        for class_idx in unique_classes:
            if class_idx == self.ignore_index:
                continue
            # Create binary mask for this class
            class_mask = (semantic_mask == class_idx).astype(np.uint8)
            labels.append(int(class_idx))
            masks.append(class_mask.astype(np.float32))
        
        return labels, masks
    
    def instances_to_semantic(self, targets: List[dict], ignore_index: int = 255):
        batch_size = len(targets)
        _, H, W = targets[0]['masks'].shape
        masks_list = torch.ones((batch_size, H, W))
        masks_list *= ignore_index

        for i, target in enumerate(targets):
            labels = target['labels']
            masks = target['masks']
            for label, mask in zip(labels, masks):
                masks_list[i][mask.bool()] = label

        return masks_list

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.label_paths[index])
        orig_size = mask.size

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = np.array(mask).astype(int)

        if self.seg_type == "semantic_segmentation":
            mask = torch.tensor(mask, dtype=torch.long) 
            return image, mask

        elif self.seg_type == "instance_segmentation":
            labels, instance_masks = self.semantic_to_instances(mask)
            # Handle case with no instances
            if len(labels) == 0:
                labels = torch.zeros(0, dtype=torch.long)
                masks = torch.zeros(0, semantic_array.shape[0], semantic_array.shape[1], dtype=torch.float32)
            else:
                labels = torch.tensor(labels, dtype=torch.long)
                masks = torch.stack([torch.from_numpy(mask) for mask in instance_masks])
                
            # Create target dictionary
            target = {
                "labels": labels,
                "masks": masks,
                "image_id": index,
                "orig_size": orig_size,  # (H, W)
            }
            
            # print(target)
            return image, target
        else:
            raise ValueError(f"Unsupported task_type: {self.seg_type}")
    @property
    def collate_fn(self):
        def func(batch):
            if self.seg_type == "semantic_segmentation":
                imgs = [e[0] for e in batch]
                targets = [e[1] for e in batch]
                imgs = default_collate(imgs)
                targets = default_collate(targets)
                return imgs, targets
            elif self.seg_type == "instance_segmentation":
                imgs = [e[0] for e in batch]
                targets = [e[1] for e in batch]
                imgs = default_collate(imgs)
                return imgs, targets
        return func