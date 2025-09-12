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

class BDD100kR2S100k(Dataset):
    def __init__(self, 
                        image_base_bdd: str, 
                        image_base_r2s: str, 
                        label_base_bdd: str, 
                        label_base_r2s: str, 
                        # image_transform, 
                        # mask_transform, 
                        transform,
                        label_colors_list_bdd, 
                        label_colors_list_r2s, 
                        class_names_bdd, 
                        class_names_r2s, 
                        split, 
                        seg_type,
                        ignore_index,
                        ):
        ## Loading path from BDD100k
        self.image_paths_bdd = glob.glob(f"{image_base_bdd}/*")
        self.label_paths_bdd = glob.glob(f"{label_base_bdd}/*")
        self.image_paths_bdd.sort()
        self.label_paths_bdd.sort()
        self.label_colors_list_bdd = label_colors_list_bdd
        self.class_names_bdd = class_names_bdd
        self.class_values_bdd = [self.class_names_bdd.index(cls.lower()) for cls in self.class_names_bdd]

        ## Loading path from R2S100k
        self.image_paths_r2s = glob.glob(f"{image_base_r2s}/*")
        self.label_paths_r2s = glob.glob(f"{label_base_r2s}/*")
        self.image_paths_r2s.sort()
        self.label_paths_r2s.sort()
        self.label_colors_list_r2s = label_colors_list_r2s
        self.class_names_r2s = class_names_r2s
        self.class_values_r2s = [self.class_names_r2s.index(cls.lower()) for cls in self.class_names_r2s]

        # print(self.class_names_r2s)
        # print(self.class_values_r2s)
        for img_path, label_path in zip(self.image_paths_bdd, self.label_paths_bdd):
            assert os.path.basename(img_path).split('.')[0] == os.path.basename(label_path).split('.')[0]

        for img_path, label_path in zip(self.image_paths_r2s, self.label_paths_r2s):
            assert os.path.basename(img_path).split('.')[0] == os.path.basename(label_path).split('.')[0]

        # self.image_transform = image_transform
        # self.mask_transform = mask_transform
        self.transform = transform
        
        self.seg_type = seg_type
        self.ignore_index = ignore_index
        self.split = split

    def __len__(self):
        return len(self.image_paths_bdd) + len(self.image_paths_r2s)
    
    def get_label_mask(self, mask, class_values, label_colors_list): 
        """
        This function encodes the pixels belonging to the same class
        in the image into the same label
        """
        label_mask = np.ones((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        label_mask *= self.ignore_index
        for value in class_values:
            for ii, label in enumerate(label_colors_list):
                if value == label_colors_list.index(label):
                    label = np.array(label)
                    label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)

        return label_mask
    
    def label_mask_to_color_mask(self, label_mask, label_colors_list, class_values):
        red_map = np.zeros_like(label_mask).astype(np.uint8)
        green_map = np.zeros_like(label_mask).astype(np.uint8)
        blue_map = np.zeros_like(label_mask).astype(np.uint8)
        
        for label_num in range(0, len(label_colors_list)):
            if label_num in class_values:
                idx = label_mask == label_num
                red_map[idx] = np.array(label_colors_list)[label_num, 0]
                green_map[idx] = np.array(label_colors_list)[label_num, 1]
                blue_map[idx] = np.array(label_colors_list)[label_num, 2]
            
        segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
        return segmented_image

    def semantic_to_instances(self, semantic_mask: np.ndarray) -> Tuple[List[int], List[np.ndarray]]:
        labels = []
        masks = []
        
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

    # retry a few times to avoid all-ignored crops (useful if you ignore bg downstream)
    def apply_aug_with_retry(self, img, msk, retries=6, min_fg_px=128):
        out = None
        for _ in range(retries):
            out = self.transform(image=img, mask=msk) if self.transform is not None else {"image": img, "mask": msk}
            m = out["mask"]
            # keep if has something other than ignore_index
            if (m != self.ignore_index).sum() >= min_fg_px:
                break
        return out
    
    def __getitem__(self, index):
        ## Identity which dataset is used
        if index < len(self.image_paths_bdd):
            ## is BDD100k
            image = Image.open(self.image_paths_bdd[index])
            image = np.array(image)
            mask = Image.open(self.label_paths_bdd[index])
            orig_size = mask.size

            # if self.image_transform:
            #     image = self.image_transform(image)

            # if self.mask_transform:
            #     mask = self.mask_transform(mask)

            mask = np.array(mask).astype(int)

            out = self.apply_aug_with_retry(image, mask)
            image = out["image"]                # tensor CxHxW (from ToTensorV2)
            # mask = torch.as_tensor(out["mask"], dtype=torch.long)
            mask = np.array(out["mask"]).astype(int)
            
            if self.seg_type == "semantic_segmentation":
                mask = torch.tensor(mask, dtype=torch.long) 
                return image, mask

            elif self.seg_type == "instance_segmentation":
                labels, instance_masks = self.semantic_to_instances(mask)
                # Handle case with no instances
                if len(labels) == 0:
                    labels = torch.zeros(0, dtype=torch.long)
                    masks = torch.zeros(0, mask.shape[0], mask.shape[1], dtype=torch.float32)
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
                return image, target, 0 ## return 0 for bdd100k
            else:
                raise ValueError(f"Unsupported task_type: {self.seg_type}")
        else:
            image = Image.open(self.image_paths_r2s[index - len(self.image_paths_bdd)])
            image = np.array(image)
            mask = Image.open(self.label_paths_r2s[index - len(self.image_paths_bdd)])
            orig_size = mask.size

            # if self.image_transform:
            #     image = self.image_transform(image)

            # if self.mask_transform:
            #     mask = self.mask_transform(mask)
            
            mask = np.array(mask)
            mask = self.get_label_mask(mask, class_values=self.class_values_r2s, label_colors_list=self.label_colors_list_r2s)

            out = self.apply_aug_with_retry(image, mask)
            image = out["image"]                # tensor CxHxW (from ToTensorV2)
            # mask = torch.as_tensor(out["mask"], dtype=torch.long)
            mask = np.array(out["mask"]).astype(int)
            if self.seg_type == "semantic_segmentation":
                mask = torch.tensor(mask, dtype=torch.long) 
                return image, mask

            elif self.seg_type == "instance_segmentation":
                labels, instance_masks = self.semantic_to_instances(mask)
                # Handle case with no instances
                if len(labels) == 0:
                    labels = torch.zeros(0, dtype=torch.long)
                    masks = torch.zeros(0, mask.shape[0], mask.shape[1], dtype=torch.float32)
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
                
                return image, target, 1
            else:
                raise ValueError(f"Unsupported task_type: {self.seg_type}")

    @property
    def collate_fn(self):
        def func(batch):
            if self.seg_type == "semantic_segmentation":
                imgs = [e[0] for e in batch]
                targets = [e[1] for e in batch]
                dataset_ids = [e[2] for e in batch]
                imgs = default_collate(imgs)
                targets = default_collate(targets)
                dataset_ids = default_collate(dataset_ids)

                return imgs, targets, dataset_ids
            elif self.seg_type == "instance_segmentation":
                imgs = [e[0] for e in batch]
                targets = [e[1] for e in batch]
                dataset_ids = [e[2] for e in batch]
                imgs = default_collate(imgs)
                dataset_ids = default_collate(dataset_ids)

                return imgs, targets, dataset_ids
        return func