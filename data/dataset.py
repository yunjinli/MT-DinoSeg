from typing import Dict, Type, Any, Optional
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from pathlib import Path

from config.base import DataConfig
from .r2s100k.dataset import R2S100k
from .bdd100k.dataset import SemsegBDD100k, MultitaskBDD100k
from .combined.dataset import BDD100kR2S100k, BDD100kR2S100kTwoViewAug

# Registry of dataset implementations
DATASET_REGISTRY = {
    'r2s100k': R2S100k,
    'semseg_bdd100k': SemsegBDD100k,
    'multitask_bdd100k': MultitaskBDD100k,
    'semseg_bdd100k_r2s100k': BDD100kR2S100k,
    'semseg_bdd100k_r2s100k_two_view_aug': BDD100kR2S100kTwoViewAug,
    # Add other datasets here
}


def get_params_semseg_bdd100k_r2s100k_two_view_aug(config, split):
    transform = config.get_two_view_transforms(split=split)
    # Get dataset path
    base_path_bdd = Path(config.dataset_path_bdd)
    base_path_r2s = Path(config.dataset_path_r2s)

    if split == "train":
        image_base_bdd = base_path_bdd / "images/10k/train"
        label_base_bdd = base_path_bdd / "labels/sem_seg/masks/train"
        image_base_r2s = base_path_r2s / "train"
        label_base_r2s = base_path_r2s / "Train-Labels"
    elif split == "val":
        image_base_bdd = base_path_bdd / "images/10k/val"
        label_base_bdd = base_path_bdd / "labels/sem_seg/masks/val"
        image_base_r2s = base_path_r2s / "val"
        label_base_r2s = base_path_r2s / "val_labels"
    elif split == "test":
        print(f"Warning: BDD100k doesn't have an official test dataset")
        print(f"Warning: Loading BDD100K validation set instead...")
        image_base_bdd = base_path_bdd / "images/10k/val"
        label_base_bdd = base_path_bdd / "labels/sem_seg/masks/val"
        image_base_r2s = base_path_r2s / "test"
        label_base_r2s = base_path_r2s / "test_labels"
    else:
        return None
    
    # Check if the split exists
    if not image_base_bdd.exists() or not label_base_bdd.exists():
        print(f"Warning: BDD100k {split} split not found at {image_base_bdd} and {label_base_bdd}")
        return None
    if not image_base_r2s.exists() or not label_base_r2s.exists():
        print(f"Warning: R2S100K {split} split not found at {image_base_r2s} and {label_base_r2s}")
        return None

    label_colors_list_bdd, class_names_bdd = config.parse_color_and_names(task='bdd100k')
    label_colors_list_r2s, class_names_r2s = config.parse_color_and_names(task='r2s100k')
    
    return {
        'image_base_bdd': str(image_base_bdd),
        'image_base_r2s': str(image_base_r2s),
        'label_base_bdd': str(label_base_bdd),
        'label_base_r2s': str(label_base_r2s),
        # 'image_transform': image_transform,
        # 'mask_transform': mask_transform,
        'transform': transform,
        'class_names_bdd': class_names_bdd,
        'class_names_r2s': class_names_r2s,
        'label_colors_list_bdd': label_colors_list_bdd,
        'label_colors_list_r2s': label_colors_list_r2s,
        'split': split,
        'seg_type': config.task_type,
        'ignore_index': config.ignore_index,
    }
    
def get_params_semseg_bdd100k_r2s100k(config, split):
    # Create transforms
    # image_transform, mask_transform = config.get_transforms()
    transform = config.get_transforms(split=split)
    # Get dataset path
    base_path_bdd = Path(config.dataset_path_bdd)
    base_path_r2s = Path(config.dataset_path_r2s)

    if split == "train":
        image_base_bdd = base_path_bdd / "images/10k/train"
        label_base_bdd = base_path_bdd / "labels/sem_seg/masks/train"
        image_base_r2s = base_path_r2s / "train"
        label_base_r2s = base_path_r2s / "Train-Labels"
    elif split == "val":
        image_base_bdd = base_path_bdd / "images/10k/val"
        label_base_bdd = base_path_bdd / "labels/sem_seg/masks/val"
        image_base_r2s = base_path_r2s / "val"
        label_base_r2s = base_path_r2s / "val_labels"
    elif split == "test":
        print(f"Warning: BDD100k doesn't have an official test dataset")
        print(f"Warning: Loading BDD100K validation set instead...")
        image_base_bdd = base_path_bdd / "images/10k/val"
        label_base_bdd = base_path_bdd / "labels/sem_seg/masks/val"
        image_base_r2s = base_path_r2s / "test"
        label_base_r2s = base_path_r2s / "test_labels"
    else:
        return None
    
    # Check if the split exists
    if not image_base_bdd.exists() or not label_base_bdd.exists():
        print(f"Warning: BDD100k {split} split not found at {image_base_bdd} and {label_base_bdd}")
        return None
    if not image_base_r2s.exists() or not label_base_r2s.exists():
        print(f"Warning: R2S100K {split} split not found at {image_base_r2s} and {label_base_r2s}")
        return None

    label_colors_list_bdd, class_names_bdd = config.parse_color_and_names(task='bdd100k')
    label_colors_list_r2s, class_names_r2s = config.parse_color_and_names(task='r2s100k')
    
    return {
        'image_base_bdd': str(image_base_bdd),
        'image_base_r2s': str(image_base_r2s),
        'label_base_bdd': str(label_base_bdd),
        'label_base_r2s': str(label_base_r2s),
        # 'image_transform': image_transform,
        # 'mask_transform': mask_transform,
        'transform': transform,
        'class_names_bdd': class_names_bdd,
        'class_names_r2s': class_names_r2s,
        'label_colors_list_bdd': label_colors_list_bdd,
        'label_colors_list_r2s': label_colors_list_r2s,
        'split': split,
        'seg_type': config.task_type,
        'ignore_index': config.ignore_index,
    }

def get_params_r2s100k(config, split):
    # Create transforms
    image_transform, mask_transform = config.get_transforms()
    # Get dataset path
    base_path = Path(config.dataset_path)

    # R2S100K specific handling
    if split == "train":
        image_base = base_path / "train"
        seg_base = base_path / "Train-Labels"
    elif split == "val":
        image_base = base_path / "val"
        seg_base = base_path / "val_labels"
    elif split == "test":
        image_base = base_path / "test"
        seg_base = base_path / "test_labels"
    else:
        return None
    
    # Check if the split exists
    if not image_base.exists() or not seg_base.exists():
        print(f"Warning: R2S100K {split} split not found at {image_base} and {seg_base}")
        return None
    
    return {
        'image_base': str(image_base),
        'seg_base': str(seg_base),
        'image_transform': image_transform,
        'mask_transform': mask_transform,
        'class_names': config.class_names,
        'label_colors_list': config.label_colors_list,
        'split': split,
        'task_type': config.task_type,
    }

def get_params_semseg_bdd100k(config, split):
    # Create transforms
    image_transform, mask_transform = config.get_transforms()
    # Get dataset path
    base_path = Path(config.dataset_path)

    if split == "train":
        image_base = base_path / "images/10k/train"
        label_base = base_path / "labels/sem_seg/masks/train"
    elif split == "val":
        image_base = base_path / "images/10k/val"
        label_base = base_path / "labels/sem_seg/masks/val"
    elif split == "test":
        print(f"Warning: BDD100k doesn't have an official test dataset")
        print(f"Warning: Loading BDD100K validation set instead...")
        image_base = base_path / "images/10k/val"
        label_base = base_path / "labels/sem_seg/masks/val"
    else:
        return None
    
    # Check if the split exists
    if not image_base.exists() or not label_base.exists():
        print(f"Warning: BDD100k {split} split not found at {image_base} and {seg_base}")
        return None
    
    return {
        'image_base': str(image_base),
        'label_base': str(label_base),
        'image_transform': image_transform,
        'mask_transform': mask_transform,
        'class_names': config.class_names,
        'label_colors_list': config.label_colors_list,
        'split': split,
        'seg_type': config.task_type,
        'ignore_index': config.ignore_index,
    }

def get_params_multitask_bdd100k(config, split):
    # Create transforms
    image_transform, mask_transform = config.get_transforms()
    # Get dataset path
    base_path = Path(config.dataset_path)

    tasks = config.tasks
    # print(tasks)
    label_base_dict = {}
    label_colors_list_dict = {}
    class_names_dict = {}
    for task in tasks:
        if split == "train":
            image_base = base_path / "images/10k/train"
            label_base = base_path / f"labels/{task}/masks/train"
        elif split == "val":
            image_base = base_path / "images/10k/val"
            label_base = base_path / f"labels/{task}/masks/val"
        elif split == "test":
            print(f"Warning: BDD100k doesn't have an official test dataset")
            print(f"Warning: Loading BDD100K validation set instead...")
            image_base = base_path / "images/10k/val"
            label_base = base_path / f"labels/{task}/masks/val"
        else:
            return None

        label_base_dict[task] = str(label_base)
        label_colors_list, class_names = config.parse_color_and_names(task)
        label_colors_list_dict[task] = label_colors_list
        class_names_dict[task] = class_names
        # Check if the split exists
        if not image_base.exists() or not label_base.exists():
            print(f"Warning: BDD100k_{task} {split} split not found at {image_base} and {seg_base}")
            return None
    # print(label_colors_list_dict)
    # print(class_names_dict)
    return {
        'image_base': str(image_base),
        'label_base': label_base_dict,
        'image_transform': image_transform,
        'mask_transform': mask_transform,
        'class_names': class_names_dict,
        'label_colors_list': label_colors_list_dict,
        'split': split,
        'seg_type': config.task_type,
        'ignore_index': config.ignore_index,
    }

get_params = {
    'r2s100k': get_params_r2s100k,
    'semseg_bdd100k': get_params_semseg_bdd100k,
    'multitask_bdd100k': get_params_multitask_bdd100k,
    'semseg_bdd100k_r2s100k': get_params_semseg_bdd100k_r2s100k,
    'semseg_bdd100k_r2s100k_two_view_aug': get_params_semseg_bdd100k_r2s100k_two_view_aug,
}

def get_dataset(config: DataConfig, split: str = "train") -> Optional[Dataset]:
    """
    Create a dataset based on configuration.
    
    Args:
        config: Dataset configuration
        split: Dataset split ("train", "val", or "test")
        
    Returns:
        Initialized dataset or None if the split doesn't exist
    """
    if config.dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")
    
    # print(f"Loading {config.task_type} dataset with {config.num_classes} classes")
    # Get dataset class
    dataset_class = DATASET_REGISTRY[config.dataset_name]
    # Get dataset-specific params
    dataset_params = get_params[config.dataset_name](config=config, split=split)
    
    if dataset_params is not None:
        return dataset_class(**dataset_params)
    else:
        return None