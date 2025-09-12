"""Dataset configurations."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from .base import DataConfig
from .registry import ConfigRegistry
import torchvision.transforms as T
from collections import namedtuple
from torchvision.transforms import InterpolationMode
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

DataRegistry = ConfigRegistry[DataConfig]("DataRegistry")

@DataRegistry.register("r2s100k")
@dataclass
class R2S100KConfig(DataConfig):
    input_size: Tuple[int, int] = None
    dataset_name: str = "r2s100k"
    dataset_path: str = "/home/phd_li/dataset/r2s100k"
    task_type: str = "semantic_segmentation"  # Can be: semantic_segmentation, instance_segmentation, panoptic_segmentation

    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Data loading parameters
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 4
    drop_last: bool = True

    ## For r2s100k
    # Raw dataset information - this represents the complete ground truth
    _raw_num_classes: int = 15
    _raw_class_names: List[str] = field(default_factory=lambda: [
        'bg', 'wet_road_region', 'road_region', 'mud', 'earthen_patch', 
        'mountain-stones', 'dirt', 'vegitation_misc', 'distressed_patch', 
        'drainage_grate', 'water_puddle', 'speed_breaker', 'misc', 
        'gravel_patch', 'concrete_material'
    ])
    _raw_label_colors_list: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (0, 0, 0),          # BG
        (2, 79, 59),        # Wet_Road_Region
        (17, 163, 74),      # Road_region
        (112, 84, 62),      # Mud
        (225, 148, 79),     # Earthen_Patch
        (120, 114, 104),    # Mountain-stones
        (166, 130, 95),     # Dirt
        (128, 222, 91),     # Vegitation_Misc
        (119, 61, 128),     # Distressed_Patch
        (93, 86, 176),      # Drainage_Grate
        (140, 160, 222),    # Water_puddle
        (234, 133, 5),      # Speed_Breaker
        (156, 28, 39),      # Misc 
        (99, 122, 130),     # Gravel_Patch 
        (123, 43, 31),      # Concrete_Material
    ])

    # num_classes = 15
    # ignore_index = 0
    # label_colors_list = [
    #         (0, 0, 0), # BG
    #         (2, 79, 59), # Wet_Road_Region
    #         (17, 163, 74), # Road_region
    #         (112, 84, 62), # Mud
    #         (225, 148, 79), # Earthen_Patch
    #         (120, 114, 104), # Mountain-stones
    #         (166, 130, 95), # Dirt
    #         (128, 222, 91), # Vegitation_Misc
    #         (119, 61, 128), # Distressed_Patch
    #         (93, 86, 176), # Drainage_Grate
    #         (140, 160, 222), # Water_puddle
    #         (234, 133, 5), #Speed_Breaker
    #         (156, 28, 39), # Misc 
    #         (99, 122, 130), # Gravel_Patch 
    #         (123, 43, 31), # Concrete_Material
    #     ]
    # all the classes that are present in the dataset
    # class_names = ['bg', 'wet_road_region', 'road_region', 'mud', 'earthen_patch', 'mountain-stones', 'dirt', 'vegitation_misc', 'distressed_patch', 'drainage_grate', 'water_puddle', 'speed_breaker', 'misc', 'gravel_patch', 'concrete_material']
    
    @property
    def num_classes(self) -> int:
        """Return the number of classes appropriate for the current task type."""
        if self.task_type == "semantic_segmentation":
            return self._raw_num_classes  # All 15 classes
        
        elif self.task_type == "instance_segmentation":
            # Only classes that can form instances (exclude background)
            # return len([cls_name for cls_name in self._raw_class_names if cls_name != 'bg'])  # 14 classes
            return self._raw_num_classes  # All 15 classes

        elif self.task_type == "panoptic_segmentation":
            # For panoptic, we typically count only the "thing" classes for instance detection
            # "Stuff" classes are handled separately through semantic segmentation
            # return len(self._thing_classes)  # Number of thing classes
            raise NotImplementedError(f"No panoptic segmentation task in {self.dataset_name}")
        
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    @property
    def class_names(self) -> List[str]:
        """Return the class names appropriate for the current task type."""
        if self.task_type == "semantic_segmentation":
            return self._raw_class_names
        
        elif self.task_type == "instance_segmentation":
            # Remove background, return detectable classes
            # return [cls for cls in self._raw_class_names if cls != 'bg']
            return self._raw_class_names

        elif self.task_type == "panoptic_segmentation":
            # Return thing classes for instance detection
            raise NotImplementedError(f"No panoptic segmentation task in {self.dataset_name}")
        
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    @property
    def ignore_index(self) -> Optional[int]:
        """Return the ignore index appropriate for the current task type."""
        if self.task_type == "semantic_segmentation":
            return 0  # Ignore background in loss computation
        
        elif self.task_type == "instance_segmentation":
            # return 255  # No ignore index needed - background simply doesn't generate instances
            return 0  
        
        elif self.task_type == "panoptic_segmentation":
           raise NotImplementedError(f"No panoptic segmentation task in {self.dataset_name}")
        
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    @property
    def label_colors_list(self) -> List[Tuple[int, int, int]]:
        """Return the label colors appropriate for the current task type."""
        if self.task_type == "semantic_segmentation":
            return self._raw_label_colors_list
        
        elif self.task_type == "instance_segmentation":
            # Return colors for detectable classes (excluding background)
            # return [color for i, color in enumerate(self._raw_label_colors_list) if i != 0]
            return self._raw_label_colors_list
        
        elif self.task_type == "panoptic_segmentation":
            # Return colors for thing classes
            raise NotImplementedError(f"No panoptic segmentation task in {self.dataset_name}")
        
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    def get_transforms(self):
        """Get transforms based on configuration."""
        try:
            print("Input size: ", self.input_size)
            print(f"Mean: {self.mean}, Std: {self.std}")
            image_transform = T.Compose([ 
                T.Resize(self.input_size),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])
            
            mask_transform = T.Compose([
                T.Resize(self.input_size)
            ])
        except:
            print("Input size not specified, use (224, 224) ...")
            image_transform = T.Compose([ 
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])
            
            mask_transform = T.Compose([
                T.Resize((224, 224))
            ])
        
        return image_transform, mask_transform

# Copied from https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/label.py
# a label and all meta information
# Code inspired by Cityscapes https://github.com/mcordts/cityscapesScripts
Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class
        "id",  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images An ID
        # of -1 means that this label does not have an ID and thus is ignored
        # when creating ground truth images (e.g. license plate). Do not modify
        # these IDs, since exactly these IDs are expected by the evaluation
        # server.
        "trainId",
        # Feel free to modify these IDs as suitable for your method. Then
        # create ground truth images with train IDs, using the tools provided
        # in the 'preparation' folder. However, make sure to validate or submit
        # results to our evaluation server using the regular IDs above! For
        # trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the
        # inverse mapping, we use the label that is defined first in the list
        # below. For example, mapping all void-type classes to the same ID in
        # training, might make sense for some approaches. Max value is 255!
        "category",  # The name of the category that this label belongs to
        "categoryId",
        # The ID of this category. Used to create ground truth images
        # on category level.
        "hasInstances",
        # Whether this label distinguishes between single instances or not
        "ignoreInEval",
        # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not
        "color",  # The color of this label
    ],
)

# Our extended list of label types. Our train id is compatible with Cityscapes
bdd100k_labels = [
    #       name                     id    trainId   category catId
    #       hasInstances   ignoreInEval   color
    Label("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    Label("dynamic", 1, 255, "void", 0, False, True, (111, 74, 0)),
    Label("ego vehicle", 2, 255, "void", 0, False, True, (0, 0, 0)),
    Label("ground", 3, 255, "void", 0, False, True, (81, 0, 81)),
    Label("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    Label("parking", 5, 255, "flat", 1, False, True, (250, 170, 160)),
    Label("rail track", 6, 255, "flat", 1, False, True, (230, 150, 140)),
    Label("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
    Label("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    Label("bridge", 9, 255, "construction", 2, False, True, (150, 100, 100)),
    Label("building", 10, 2, "construction", 2, False, False, (70, 70, 70)),
    Label("fence", 11, 4, "construction", 2, False, False, (190, 153, 153)),
    Label("garage", 12, 255, "construction", 2, False, True, (180, 100, 180)),
    Label(
        "guard rail", 13, 255, "construction", 2, False, True, (180, 165, 180)
    ),
    Label("tunnel", 14, 255, "construction", 2, False, True, (150, 120, 90)),
    Label("wall", 15, 3, "construction", 2, False, False, (102, 102, 156)),
    Label("banner", 16, 255, "object", 3, False, True, (250, 170, 100)),
    Label("billboard", 17, 255, "object", 3, False, True, (220, 220, 250)),
    Label("lane divider", 18, 255, "object", 3, False, True, (255, 165, 0)),
    Label("parking sign", 19, 255, "object", 3, False, False, (220, 20, 60)),
    Label("pole", 20, 5, "object", 3, False, False, (153, 153, 153)),
    Label("polegroup", 21, 255, "object", 3, False, True, (153, 153, 153)),
    Label("street light", 22, 255, "object", 3, False, True, (220, 220, 100)),
    Label("traffic cone", 23, 255, "object", 3, False, True, (255, 70, 0)),
    Label(
        "traffic device", 24, 255, "object", 3, False, True, (220, 220, 220)
    ),
    Label("traffic light", 25, 6, "object", 3, False, False, (250, 170, 30)),
    Label("traffic sign", 26, 7, "object", 3, False, False, (220, 220, 0)),
    Label(
        "traffic sign frame",
        27,
        255,
        "object",
        3,
        False,
        True,
        (250, 170, 250),
    ),
    Label("terrain", 28, 9, "nature", 4, False, False, (152, 251, 152)),
    Label("vegetation", 29, 8, "nature", 4, False, False, (107, 142, 35)),
    Label("sky", 30, 10, "sky", 5, False, False, (70, 130, 180)),
    Label("person", 31, 11, "human", 6, True, False, (220, 20, 60)),
    Label("rider", 32, 12, "human", 6, True, False, (255, 0, 0)),
    Label("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
    Label("bus", 34, 15, "vehicle", 7, True, False, (0, 60, 100)),
    Label("car", 35, 13, "vehicle", 7, True, False, (0, 0, 142)),
    Label("caravan", 36, 255, "vehicle", 7, True, True, (0, 0, 90)),
    Label("motorcycle", 37, 17, "vehicle", 7, True, False, (0, 0, 230)),
    Label("trailer", 38, 255, "vehicle", 7, True, True, (0, 0, 110)),
    Label("train", 39, 16, "vehicle", 7, True, False, (0, 80, 100)),
    Label("truck", 40, 14, "vehicle", 7, True, False, (0, 0, 70)),
]

bdd100k_drivables = [
    #       name                     id    trainId   category catId
    #       hasInstances   ignoreInEval   color
    Label("direct", 0, 0, "drivable", 0, False, False, (219, 94, 86)),
    Label("alternative", 1, 1, "drivable", 0, False, False, (86, 211, 219)),
    Label("background", 2, 2, "drivable", 0, False, False, (0, 0, 0)),
]

bdd100k_lane_directions = [
    #       name                     id    trainId   category catId
    #       hasInstances   ignoreInEval   color
    Label("parallel", 0, 0, "lane_mark", 0, False, False, (0, 0, 0)),
    Label("vertical", 1, 1, "lane_mark", 0, False, False, (0, 0, 0)),
]

bdd100k_lane_styles = [
    #       name                     id    trainId   category catId
    #       hasInstances   ignoreInEval   color
    Label("solid", 0, 0, "lane_mark", 0, False, False, (0, 0, 0)),
    Label("dashed", 1, 1, "lane_mark", 0, False, False, (0, 0, 0)),
]

bdd100k_lane_categories = [
    #       name                     id    trainId   category catId
    #       hasInstances   ignoreInEval   color
    Label("crosswalk", 0, 0, "lane_mark", 0, False, False, (219, 94, 86)),
    Label("double other", 1, 1, "lane_mark", 0, False, False, (219, 194, 86)),
    Label("double white", 2, 2, "lane_mark", 0, False, False, (145, 219, 86)),
    Label("double yellow", 3, 3, "lane_mark", 0, False, False, (86, 219, 127)),
    Label("road curb", 4, 4, "lane_mark", 0, False, False, (86, 211, 219)),
    Label("single other", 5, 5, "lane_mark", 0, False, False, (86, 111, 219)),
    Label("single white", 6, 6, "lane_mark", 0, False, False, (160, 86, 219)),
    Label("single yellow", 7, 7, "lane_mark", 0, False, False, (219, 86, 178)),
    Label("background", 8, 8, "lane_mark", 0, False, False, (0, 0, 0)), ## Added by myself

]

BDD100K_ANNOTATIONs = {
    'sem_seg': bdd100k_labels,
    'lane': bdd100k_lane_categories,
    'drivable': bdd100k_drivables,
}

@DataRegistry.register("drivableseg_bdd100k")
@dataclass
class DrivablesegBDD100kConfig(DataConfig):
    input_size: Tuple[int, int] = None
    dataset_name: str = "drivableseg_bdd100k"
    dataset_path: str = "/home/phd_li/dataset/bdd100k"
    task_type: str = "semantic_segmentation"  # Can be: semantic_segmentation, instance_segmentation, panoptic_segmentation    
    
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Data loading parameters
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 4
    drop_last: bool = True

    ## For bdd100k
    ignore_index = 255
    num_classes = None
    class_names = None
    label_colors_list = None
    
    def parse_color_and_names(self):
        trainid_colors = []
        labels = BDD100K_ANNOTATIONs['drivable']
        for idx, label in enumerate(labels):
            if label.trainId != 255:
                trainid_colors.append({'trainId': label.trainId, 'color': label.color, 'name': label.name})
        label_colors_list = [None] * len(trainid_colors)
        class_names = [None] * len(trainid_colors)

        for label in trainid_colors:
            label_colors_list[label['trainId']] = label['color']
            class_names[label['trainId']] = label['name']
        
        return label_colors_list, class_names
        
    def get_transforms(self):
        """Get transforms based on configuration."""
        try:
            print("Input size: ", self.input_size)
            print(f"Mean: {self.mean}, Std: {self.std}")
            image_transform = T.Compose([ 
                T.Resize(self.input_size),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])
            
            mask_transform = T.Compose([
                T.Resize(self.input_size, interpolation=InterpolationMode.NEAREST)
            ])
        except:
            print("Input size not specified, use (224, 224) ...")
            image_transform = T.Compose([ 
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])
            
            mask_transform = T.Compose([
                T.Resize((224, 224), interpolation=InterpolationMode.NEAREST)
            ])
        
        return image_transform, mask_transform

@DataRegistry.register("laneseg_bdd100k")
@dataclass
class LanesegBDD100kConfig(DataConfig):
    input_size: Tuple[int, int] = None
    dataset_name: str = "laneseg_bdd100k"
    dataset_path: str = "/home/phd_li/dataset/bdd100k"
    task_type: str = "semantic_segmentation"  # Can be: semantic_segmentation, instance_segmentation, panoptic_segmentation    
    
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Data loading parameters
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 4
    drop_last: bool = True

    ## For bdd100k
    ignore_index = 255
    num_classes = None
    class_names = None
    label_colors_list = None
    
    def parse_color_and_names(self):
        trainid_colors = []
        labels = BDD100K_ANNOTATIONs['lane']
        for idx, label in enumerate(labels):
            if label.trainId != 255:
                trainid_colors.append({'trainId': label.trainId, 'color': label.color, 'name': label.name})
        label_colors_list = [None] * len(trainid_colors)
        class_names = [None] * len(trainid_colors)

        for label in trainid_colors:
            label_colors_list[label['trainId']] = label['color']
            class_names[label['trainId']] = label['name']
        
        return label_colors_list, class_names
        
    def get_transforms(self):
        """Get transforms based on configuration."""
        try:
            print("Input size: ", self.input_size)
            print(f"Mean: {self.mean}, Std: {self.std}")
            image_transform = T.Compose([ 
                T.Resize(self.input_size),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])
            
            mask_transform = T.Compose([
                T.Resize(self.input_size, interpolation=InterpolationMode.NEAREST)
            ])
        except:
            print("Input size not specified, use (224, 224) ...")
            image_transform = T.Compose([ 
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])
            
            mask_transform = T.Compose([
                T.Resize((224, 224), interpolation=InterpolationMode.NEAREST)
            ])
        
        return image_transform, mask_transform

@DataRegistry.register("semseg_bdd100k")
@dataclass
class SemsegBDD100kConfig(DataConfig):
    input_size: Tuple[int, int] = None
    dataset_name: str = "semseg_bdd100k"
    dataset_path: str = "/home/phd_li/dataset/bdd100k"
    task_type: str = "semantic_segmentation"  # Can be: semantic_segmentation, instance_segmentation, panoptic_segmentation    
    
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Data loading parameters
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 4
    drop_last: bool = True

    ## For bdd100k
    ignore_index = 255
    num_classes = None
    class_names = None
    label_colors_list = None
    
    def parse_color_and_names(self):
        trainid_colors = []
        labels = BDD100K_ANNOTATIONs['sem_seg']
        for idx, label in enumerate(labels):
            if label.trainId != 255:
                trainid_colors.append({'trainId': label.trainId, 'color': label.color, 'name': label.name})
        label_colors_list = [None] * len(trainid_colors)
        class_names = [None] * len(trainid_colors)

        for label in trainid_colors:
            label_colors_list[label['trainId']] = label['color']
            class_names[label['trainId']] = label['name']
        
        return label_colors_list, class_names
        
    def get_transforms(self):
        """Get transforms based on configuration."""
        try:
            print("Input size: ", self.input_size)
            print(f"Mean: {self.mean}, Std: {self.std}")
            image_transform = T.Compose([ 
                T.Resize(self.input_size),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])
            
            mask_transform = T.Compose([
                T.Resize(self.input_size, interpolation=InterpolationMode.NEAREST)
            ])
        except:
            print("Input size not specified, use (224, 224) ...")
            image_transform = T.Compose([ 
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])
            
            mask_transform = T.Compose([
                T.Resize((224, 224), interpolation=InterpolationMode.NEAREST)
            ])
        
        return image_transform, mask_transform

@DataRegistry.register("multitask_bdd100k")
@dataclass
class BDD100kConfig(DataConfig):
    input_size: Tuple[int, int] = None
    dataset_name: str = "multitask_bdd100k"
    dataset_path: str = "/home/phd_li/dataset/bdd100k"
    task_type: str = "semantic_segmentation"  # Can be: semantic_segmentation, instance_segmentation, panoptic_segmentation
    
    tasks: List[str] = field(default_factory=lambda: ['sem_seg', 'lane', 'drivable'])
    
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Data loading parameters
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 4
    drop_last: bool = True

    ## For bdd100k
    ignore_index = 255

    # num_classes = None
    # class_names = None
    # label_colors_list = None
    

    def parse_color_and_names(self, task):
        trainid_colors = []
        labels = BDD100K_ANNOTATIONs[task]
        for idx, label in enumerate(labels):
            if label.trainId != 255:
                trainid_colors.append({'trainId': label.trainId, 'color': label.color, 'name': label.name})
        label_colors_list = [None] * len(trainid_colors)
        class_names = [None] * len(trainid_colors)

        for label in trainid_colors:
            label_colors_list[label['trainId']] = label['color']
            class_names[label['trainId']] = label['name']
        
        return label_colors_list, class_names
    
    def get_transforms(self):
        """Get transforms based on configuration."""
        try:
            print("Input size: ", self.input_size)
            print(f"Mean: {self.mean}, Std: {self.std}")
            image_transform = T.Compose([ 
                T.Resize(self.input_size),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])
            
            mask_transform = T.Compose([
                T.Resize(self.input_size, interpolation=InterpolationMode.NEAREST)
            ])
        except:
            print("Input size not specified, use (224, 224) ...")
            image_transform = T.Compose([ 
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])
            
            mask_transform = T.Compose([
                T.Resize((224, 224), interpolation=InterpolationMode.NEAREST)
            ])
        
        return image_transform, mask_transform

@DataRegistry.register("semseg_bdd100k_r2s100k")
@dataclass
class SemsegBDD100kR2S100kConfig(DataConfig):
    input_size: Tuple[int, int] = None
    dataset_name: str = "semseg_bdd100k_r2s100k"
    # dataset_path = None
    # dataset_path: str = "/home/phd_li/dataset/bdd100k"
    # dataset_path_bdd = "/home/phd_li/dataset/bdd100k"
    # dataset_path_r2s = "/home/phd_li/dataset/r2s100k"
    dataset_path: str = "/mnt/sda/bdd100k"
    dataset_path_bdd = "/mnt/sda/bdd100k"
    dataset_path_r2s = "/mnt/sda/r2s100k"
    
    task_type: str = "semantic_segmentation"  # Can be: semantic_segmentation, instance_segmentation, panoptic_segmentation    
    
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # R2S100k
    _raw_num_classes_r2s: int = 15
    _raw_class_names_r2s: List[str] = field(default_factory=lambda: [
        'bg', 'wet_road_region', 'road_region', 'mud', 'earthen_patch', 
        'mountain-stones', 'dirt', 'vegitation_misc', 'distressed_patch', 
        'drainage_grate', 'water_puddle', 'speed_breaker', 'misc', 
        'gravel_patch', 'concrete_material'
    ])
    _raw_label_colors_list_r2s: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (0, 0, 0),          # BG
        (2, 79, 59),        # Wet_Road_Region
        (17, 163, 74),      # Road_region
        (112, 84, 62),      # Mud
        (225, 148, 79),     # Earthen_Patch
        (120, 114, 104),    # Mountain-stones
        (166, 130, 95),     # Dirt
        (128, 222, 91),     # Vegitation_Misc
        (119, 61, 128),     # Distressed_Patch
        (93, 86, 176),      # Drainage_Grate
        (140, 160, 222),    # Water_puddle
        (234, 133, 5),      # Speed_Breaker
        (156, 28, 39),      # Misc 
        (99, 122, 130),     # Gravel_Patch 
        (123, 43, 31),      # Concrete_Material
    ])

    # Data loading parameters
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 4
    drop_last: bool = True

    ignore_index = 255
    # num_classes = None
    # class_names = None
    # label_colors_list = None
    tasks: List[str] = field(default_factory=lambda: ['bdd100k', 'r2s100k'])
    
    def get_manual_superset_color_and_names(self):
        classnames_superset = [
        # fine-grained road surface (from R2S)
        "road_region", "wet_road_region", "water_puddle", "mud", "earthen_patch",
        "dirt", "gravel_patch", "distressed_patch", "drainage_grate", "speed_breaker", "misc",
        # rest from BDD
        "sidewalk", "building", "wall", "fence", "pole", "traffic_light", "traffic_sign",
        "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus",
        "train", "motorcycle", "bicycle",
        ]
        color_map_superset = [
            # fine-grained road surface (from R2S)
            (17, 163, 74),      # road_region
            (2, 79, 59),        # wet_road_region
            (140, 160, 222),    # water_puddle
            (112, 84, 62),      # mud
            (225, 148, 79),     # earthen_patch
            (166, 130, 95),     # dirt
            (99, 122, 130),     # gravel_patch
            (119, 61, 128),     # distressed_patch
            (93, 86, 176),      # drainage_grate
            (234, 133, 5),      # speed_breaker
            (156, 28, 39),      # misc
            # rest from BDD
            (244, 35, 232),     # sidewalk
            (70, 70, 70),       # building
            (102, 102, 156),    # wall
            (190, 153, 153),    # fence
            (153, 153, 153),    # pole
            (250, 170, 30),     # traffic_light
            (220, 220, 0),      # traffic_sign
            (107, 142, 35),     # vegetation
            (152, 251, 152),    # terrain
            (70, 130, 180),     # sky
            (220, 20, 60),      # person
            (255, 0, 0),        # rider
            (0, 0, 142),        # car
            (0, 0, 70),         # truck
            (0, 60, 100),       # bus
            (0, 80, 100),       # train
            (0, 0, 230),        # motorcycle
            (119, 11, 32),      # bicycle
        ]
        # --- Define your dataset label orders once (adjust to your exact loaders) ---
        bdd_order = [
            "road","sidewalk","building","wall","fence","pole","traffic_light","traffic_sign",
            "vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"
        ]
        r2s_order = [
            # Use your true R2S class order here
            # "bg", "wet_road_region","road_region","mud","earthen_patch","mountain-stones",
            "wet_road_region","road_region","mud","earthen_patch","mountain-stones",
            "dirt","vegitation_misc","distressed_patch","drainage_grate","water_puddle",
            "speed_breaker","misc","gravel_patch","concrete_material"
            # (if R2S has 16, add the missing one here)
        ]

        # --- Canonicalize name quirks so mapping is robust ---
        def canon(name: str):
            return name.replace("-", "_").replace("vegitation", "vegetation")

        sup2idx = {canon(n): i for i, n in enumerate(classnames_superset)}

        # --- Dataset -> Superset allowed sets (for training) ---
        bdd2sup_names = {
            "road": [
                "road_region","wet_road_region","water_puddle","mud","earthen_patch",
                "dirt","gravel_patch","distressed_patch","drainage_grate","speed_breaker","misc",
            ],
            "sidewalk": ["sidewalk"],
            "building": ["building"],
            "wall": ["wall"],
            "fence": ["fence"],
            "pole": ["pole"],
            "traffic_light": ["traffic_light"],
            "traffic_sign": ["traffic_sign"],
            "vegetation": ["vegetation"],
            "terrain": ["terrain"],
            "sky": ["sky"],
            "person": ["person"],
            "rider": ["rider"],
            "car": ["car"],
            "truck": ["truck"],
            "bus": ["bus"],
            "train": ["train"],
            "motorcycle": ["motorcycle"],
            "bicycle": ["bicycle"],
        }
        r2s2sup_names = {
            # "bg": [
            #     'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign',
            #     'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
            #     ],  # ignore or low weight
            "wet_road_region": ["wet_road_region"],
            "road_region": ["road_region"],
            "mud": ["mud"],
            "earthen_patch": ["earthen_patch"],
            "mountain-stones": ["terrain"],            # or a dedicated class if you kept it
            "dirt": ["dirt"],
            "vegitation_misc": ["vegetation"],
            "distressed_patch": ["distressed_patch"],
            "drainage_grate": ["drainage_grate"],
            "water_puddle": ["water_puddle"],
            "speed_breaker": ["speed_breaker"],
            "misc": ["misc"],
            "gravel_patch": ["gravel_patch"],
            "concrete_material": ["sidewalk"],         # curb/sidewalk
            # add any missing R2S label if needed
        }

        # Convert to index lists for set-aware loss
        def to_idx_lists(order, mapping_dict):
            idx_lists = []
            for name in order:
                S_names = mapping_dict.get(name, [])
                idx_lists.append([sup2idx[canon(s)] for s in S_names if canon(s) in sup2idx])
            return idx_lists

        map_bdd2sup_idx = to_idx_lists(bdd_order, bdd2sup_names)
        map_r2s2sup_idx = to_idx_lists(r2s_order, r2s2sup_names)

        # --- Superset -> Dataset projection matrices for eval ---
        
        def build_proj(order, idx_lists, C_sup):
            M = np.zeros((len(order), C_sup), dtype=np.float32)
            for i, S in enumerate(idx_lists):
                for j in S:
                    M[i, j] = 1.0
            return M

        M_sup_to_bdd = build_proj(bdd_order, map_bdd2sup_idx, len(classnames_superset))
        M_sup_to_r2s = build_proj(r2s_order, map_r2s2sup_idx, len(classnames_superset))

        # Sanity checks (optional)
        # assert M_sup_to_bdd.shape[1] == len(classnames_superset)

        return {
            "sup_names": classnames_superset,
            "sup_colors": color_map_superset,
            "bdd_order": bdd_order,
            "r2s_order": r2s_order,
            "map_bdd2sup_idx": map_bdd2sup_idx,
            "map_r2s2sup_idx": map_r2s2sup_idx,
            "M_sup_to_bdd": M_sup_to_bdd,   # use: P_bdd = P_sup @ M_sup_to_bdd.T
            "M_sup_to_r2s": M_sup_to_r2s,   # use: P_r2s = P_sup @ M_sup_to_r2s.T
        }
        
    def parse_color_and_names(self, task):
        if task == 'r2s100k':
        #    return self._raw_label_colors_list_r2s, self._raw_class_names_r2s
            return [color for i, color in enumerate(self._raw_label_colors_list_r2s) if i != 0], [cls for cls in self._raw_class_names_r2s if cls != 'bg']
        elif task == 'bdd100k':
            trainid_colors = []
            labels = BDD100K_ANNOTATIONs['sem_seg']
            for idx, label in enumerate(labels):
                if label.trainId != 255:
                    trainid_colors.append({'trainId': label.trainId, 'color': label.color, 'name': label.name})
            label_colors_list = [None] * len(trainid_colors)
            class_names = [None] * len(trainid_colors)

            for label in trainid_colors:
                label_colors_list[label['trainId']] = label['color']
                class_names[label['trainId']] = label['name']
            
            return label_colors_list, class_names
        else:
            raise NotImplementedError
        
    # def get_transforms(self):
    #     """Get transforms based on configuration."""
    #     try:
    #         print("Input size: ", self.input_size)
    #         print(f"Mean: {self.mean}, Std: {self.std}")
    #         image_transform = T.Compose([ 
    #             T.Resize(self.input_size),
    #             T.ToTensor(),
    #             T.Normalize(mean=self.mean, std=self.std),
    #         ])
            
    #         mask_transform = T.Compose([
    #             T.Resize(self.input_size, interpolation=InterpolationMode.NEAREST)
    #         ])
    #     except:
    #         print("Input size not specified, use (224, 224) ...")
    #         image_transform = T.Compose([ 
    #             T.Resize((224, 224)),
    #             T.ToTensor(),
    #             T.Normalize(mean=self.mean, std=self.std),
    #         ])
            
    #         mask_transform = T.Compose([
    #             T.Resize((224, 224), interpolation=InterpolationMode.NEAREST)
    #         ])
        
    #     return image_transform, mask_transform
    def get_transforms(self, split="train"):
        """Get transforms based on configuration."""
        # photometric = A.ColorJitter(0.2, 0.2, 0.2, 0.02, p=0.5) if dataset=="bdd" \
        #           else A.ColorJitter(0.15, 0.15, 0.15, 0.02, p=0.3)
        
        # transform = {}
        
        crop_h, crop_w = self.input_size if self.input_size is not None else (224, 224)
        
        if split == "train":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.RandomResizedCrop(height=crop_h, width=crop_w, scale=(0.1, 1.0), ratio=(1.8, 2.2), p=1.0),
                    A.Compose([
                        A.LongestMaxSize(max_size=max(crop_h, crop_w), interpolation=1, p=1.0),  # keep aspect
                        A.PadIfNeeded(min_height=crop_h, min_width=crop_w, border_mode=0, value=0, mask_value=self.ignore_index),
                        A.RandomCrop(height=crop_h, width=crop_w, p=1.0),
                    ]),
                ], p=1.0),
                A.Affine(
                    scale=(1.0, 1.0),
                    translate_percent=(0.0, 0.0),
                    rotate=(-5, 5),
                    interpolation=cv2.INTER_LINEAR,          # image
                    mask_interpolation=cv2.INTER_NEAREST,    # MASK
                    mode=cv2.BORDER_CONSTANT,
                    cval=(0, 0, 0),                          # image fill (RGB)
                    cval_mask=self.ignore_index,                        # MASK fill
                    p=0.2,
                ),
                A.ColorJitter(0.15, 0.15, 0.15, 0.02, p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                # A.GaussNoise(var_limit=(5.0, 10.0), p=0.2),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ], additional_targets={'mask':'mask'})
        elif split in ["val", "test"]:
            return A.Compose([
                A.Resize(height=crop_h, width=crop_w, interpolation=1, p=1.0),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ], additional_targets={'mask':'mask'})
        else:
            raise NotImplementedError