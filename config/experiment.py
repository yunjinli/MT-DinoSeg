from dataclasses import dataclass, field
from typing import Optional, List, Dict

from .base import Config, TrainingConfig, OptimizerConfig, SchedulerConfig, ModelConfig, DataConfig
from .registry import ConfigRegistry
from .model import ModelRegistry
from .data import DataRegistry

# Create registry for experiment configurations
ExperimentRegistry = ConfigRegistry[Config]("ExperimentRegistry")

@ExperimentRegistry.register("segmentation_r2s100k")
@dataclass
class SegmentationR2S100k(Config):
    experiment_name: str = "segmentation_r2s100k"
    # run_id: Optional[str] = None
    model_name: str = "custom_model"
    model: ModelConfig = field(init=False)
    # model: ModelConfig = None
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            batch_size=16,
            epochs=100,
            use_amp=False,
            grad_clip_val=1.0,
            optimizer=OptimizerConfig(
                name="adamw",
                learning_rate=0.001,
                weight_decay=0.01
            ),
            scheduler=SchedulerConfig(
                name="cosine",
                warmup_epochs=5
            )
        )
    )
    data: DataConfig = field(default_factory=lambda: DataRegistry.get("r2s100k")())
    seed: int = 0

    def __post_init__(self):
        self.model = ModelRegistry.get(self.model_name)()
        if self.model.decoder.name == 'mask2former_head':
            print("Mask2Former is used, changing to instance segmentation data format")
            self.data.task_type = 'instance_segmentation'
        self.model.decoder.num_classes = self.data.num_classes
        self.data.input_size = self.model.input_size
        if 'clip' in self.model.name:
            print("CLIP is used, changing to its default mean and std for image_transform")
            self.data.mean = [0.48145466, 0.4578275, 0.40821073]
            self.data.std = [0.26862954, 0.26130258, 0.27577711]
        
        
@ExperimentRegistry.register("semseg_bdd100k")
@dataclass
class SemsegBDD100k(Config):
    experiment_name: str = "semseg_bdd100k"
    model_name: str = "custom_model"
    model: ModelConfig = field(init=False)
    # model: ModelConfig = None
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            batch_size=16,
            epochs=100,
            use_amp=False,
            grad_clip_val=1.0,
            optimizer=OptimizerConfig(
                name="adamw",
                learning_rate=0.001,
                weight_decay=0.01
            ),
            scheduler=SchedulerConfig(
                name="cosine",
                warmup_epochs=5
            )
        )
    )
    data: DataConfig = field(default_factory=lambda: DataRegistry.get("semseg_bdd100k")())
    seed: int = 0

    def __post_init__(self):
        self.model = ModelRegistry.get(self.model_name)()
        self.data.label_colors_list, self.data.class_names = self.data.parse_color_and_names()
        self.data.num_classes = len(self.data.class_names)

        if self.model.decoder.name in ['mask2former_head', 'open_mask2former_head']:
            print("Mask2Former is used, changing to instance segmentation data format")
            self.data.task_type = 'instance_segmentation'
        
        self.model.decoder.num_classes = self.data.num_classes
        self.data.input_size = self.model.input_size
        if 'clip' in self.model.name:
            print("CLIP is used, changing to its default mean and std for image_transform")
            self.data.mean = [0.48145466, 0.4578275, 0.40821073]
            self.data.std = [0.26862954, 0.26130258, 0.27577711]
        
        _, self.model.decoder.class_names = self.data.parse_color_and_names()
        print(f"Get class names for CLIP: {self.model.decoder.class_names}")


@ExperimentRegistry.register("semseg_lane_drivable_bdd100k")
@dataclass
class SemsegLaneDrivableBDD100k(Config):
    experiment_name: str = "semseg_lane_drivable_bdd100k"
    model_name: str = "custom_model"
    model: ModelConfig = field(init=False)
    # model: ModelConfig = None
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            batch_size=16,
            epochs=100,
            use_amp=False,
            grad_clip_val=1.0,
            optimizer=OptimizerConfig(
                name="adamw",
                learning_rate=0.001,
                weight_decay=0.01
            ),
            scheduler=SchedulerConfig(
                name="cosine",
                warmup_epochs=5
            )
        )
    )
    data: DataConfig = field(default_factory=lambda: DataRegistry.get("multitask_bdd100k")())
    seed: int = 0

    # tasks: List[str] = field(default_factory=lambda: ['sem_seg', 'lane', 'drivable'])
    tasks: List[str] = field(default_factory=lambda: ['sem_seg', 'drivable'])

    num_class_dict: Dict = None

    def __post_init__(self):
        self.model = ModelRegistry.get(self.model_name)()
        self.data.tasks = self.tasks
        self.num_class_dict = {}

        for t in self.tasks:
            _, class_names = self.data.parse_color_and_names(t)
            self.num_class_dict[t] = len(class_names)
        print("Initialize Experiment with Multi-Task Training...")
        print(f"Tasks: {self.tasks}")
        print(f"Number of classes per task: {self.num_class_dict}")

        self.model.decoder.num_class_dict = self.num_class_dict
        # self.data.num_classes = len(self.data.class_names)
        self.data.num_classes = self.model.decoder.num_classes
        if self.model.decoder.name in ['mask2former_head', 'open_mask2former_head', 'multitask_mask2former_head']:
            print("Mask2Former is used, changing to instance segmentation data format")
            self.data.task_type = 'instance_segmentation'
        
        # self.model.decoder.num_classes = self.data.num_classes
        self.data.input_size = self.model.input_size
        if 'clip' in self.model.name:
            print("CLIP is used, changing to its default mean and std for image_transform")
            self.data.mean = [0.48145466, 0.4578275, 0.40821073]
            self.data.std = [0.26862954, 0.26130258, 0.27577711]
        
        # _, self.model.decoder.class_names = self.data.parse_color_and_names()
        # print(f"Get class names for CLIP: {self.model.decoder.class_names}")

@ExperimentRegistry.register("semseg_bdd100k_r2s100k")
@dataclass
class SemsegBDD100kR2S100k(Config):
    experiment_name: str = "semseg_bdd100k_r2s100k"
    model_name: str = "custom_model"
    model: ModelConfig = field(init=False)
    # model: ModelConfig = None
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            batch_size=16,
            epochs=100,
            use_amp=False,
            grad_clip_val=1.0,
            optimizer=OptimizerConfig(
                name="adamw",
                learning_rate=0.001,
                weight_decay=0.01
            ),
            scheduler=SchedulerConfig(
                name="cosine",
                warmup_epochs=5
            )
        )
    )
    data: DataConfig = field(default_factory=lambda: DataRegistry.get("semseg_bdd100k_r2s100k")())
    seed: int = 0

    # tasks: List[str] = field(default_factory=lambda: ['sem_seg', 'lane', 'drivable'])
    tasks: List[str] = field(default_factory=lambda: ['bdd100k', 'r2s100k'])
    num_obj_queries_dict: Dict = field(default_factory=lambda: {'bdd100k': 100, 'r2s100k': 100})

    num_class_dict: Dict = None

    def __post_init__(self):
        self.model = ModelRegistry.get(self.model_name)()
        self.data.tasks = self.tasks
        self.num_class_dict = {}

        for t in self.tasks:
            _, class_names = self.data.parse_color_and_names(t)
            self.num_class_dict[t] = len(class_names)
        print("Initialize Experiment with Multi-Task Training...")
        print(f"Tasks: {self.tasks}")
        print(f"Number of classes per task: {self.num_class_dict}")

        self.model.decoder.num_class_dict = self.num_class_dict
        self.model.decoder.num_obj_queries_dict = self.num_obj_queries_dict
        # self.data.num_classes = len(self.data.class_names)
        self.data.num_classes = self.model.decoder.num_classes
        if self.model.decoder.name in ['mask2former_head', 'open_mask2former_head', 'multitask_mask2former_head']:
            print("Mask2Former is used, changing to instance segmentation data format")
            self.data.task_type = 'instance_segmentation'
        
        # self.model.decoder.num_classes = self.data.num_classes
        self.data.input_size = self.model.input_size
        if 'clip' in self.model.name:
            print("CLIP is used, changing to its default mean and std for image_transform")
            self.data.mean = [0.48145466, 0.4578275, 0.40821073]
            self.data.std = [0.26862954, 0.26130258, 0.27577711]
        
