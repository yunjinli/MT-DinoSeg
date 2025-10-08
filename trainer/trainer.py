import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import yaml
import datetime
from torch.cuda.amp import autocast, GradScaler
from config.base import Config
from utility.metrics import AverageMeter, ConfusionMatrix
from utility.distributed import (
    is_distributed, 
    get_rank, 
    get_world_size, 
    is_main_process,
    reduce_dict,
    save_on_master,
    synchronize,
    gather_tensor
)
from tqdm import tqdm
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import io
from PIL import Image
import torch.nn.functional as F
from .criterion import SetCriterion
from .matcher import HungarianMatcher
# import torch.distributed as dist
from utility.ema import EMA
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from torchvision.utils import make_grid
import math
import torchvision
class Trainer:
    """Trainer class for handling the training loop with distributed support."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        run_id: Optional[str] = None,
        resume_from: Optional[Union[str, Path]] = None,
        resume_training: bool = False,
        rank: int = 0,
        world_size: int = 1
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to train on
            run_id: Unique run identifier
            resume_from: Path to checkpoint to resume from
            resume_training: Whether to resume training
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.rank = rank
        self.world_size = world_size
        
        # Set device
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.use_amp = getattr(self.config.training, 'use_amp', False)
        self.scaler = GradScaler() if self.use_amp and torch.cuda.is_available() else None
        
        if is_main_process() and self.use_amp:
            print(f"Using Automatic Mixed Precision (AMP)")
            if not torch.cuda.is_available():
                print("Warning: AMP enabled but CUDA not available, falling back to FP32")
        
        # Create base experiment directory (only on main process)
        if is_main_process():
            self.experiment_dir = Path(f"outputs/{config.experiment_name}")
            os.makedirs(self.experiment_dir, exist_ok=True)
        else:
            self.experiment_dir = Path(f"outputs/{config.experiment_name}")

        # Handle resume logic
        self.start_epoch = 0
        self.resume_checkpoint = None

        if resume_from or resume_training:
            self.resume_checkpoint = self._find_checkpoint_to_resume(resume_from, run_id)
            if self.resume_checkpoint:
                # Use the run_id from the checkpoint if resuming
                checkpoint_run_id = self._extract_run_id_from_checkpoint(self.resume_checkpoint)
                if checkpoint_run_id:
                    run_id = checkpoint_run_id
                    if is_main_process():
                        print(f"Resuming training from {self.resume_checkpoint}")

        # Generate a unique run ID if not provided
        if run_id is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{timestamp}"
            if is_main_process():
                print(f"run_id is not configured -> Create new run_id: {run_id}")
        
        # Create run-specific directory (only on main process)
        if is_main_process():
            self.output_dir = self.experiment_dir / f"run_{run_id}"
            if not self.resume_checkpoint and self.output_dir.exists():
                print(f"Warning: Run directory {self.output_dir} already exists. Adding unique suffix.")
                import uuid
                self.output_dir = self.experiment_dir / f"run_{run_id}__{uuid.uuid4().hex[:6]}"
            
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Saving outputs to {self.output_dir}")
        else:
            self.output_dir = self.experiment_dir / f"run_{run_id}"

        # Synchronize all processes before continuing
        if is_distributed():
            synchronize()
            
       # Set up optimizer (only needed for training)
        if self.train_loader is not None:
            self.optimizer = self._create_optimizer()
            # Set up learning rate scheduler
            self.scheduler = self._create_scheduler()
        else:
            self.optimizer = None
            self.scheduler = None
        
        # Set up loss function based on task type
        # if hasattr(config.data, 'task_type') and config.data.task_type == 'segmentation':
        #     # For segmentation tasks
        #     self.criterion = nn.CrossEntropyLoss(ignore_index=config.data.ignore_index 
        #                                         if hasattr(config.data, 'ignore_index') else 255)
        # else:
        #     # Default for classification
        #     self.criterion = nn.CrossEntropyLoss()

        if self.config.data.task_type == 'semantic_segmentation':
            self.criterion = nn.CrossEntropyLoss()
            self.criterion_eval = nn.CrossEntropyLoss(ignore_index=config.data.ignore_index 
                                                if hasattr(config.data, 'ignore_index') else 255)
        elif self.config.data.task_type == 'instance_segmentation':
            # Loss parameters:
            deep_supervision = self.config.model.decoder.deep_supervision
            no_object_weight = self.config.model.decoder.no_object_weight

            # loss weights
            class_weight = self.config.model.decoder.class_weight
            dice_weight = self.config.model.decoder.dice_weight
            mask_weight = self.config.model.decoder.mask_weight

            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=self.config.model.decoder.train_num_points,
            )

            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

            if deep_supervision:
                dec_layers = self.config.model.decoder.dec_layers
                aux_weight_dict = {}
                for i in range(dec_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            
            losses = ["labels", "masks"]

            self.criterion = SetCriterion(
                self.config.data.num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=list(no_object_weight.values())[0],
                losses=losses,
                num_points=self.config.model.decoder.train_num_points,
                oversample_ratio=self.config.model.decoder.oversample_ratio,
                importance_sample_ratio=self.config.model.decoder.importance_sample_ratio,
            ).to(device)
        
        # Initialize training state variables
        self.best_val_loss = float('inf')
        self.best_miou = 0.0
        self.best_epoch = 0
        self.best_metrics = {}
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'mean_iou': [],
            'pixel_accuracy': [],
            'learning_rate': []
        }
        
        # Resume from checkpoint if specified
        if self.resume_checkpoint:
            self._load_checkpoint(self.resume_checkpoint)
        else:
            # Save initial configuration for new runs (only on main process)
            if is_main_process():
                self.save_run_config(config)

        # Initialize TensorBoard writer for all processes
        self.writer = None
        
        if is_main_process():
            tensorboard_dir = self.output_dir / "tensorboard"
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
            
            # Log model graph if not resuming
            # if not self.resume_checkpoint:
            #     self.log_model_graph()
            
            print(f"TensorBoard logs will be saved to: {tensorboard_dir}")
            print(f"Start TensorBoard with: tensorboard --logdir {tensorboard_dir}")
        
        
        # For visualization, store class colors
        if hasattr(config.data, 'label_colors_list'):
            self.class_colors = torch.tensor(config.data.label_colors_list, 
                                           dtype=torch.float32) / 255.0
        else:
            # Default colormap if not provided
            self.class_colors = None
    
    def test(self) -> Dict[str, Any]:
        """
        Run inference on test dataset.
        
        Returns:
            Dictionary with test metrics and statistics
        """
        if self.test_loader is None:
            raise ValueError("No test loader provided for testing")
        
        if is_main_process():
            print("Starting test evaluation...")
        
        start_time = time.time()
        
        # Run test evaluation
        test_metrics = self.test_epoch()
        # test_metrics = self.test_epoch_r2s() ## has verified that it's identical to our computation
        
        # Calculate test time
        test_time = time.time() - start_time
        
        # Save test results
        if is_main_process():
            self._save_test_results(test_metrics, test_time)
            
            # Print test results
            print(f"\nTest Results:")
            print(f"  Test Loss: {test_metrics.get('test_loss', 'N/A'):.4f}")
            print(f"  Pixel Accuracy: {test_metrics.get('pixel_accuracy', 'N/A'):.4f}")
            print(f"  Mean Accuracy: {test_metrics.get('mean_accuracy', 'N/A'):.4f}")
            print(f"  Mean IoU: {test_metrics.get('mean_iou', 'N/A'):.4f}")
            print(f"  Freq Weighted IoU: {test_metrics.get('fw_iou', 'N/A'):.4f}")
            print(f"  Test time: {test_time:.2f} seconds")
            
            # Print per-class IoU if available
            if 'iou' in test_metrics:
                print("\n  IoU per class:")
                class_names = self.config.data.class_names if hasattr(self.config.data, 'class_names') else None
                for i, iou in enumerate(test_metrics['iou']):
                    class_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
                    print(f"    {class_name}: {iou:.4f}")
        
        return {
            'test_metrics': test_metrics,
            'test_time': test_time
        }

    def test_epoch(self) -> Dict[str, float]:
        """
        Run inference for one epoch on test data.
        
        Returns:
            Dictionary with test metrics
        """
        self.model.eval()
        
        loss_meter = AverageMeter()
        cuda_mem = AverageMeter()

        # Initialize confusion matrix for IoU calculation
        confusion_matrix = ConfusionMatrix(self.config.data.num_classes, self.config.data.ignore_index)
        if is_main_process():
            print(f"Ignore indices: {self.config.data.ignore_index}")
        # confusion_matrix = ConfusionMatrix(self.config.data.num_classes, 0)
        
        # Storage for test samples visualization
        num_samples_to_visualize = min(8, self.config.training.batch_size if hasattr(self.config.training, 'batch_size') else 4)
        samples_visualized = 0
        test_samples = []

        with torch.no_grad():
            # Only show progress bar on main process
            if is_main_process():
                pbar = tqdm(self.test_loader, desc='[TESTING]')
            else:
                pbar = self.test_loader

            for batch_idx, batch in enumerate(pbar):
                # Get data
                if isinstance(batch, dict):
                    inputs = batch['image'].to(self.device)
                    targets = batch['target'].to(self.device)
                else:  # Assume tuple (inputs, targets)
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    if not isinstance(targets, list):
                        targets = targets.to(self.device)
                
                # Run forward pass
                outputs = self.model(inputs)

                if self.config.data.task_type == 'semantic_segmentation':
                    outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[1:], mode="bilinear", align_corners=False)
                    loss = self.criterion_eval(outputs, targets)
                elif self.config.data.task_type == 'instance_segmentation':
                    loss = self.criterion(outputs, targets)
                    if isinstance(loss, dict):
                        for k in list(loss.keys()):
                            if k in self.criterion.weight_dict:
                                loss[k] *= self.criterion.weight_dict[k]
                            else:
                                # remove this loss if not specified in `weight_dict`
                                loss.pop(k)
                        loss_dict = loss.copy()
                        loss = sum(loss_dict.values())
                
                # Update metrics
                loss_meter.update(loss.item(), inputs.size(0))
                if torch.cuda.is_available():
                    cuda_mem.update(torch.cuda.max_memory_allocated(device=None) / (1024 * 1024 * 1024))
                
                # Update confusion matrix for IoU calculation
                if isinstance(outputs, dict) and isinstance(targets, list):
                    ## process mask2former head outputs and targets into [B, C, H, W] and [B, H, W]
                    target_hw = targets[0]['masks'].shape[1:]
                    mask_cls_results = outputs["pred_logits"]
                    mask_pred_results = outputs["pred_masks"]
                    # upsample masks
                    mask_pred_results = F.interpolate(
                        mask_pred_results,
                        size=target_hw,
                        mode="bilinear",
                        align_corners=False,
                    )

                    outputs = []
                    for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):
                        # semantic segmentation inference
                        if is_distributed():
                            semmap = self.model.module.decoder.semantic_inference(mask_cls_result, mask_pred_result)
                        else:
                            semmap = self.model.decoder.semantic_inference(mask_cls_result, mask_pred_result)
                        outputs.append(semmap)
                    outputs = torch.stack(outputs)
                    targets = self.test_loader.dataset.instances_to_semantic(targets=targets, ignore_index=self.config.data.ignore_index)
                    
                confusion_matrix.update(outputs, targets)
                
                # Store samples for visualization
                if samples_visualized < num_samples_to_visualize:
                    for i in range(min(num_samples_to_visualize - samples_visualized, inputs.shape[0])):
                        test_samples.append({
                            'image': inputs[i].cpu(),
                            'target': targets[i].cpu(),
                            'prediction': outputs[i].cpu(),
                            'sample_id': f"batch_{batch_idx}_sample_{i}"
                        })
                        samples_visualized += 1
                        if samples_visualized >= num_samples_to_visualize:
                            break
                
                # Update progress bar only on main process
                if is_main_process():
                    monitor = {
                        'Loss': f'{loss_meter.val:.4f} ({loss_meter.avg:.4f})',
                        'CUDA': f'{cuda_mem.val:.2f} ({cuda_mem.avg:.2f})'
                    }
                    pbar.set_postfix(monitor)

        # Gather metrics from all processes (similar to validation)
        if is_distributed():
            # Gather basic loss and memory stats
            process_stats = torch.tensor([
                loss_meter.avg,
                cuda_mem.avg,
                loss_meter.sum,   # Total loss
                loss_meter.count  # Total samples
            ], device=self.device)
            
            gathered_stats = gather_tensor(process_stats)
            
            # Gather and sum confusion matrices for accurate global metrics
            if confusion_matrix.mat is not None:
                cm_tensor = torch.tensor(confusion_matrix.mat, dtype=torch.float32, device=self.device)
            else:
                cm_tensor = torch.zeros((self.config.data.num_classes, self.config.data.num_classes), 
                                      dtype=torch.float32, device=self.device)
            
            gathered_cm = gather_tensor(cm_tensor)
            
            if is_main_process():
                world_size = get_world_size()
                
                # Extract individual process stats
                all_avg_losses = gathered_stats[0::4]
                all_cuda_mem = gathered_stats[1::4]
                all_total_losses = gathered_stats[2::4]
                all_sample_counts = gathered_stats[3::4]
                
                # Sum confusion matrices to get global confusion matrix
                num_classes = self.config.data.num_classes
                global_cm = gathered_cm.view(world_size, num_classes, num_classes).sum(dim=0)
                
                # Compute global metrics from the summed confusion matrix
                confusion_matrix.mat = global_cm.cpu().numpy().astype(np.int64)
                global_metrics = confusion_matrix.compute()
                
                # Compute global average loss (weighted by sample count)
                total_samples = all_sample_counts.sum().item()
                global_loss = all_total_losses.sum().item() / total_samples if total_samples > 0 else 0
                
                # Log test results to TensorBoard
                if self.writer:
                    # Use current time as "epoch" for test results
                    test_step = int(time.time())
                    self.writer.add_scalar('Test/loss', global_loss, test_step)
                    self.writer.add_scalar('Test/pixel_accuracy', global_metrics['pixel_accuracy'], test_step)
                    self.writer.add_scalar('Test/mean_accuracy', global_metrics['mean_accuracy'], test_step)
                    self.writer.add_scalar('Test/mean_iou', global_metrics['mean_iou'], test_step)
                    self.writer.add_scalar('Test/fw_iou', global_metrics['fw_iou'], test_step)
                    
                    # Log per-class IoU
                    class_names = self.config.data.class_names if hasattr(self.config.data, 'class_names') else None
                    for i, iou in enumerate(global_metrics['iou']):
                        class_name = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
                        self.writer.add_scalar(f'Test/IoU_per_class/{class_name}', iou, test_step)
                    
                    # Visualize test samples
                    for idx, sample in enumerate(test_samples):
                        self.log_segmentation_sample(
                            sample['image'], 
                            sample['prediction'], 
                            sample['target'],
                            test_step, 
                            f"test_{sample['sample_id']}"
                        )
                
                return {
                    'test_loss': global_loss,
                    'pixel_accuracy': global_metrics['pixel_accuracy'],
                    'mean_accuracy': global_metrics['mean_accuracy'],
                    'mean_iou': global_metrics['mean_iou'],
                    'fw_iou': global_metrics['fw_iou'],
                    'iou': global_metrics['iou'],
                    'confusion_matrix': global_cm.cpu().numpy()
                }
            else:
                # Non-main processes return dummy metrics
                return {
                    'test_loss': loss_meter.avg,
                    'pixel_accuracy': 0.0,
                    'mean_accuracy': 0.0,
                    'mean_iou': 0.0,
                    'fw_iou': 0.0
                }
        else:
            # Single process - compute metrics normally
            metrics = confusion_matrix.compute()
            
            # Log test results to TensorBoard
            if is_main_process() and self.writer:
                test_step = int(time.time())
                self.writer.add_scalar('Test/loss', loss_meter.avg, test_step)
                self.writer.add_scalar('Test/pixel_accuracy', metrics['pixel_accuracy'], test_step)
                self.writer.add_scalar('Test/mean_accuracy', metrics['mean_accuracy'], test_step)
                self.writer.add_scalar('Test/mean_iou', metrics['mean_iou'], test_step)
                self.writer.add_scalar('Test/fw_iou', metrics['fw_iou'], test_step)
                
                # Log per-class IoU
                class_names = self.config.data.class_names if hasattr(self.config.data, 'class_names') else None
                for i, iou in enumerate(metrics['iou']):
                    class_name = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
                    self.writer.add_scalar(f'Test/IoU_per_class/{class_name}', iou, test_step)
                
                # Visualize test samples
                for idx, sample in enumerate(test_samples):
                    self.log_segmentation_sample(
                        sample['image'], 
                        sample['prediction'], 
                        sample['target'],
                        test_step, 
                        f"test_{sample['sample_id']}"
                    )
            
            return {
                'test_loss': loss_meter.avg,
                'pixel_accuracy': metrics['pixel_accuracy'],
                'mean_accuracy': metrics['mean_accuracy'],
                'mean_iou': metrics['mean_iou'],
                'fw_iou': metrics['fw_iou'],
                'iou': metrics['iou'],
                'confusion_matrix': metrics.get('confusion_matrix', confusion_matrix.mat)
            }

    def _save_test_results(self, test_metrics: Dict[str, Any], test_time: float):
        """Save test results to files."""
        results_dir = self.output_dir / "test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed metrics as JSON
        test_results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'test_time_seconds': test_time,
            'metrics': {}
        }
        
        # Copy all metrics except numpy arrays
        for key, value in test_metrics.items():
            if isinstance(value, np.ndarray):
                test_results['metrics'][key] = value.tolist()
            else:
                test_results['metrics'][key] = value
        
        with open(results_dir / "test_metrics.json", 'w') as f:
            json.dump(test_results, f, indent=4)
        
        # Save human-readable summary
        with open(results_dir / "test_summary.txt", 'w') as f:
            f.write(f"Test Results Summary\n")
            f.write(f"==================\n\n")
            f.write(f"Timestamp: {test_results['timestamp']}\n")
            f.write(f"Test time: {test_time:.2f} seconds\n\n")
            f.write(f"Overall Metrics:\n")
            f.write(f"  Test Loss: {test_metrics.get('test_loss', 'N/A'):.4f}\n")
            f.write(f"  Pixel Accuracy: {test_metrics.get('pixel_accuracy', 'N/A'):.4f}\n")
            f.write(f"  Mean Accuracy: {test_metrics.get('mean_accuracy', 'N/A'):.4f}\n")
            f.write(f"  Mean IoU: {test_metrics.get('mean_iou', 'N/A'):.4f}\n")
            f.write(f"  Freq Weighted IoU: {test_metrics.get('fw_iou', 'N/A'):.4f}\n\n")
            
            if 'iou' in test_metrics:
                f.write(f"Per-Class IoU:\n")
                class_names = self.config.data.class_names if hasattr(self.config.data, 'class_names') else None
                for i, iou in enumerate(test_metrics['iou']):
                    class_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
                    f.write(f"  {class_name}: {iou:.4f}\n")
        
        # Save confusion matrix as CSV
        if 'confusion_matrix' in test_metrics:
            np.savetxt(results_dir / "confusion_matrix.csv", 
                      test_metrics['confusion_matrix'], 
                      delimiter=',', fmt='%d')
        
        print(f"Test results saved to {results_dir}")

    def _find_checkpoint_to_resume(self, resume_from: Optional[str], run_id: Optional[str]) -> Optional[Path]:
        """Find the checkpoint to resume from."""
        if resume_from:
            # Direct path to checkpoint specified
            checkpoint_path = Path(resume_from)
            if checkpoint_path.exists():
                return checkpoint_path
            else:
                if is_main_process():
                    print(f"Checkpoint not found at {resume_from}")
                return None
        
        # Try to find latest checkpoint in current or specified run
        search_dirs = []
        
        if run_id:
            # Look in specific run directory
            search_dirs.append(self.experiment_dir / f"run_{run_id}")
        else:
            # Look in all run directories, prioritizing the most recent
            search_dirs.extend(sorted(self.experiment_dir.glob("run_*"), reverse=True))
        
        for search_dir in search_dirs:
            if search_dir.exists():
                checkpoints_dir = search_dir / "checkpoints"
                if checkpoints_dir.exists():
                    # Look for best model first, then latest checkpoint
                    best_checkpoint = checkpoints_dir / "best_model.pth"
                    if best_checkpoint.exists():
                        return best_checkpoint
                    
                    # Find latest numbered checkpoint
                    checkpoint_files = list(checkpoints_dir.glob("checkpoint_epoch_*.pth"))
                    if checkpoint_files:
                        # Sort by epoch number
                        latest_checkpoint = max(checkpoint_files, 
                                              key=lambda x: int(x.stem.split('_')[-1]))
                        return latest_checkpoint
        
        if is_main_process():
            print("No checkpoint found to resume from")
        return None

    def _extract_run_id_from_checkpoint(self, checkpoint_path: Path) -> Optional[str]:
        """Extract run_id from checkpoint path."""
        run_dir = checkpoint_path.parent.parent  # Go up from checkpoints/checkpoint.pth to run_*/
        if run_dir.name.startswith("run_"):
            return run_dir.name[4:]  # Remove "run_" prefix
        return None

    def _load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint and restore training state."""
        if is_main_process():
            print(f"Loading checkpoint from {checkpoint_path}")
        
        try:
            # Load checkpoint on each device
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'config' in checkpoint:
                loaded_config = checkpoint['config']
                if loaded_config.experiment_name != self.config.experiment_name:
                    if is_main_process():
                        print(f"Warning: Experiment name mismatch!")
                        print(f"  Checkpoint: {loaded_config.experiment_name}")
                        print(f"  Current: {self.config.experiment_name}")
            
            # Load model state
            if hasattr(self.model, 'module'):
                # DDP model
                self.model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # Load optimizer and scheduler state only if training
            if self.optimizer is not None:
                # Load optimizer state
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
                # Load scheduler state if available
                if 'scheduler_state_dict' in checkpoint and self.scheduler:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if self.use_amp and self.scaler is not None and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                if is_main_process():
                    print(f"Restored AMP scaler state (scale: {self.scaler.get_scale()})")
            
            # Load training progress
            self.start_epoch = checkpoint.get('epoch', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_miou = checkpoint.get('best_miou', 0.0)
            self.best_epoch = checkpoint.get('best_epoch', 0)
            self.best_metrics = checkpoint.get('best_metrics', {})
            
            # Load metrics history if available
            if 'metrics_history' in checkpoint:
                self.metrics_history = checkpoint['metrics_history']
            
            if is_main_process():
                print(f"Resumed from epoch {self.start_epoch}")
                print(f"Best validation mIoU so far: {self.best_miou:.4f} (epoch {self.best_epoch + 1})")
                amp_status = "Enabled" if self.use_amp else "Disabled"
                print(f"Mixed Precision: {amp_status}")

        except Exception as e:
            if is_main_process():
                print(f"Error loading checkpoint: {e}")
                print("Starting training from scratch...")
            self.start_epoch = 0

    def save_run_config(self, config):
        """Save experiment configuration as YAML for reproducibility."""
        if not is_main_process():
            return
            
        try:
            # Convert config to dictionary recursively
            config_dict = self._config_to_dict(config)
            
            # Save to YAML file
            config_path = self.output_dir / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
                
            print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Warning: Failed to save configuration: {e}")
    
    def _config_to_dict(self, obj):
        """
        Recursively convert configuration objects to dictionaries.
        Handles nested dataclasses, lists, and primitive types.
        """
        if hasattr(obj, '__dict__'):
            # If it's an object with attributes, convert to dict
            result = {}
            for key, value in vars(obj).items():
                # Skip private attributes (starting with _)
                if not key.startswith('_'):
                    result[key] = self._config_to_dict(value)
            return result
        elif isinstance(obj, (list, tuple)):
            # If it's a list or tuple, convert each element
            return [self._config_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            # If it's already a dict, convert its values
            return {key: self._config_to_dict(value) for key, value in obj.items()}
        elif isinstance(obj, (int, float, str, bool, type(None))):
            # Primitive types can be used as-is
            return obj
        else:
            # For any other types, convert to string
            return str(obj)

    def log_model_graph(self):
        """Log model architecture to TensorBoard."""
        if not is_main_process():
            return
            
        try:
            # Create a sample input based on the first batch
            sample_input = next(iter(self.train_loader))[0][:1].to(self.device)
            self.writer.add_graph(self.model, sample_input)
            print("Model graph added to TensorBoard")
        except Exception as e:
            print(f"Failed to add model graph to TensorBoard: {e}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        opt_config = self.config.training.optimizer
        
        effective_batch_size = self.config.training.batch_size * self.world_size
        lr_scale = effective_batch_size / self.config.training.batch_size
        # Get model parameters with separate learning rates for encoder and decoder
        params = []
        
        # Add encoder parameters
        if not is_distributed():
            encoder_params = []
            for name, param in self.model.named_parameters():
                if name.startswith('encoder.') and param.requires_grad:
                    encoder_params.append(param)
        else:
            encoder_params = []
            for name, param in self.model.named_parameters():
                if name.startswith('module.encoder.') and param.requires_grad:
                    encoder_params.append(param)

        if encoder_params:
        # if encoder_params and not self.config.model.encoder.freeze:
            params.append({
                'params': encoder_params,
                'lr': opt_config.learning_rate * lr_scale * 0.1  # Lower LR for pretrained encoder
            })
        
        # Add decoder parameters
        if not is_distributed():
            decoder_params = []
            for name, param in self.model.named_parameters():
                if name.startswith('decoder.') and param.requires_grad:
                    decoder_params.append(param)
        else:
            decoder_params = []
            for name, param in self.model.named_parameters():
                if name.startswith('module.decoder.') and param.requires_grad:
                    decoder_params.append(param)
        
        if decoder_params:
            params.append({
                'params': decoder_params,
                'lr': opt_config.learning_rate * lr_scale
            })
        
        # Create optimizer
        if opt_config.name.lower() == 'adam':
            return optim.Adam(
                params,
                betas=getattr(opt_config, 'betas', (0.9, 0.999)),
                weight_decay=opt_config.weight_decay
            )
        elif opt_config.name.lower() == 'adamw':
            return optim.AdamW(
                params,
                betas=getattr(opt_config, 'betas', (0.9, 0.999)),
                weight_decay=opt_config.weight_decay
            )
        elif opt_config.name.lower() == 'sgd':
            return optim.SGD(
                params,
                momentum=getattr(opt_config, 'momentum', 0.9),
                weight_decay=opt_config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.name}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        sched_config = self.config.training.scheduler
        if sched_config is None:
            return None
        
        if sched_config.name.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs - getattr(sched_config, 'warmup_epochs', 0),
                eta_min=getattr(sched_config, 'min_lr', 1e-6)
            )
        elif sched_config.name.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=getattr(sched_config, 'step_size', 30),
                gamma=getattr(sched_config, 'gamma', 0.1)
            )
        elif sched_config.name.lower() == 'multistep':
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=getattr(sched_config, 'milestones', [50, 80]),
                gamma=getattr(sched_config, 'gamma', 0.1)
            )
        elif sched_config.name.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=getattr(sched_config, 'factor', 0.1),
                patience=getattr(sched_config, 'patience', 5),
                min_lr=getattr(sched_config, 'min_lr', 1e-6)
            )
        else:
            return None
    
    def _save_metrics_plot(self, metrics_history: Dict[str, List[float]]) -> None:
        """
        Save plots of training and validation metrics.
        
        Args:
            metrics_history: Dictionary of metrics history
        """
        if not is_main_process() or not metrics_history:
            return
        
        # Create plots directory
        plots_dir = self.output_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot loss
        fig, ax = plt.subplots(figsize=(10, 5))
        epochs = range(1, len(metrics_history['train_loss']) + 1)
        
        ax.plot(epochs, metrics_history['train_loss'], 'b-', label='Training Loss')
        if 'val_loss' in metrics_history and metrics_history['val_loss']:
            ax.plot(epochs, metrics_history['val_loss'], 'r-', label='Validation Loss')
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True)

        ax.figure.savefig(plots_dir / "loss_history.png", bbox_inches='tight')
        plt.close(fig)
        
        # Plot mIoU and pixel accuracy
        if 'mean_iou' in metrics_history and metrics_history['mean_iou']:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(epochs, metrics_history['mean_iou'], 'g-', label='Mean IoU')
            
            if 'pixel_accuracy' in metrics_history and metrics_history['pixel_accuracy']:
                ax.plot(epochs, metrics_history['pixel_accuracy'], 'm-', label='Pixel Accuracy')
            
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Metric Value')
            ax.set_title('Validation Metrics')
            ax.legend()
            ax.grid(True)
            ax.figure.savefig(plots_dir / "metrics_history.png", bbox_inches='tight')
            plt.close(fig)
        
        # Plot learning rate
        if 'learning_rate' in metrics_history and metrics_history['learning_rate']:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(epochs, metrics_history['learning_rate'], 'k-', label='Learning Rate')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')  # Log scale often better for learning rate
            ax.grid(True)
            ax.figure.savefig(plots_dir / "lr_history.png", bbox_inches='tight')
            plt.close(fig)
            
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Returns:
            Dictionary with training statistics
        """
        if is_main_process():
            print(f"Starting training from epoch {self.start_epoch + 1}")
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config.training.epochs):
            # Set epoch for distributed sampler
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            # Training step
            train_metrics = self.train_epoch(epoch)
            
            # Only update history on main process
            if is_main_process():
                self.metrics_history['train_loss'].append(train_metrics['train_loss'])
                
                # Current learning rate
                lr = self.optimizer.param_groups[0]['lr']
                self.metrics_history['learning_rate'].append(lr)
            
            # Validation step
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.validate_epoch(epoch)
                
                # Update metrics history on main process
                if is_main_process():
                    self.metrics_history['val_loss'].append(val_metrics.get('val_loss', 0))
                    self.metrics_history['mean_iou'].append(val_metrics.get('mean_iou', 0))
                    self.metrics_history['pixel_accuracy'].append(val_metrics.get('pixel_accuracy', 0))
                    
                    # Check for best model (using mIoU)
                    current_miou = val_metrics.get('mean_iou', 0.0)
                    if current_miou > self.best_miou:
                        self.best_miou = current_miou
                        self.best_val_loss = val_metrics.get('val_loss', float('inf'))
                        self.best_epoch = epoch
                        self.best_metrics = val_metrics.copy()
                        
                        # Save best model checkpoint with metrics
                        self.save_checkpoint(epoch, is_best=True, metrics=val_metrics)
                    
                    # Also keep track of best loss
                    if val_metrics.get('val_loss', float('inf')) < self.best_val_loss and epoch != self.best_epoch:
                        self.best_val_loss = val_metrics.get('val_loss', float('inf'))

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('val_loss', 0))
                else:
                    self.scheduler.step()
            
            if is_main_process():
                self.save_checkpoint(epoch, metrics=val_metrics)
            
            # Print epoch summary on main process
            if is_main_process():
                lr = self.optimizer.param_groups[0]['lr']
                val_loss_str = f"{val_metrics.get('val_loss', 'N/A'):.4f}" if 'val_loss' in val_metrics else 'N/A'
                miou_str = f"{val_metrics.get('mean_iou', 'N/A'):.4f}" if 'mean_iou' in val_metrics else 'N/A'
                print(f"Epoch {epoch + 1}/{self.config.training.epochs} - "
                      f"Train loss: {train_metrics['train_loss']:.4f}, "
                      f"Val loss: {val_loss_str}, "
                      f"mIoU: {miou_str}, "
                      f"LR: {lr:.6f}")
        
        # Save metrics history plot
        if is_main_process():
            self._save_metrics_plot(self.metrics_history)
        
        # Calculate training time
        total_time = time.time() - start_time
        
        # Return final statistics
        return {
            'total_epochs': self.config.training.epochs,
            'resumed_from_epoch': self.start_epoch + 1,
            'best_val_loss': self.best_val_loss,
            'best_miou': self.best_miou,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'training_time': total_time,
            'metrics_history': self.metrics_history
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        loss_meter = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        cuda_mem = AverageMeter()
        
        start = time.time()
        
        # Only show progress bar on main process
        if is_main_process():
            pbar = tqdm(self.train_loader)
            pbar.set_description(f'[TRAINING] Epoch {epoch + 1}')
        else:
            pbar = self.train_loader

        total_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(pbar):
            # Measure data loading time
            data_time.update(time.time() - start)
            # if batch_idx > 5:
            #     break
            # Get data
            if isinstance(batch, dict):
                inputs = batch['image'].to(self.device)
                targets = batch['target'].to(self.device)
            else:  # Assume tuple (inputs, targets)
                inputs, targets = batch
                inputs = inputs.to(self.device)
                if not isinstance(targets, list):
                    targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()
            
            if self.use_amp and self.scaler is not None:
                # Mixed precision training
                with autocast():
                    outputs = self.model(inputs)
                    if self.config.data.task_type == 'semantic_segmentation':
                        outputs = torch.nn.functional.interpolate(
                            outputs, size=targets.shape[1:], 
                            mode="bilinear", align_corners=False
                        )
                        loss = self.criterion(outputs, targets)
                    elif self.config.data.task_type == 'instance_segmentation':
                        loss = self.criterion(outputs, targets)
                        if isinstance(loss, dict):
                            for k in list(loss.keys()):
                                if k in self.criterion.weight_dict:
                                    loss[k] *= self.criterion.weight_dict[k]
                                else:
                                    # remove this loss if not specified in `weight_dict`
                                    loss.pop(k)
                            loss_dict = loss.copy()
                            loss = sum(loss_dict.values())
                # Scale loss to prevent gradient underflow
                scaled_loss = self.scaler.scale(loss)
                scaled_loss.backward()
                
                # Gradient clipping (if configured)
                if hasattr(self.config.training, 'grad_clip_val') and self.config.training.grad_clip_val:
                    # Unscale before clipping to get true gradient norms
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.grad_clip_val
                    )
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                outputs = self.model(inputs)
                if self.config.data.task_type == 'semantic_segmentation':
                    outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[1:], mode="bilinear", align_corners=False)
                    loss = self.criterion(outputs, targets)
                elif self.config.data.task_type == 'instance_segmentation':
                    loss = self.criterion(outputs, targets)
                    if isinstance(loss, dict):
                        for k in list(loss.keys()):
                            if k in self.criterion.weight_dict:
                                loss[k] *= self.criterion.weight_dict[k]
                            else:
                                # remove this loss if not specified in `weight_dict`
                                loss.pop(k)
                        loss_dict = loss.copy()
                        loss = sum(loss_dict.values())

                loss.backward()
                
                # Gradient clipping if configured
                if hasattr(self.config.training, 'grad_clip_val') and self.config.training.grad_clip_val:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.grad_clip_val
                    )
                
                # Update weights
                self.optimizer.step()
            
            
            # Update metrics
            loss_meter.update(loss.item(), inputs.size(0))
            batch_time.update(time.time() - start)
            if torch.cuda.is_available():
                cuda_mem.update(torch.cuda.max_memory_allocated(device=None) / (1024 * 1024 * 1024))
            
            # Log batch-level metrics only from main process (less frequently)
            if is_main_process() and self.writer and batch_idx % 10 == 0:
                global_step = epoch * total_batches + batch_idx
                self.writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_Rate', current_lr, global_step)

                if self.use_amp and self.scaler is not None:
                    self.writer.add_scalar('AMP/loss_scale', self.scaler.get_scale(), global_step)
            # Update progress bar only on main process
            if is_main_process():
                monitor = {
                    'Loss': f'{loss_meter.val:.4f} ({loss_meter.avg:.4f})',
                    'CUDA': f'{cuda_mem.val:.2f} ({cuda_mem.avg:.2f})'
                }
                if self.use_amp and self.scaler is not None:
                    monitor['Scale'] = f'{self.scaler.get_scale():.0f}'
                pbar.set_postfix(monitor)

            start = time.time()
        
        # Log epoch-level metrics from all processes
        if self.writer:
            if hasattr(self, 'rank_prefix'):
                self.writer.add_scalar(f'{self.rank_prefix}Loss/train_epoch', loss_meter.avg, epoch)
                self.writer.add_scalar(f'{self.rank_prefix}Memory/avg_cuda_memory_gb', cuda_mem.avg, epoch)
            else:
                self.writer.add_scalar('Loss/train_epoch', loss_meter.avg, epoch)
                self.writer.add_scalar('Memory/avg_cuda_memory_gb', cuda_mem.avg, epoch)
        
        # Gather metrics from all processes
        if is_distributed():
            # Prepare metrics to gather
            process_metrics = torch.tensor([
                loss_meter.avg,
                cuda_mem.avg,
                batch_time.avg,
                data_time.avg,
                loss_meter.sum,   # Total loss across all batches
                loss_meter.count  # Total samples processed
            ], device=self.device)
            
            # Gather metrics from all processes to rank 0
            gathered_metrics = gather_tensor(process_metrics)
            
            if is_main_process():
                # Extract gathered metrics
                world_size = get_world_size()
                all_avg_losses = gathered_metrics[0::6]  # Every 6th element starting from 0
                all_cuda_mem = gathered_metrics[1::6]
                all_batch_times = gathered_metrics[2::6]
                all_data_times = gathered_metrics[3::6]
                all_total_losses = gathered_metrics[4::6]
                all_sample_counts = gathered_metrics[5::6]
                
                # Compute global weighted average loss
                total_samples = all_sample_counts.sum().item()
                global_loss = all_total_losses.sum().item() / total_samples if total_samples > 0 else 0
                
                # Log individual process metrics
                for rank in range(world_size):
                    self.writer.add_scalar(f'Loss/train_rank_{rank}', all_avg_losses[rank].item(), epoch)
                    self.writer.add_scalar(f'Memory/cuda_memory_rank_{rank}', all_cuda_mem[rank].item(), epoch)
                    self.writer.add_scalar(f'Timing/batch_time_rank_{rank}', all_batch_times[rank].item(), epoch)
                    self.writer.add_scalar(f'Timing/data_time_rank_{rank}', all_data_times[rank].item(), epoch)
                    self.writer.add_scalar(f'Samples/train_count_rank_{rank}', all_sample_counts[rank].item(), epoch)
                
                # Log aggregated metrics
                self.writer.add_scalar('Loss/train_global', global_loss, epoch)
                self.writer.add_scalar('Loss/train_avg_across_ranks', all_avg_losses.mean().item(), epoch)
                self.writer.add_scalar('Loss/train_min', all_avg_losses.min().item(), epoch)
                self.writer.add_scalar('Loss/train_max', all_avg_losses.max().item(), epoch)
                self.writer.add_scalar('Loss/train_std', all_avg_losses.std().item(), epoch)
                
                self.writer.add_scalar('Memory/cuda_memory_avg', all_cuda_mem.mean().item(), epoch)
                self.writer.add_scalar('Memory/cuda_memory_max', all_cuda_mem.max().item(), epoch)
                
                self.writer.add_scalar('Timing/batch_time_avg', all_batch_times.mean().item(), epoch)
                self.writer.add_scalar('Timing/data_time_avg', all_data_times.mean().item(), epoch)
                
                try:
                    # Log histograms for better visualization
                    self.writer.add_histogram('Distributions/loss_per_rank', all_avg_losses.cpu(), epoch)
                    self.writer.add_histogram('Distributions/memory_per_rank', all_cuda_mem.cpu(), epoch)
                except Exception as e: 
                    print(e)
                return {'train_loss': global_loss}
            else:
                return {'train_loss': loss_meter.avg}
        else:
            # Single process
            if is_main_process() and self.writer:
                self.writer.add_scalar('Loss/train', loss_meter.avg, epoch)
                self.writer.add_scalar('Memory/cuda_memory', cuda_mem.avg, epoch)
            
            return {'train_loss': loss_meter.avg}

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        loss_meter = AverageMeter()
        cuda_mem = AverageMeter()

        # Initialize confusion matrix for IoU calculation
        confusion_matrix = ConfusionMatrix(self.config.data.num_classes, self.config.data.ignore_index)
        
        # Create storage for visualizing a few samples
        num_samples_to_visualize = min(4, self.config.training.batch_size)
        samples_visualized = 0

        with torch.no_grad():
            # Only show progress bar on main process
            if is_main_process():
                pbar = tqdm(self.val_loader)
                pbar.set_description(f'[VALIDATION] Epoch {epoch + 1}')
            else:
                pbar = self.val_loader

            for batch_idx, batch in enumerate(pbar):
                # Get data
                if isinstance(batch, dict):
                    inputs = batch['image'].to(self.device)
                    targets = batch['target'].to(self.device)
                else:  # Assume tuple (inputs, targets)
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    if not isinstance(targets, list):
                        targets = targets.to(self.device)
                
                # Run forward pass
                outputs = self.model(inputs)
                
                if self.config.data.task_type == 'semantic_segmentation':
                    outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[1:], mode="bilinear", align_corners=False)
                    loss = self.criterion_eval(outputs, targets)
                elif self.config.data.task_type == 'instance_segmentation':
                    loss = self.criterion(outputs, targets)
                    if isinstance(loss, dict):
                        for k in list(loss.keys()):
                            if k in self.criterion.weight_dict:
                                loss[k] *= self.criterion.weight_dict[k]
                            else:
                                # remove this loss if not specified in `weight_dict`
                                loss.pop(k)
                        loss_dict = loss.copy()
                        loss = sum(loss_dict.values())
                # Update metrics
                loss_meter.update(loss.item(), inputs.size(0))
                if torch.cuda.is_available():
                    cuda_mem.update(torch.cuda.max_memory_allocated(device=None) / (1024 * 1024 * 1024))
                
                # Update confusion matrix for IoU calculation
                if isinstance(outputs, dict) and isinstance(targets, list):
                    ## process mask2former head outputs and targets into [B, C, H, W] and [B, H, W]
                    target_hw = targets[0]['masks'].shape[1:]
                    mask_cls_results = outputs["pred_logits"]
                    mask_pred_results = outputs["pred_masks"]
                    # print("mask_cls_results:", mask_cls_results.shape)
                    # print("mask_pred_results", mask_pred_results.shape)
                    # print("target_hw: ", target_hw)
                    # upsample masks
                    mask_pred_results = F.interpolate(
                        mask_pred_results,
                        size=target_hw,
                        mode="bilinear",
                        align_corners=False,
                    )

                    outputs = []
                    for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):
                        # semantic segmentation inference
                        if is_distributed():
                            semmap = self.model.module.decoder.semantic_inference(mask_cls_result, mask_pred_result)
                        else:
                            semmap = self.model.decoder.semantic_inference(mask_cls_result, mask_pred_result)
                        outputs.append(semmap)
                    outputs = torch.stack(outputs)
                    targets = self.val_loader.dataset.instances_to_semantic(targets=targets, ignore_index=self.config.data.ignore_index)
                    
                confusion_matrix.update(outputs, targets)
                
                # Update progress bar only on main process
                if is_main_process():
                    monitor = {
                        'Loss': f'{loss_meter.val:.4f} ({loss_meter.avg:.4f})',
                        'CUDA': f'{cuda_mem.val:.2f} ({cuda_mem.avg:.2f})'
                    }
                    pbar.set_postfix(monitor)

                # Visualize samples only on main process
                if is_main_process() and self.writer:
                    try:
                        if samples_visualized < num_samples_to_visualize:
                            sample_idx = np.random.randint(0, inputs.shape[0])
                            self.log_segmentation_sample(
                                inputs[sample_idx], 
                                outputs[sample_idx], 
                                targets[sample_idx],
                                epoch, 
                                f"sample_{samples_visualized}"
                            )
                            samples_visualized += 1
                    except:
                        pass

        # Gather metrics from all processes
        if is_distributed():
            # Gather basic loss and memory stats
            process_stats = torch.tensor([
                loss_meter.avg,
                cuda_mem.avg,
                loss_meter.sum,   # Total loss
                loss_meter.count  # Total samples
            ], device=self.device)
            
            gathered_stats = gather_tensor(process_stats)
            
            # Gather and sum confusion matrices for accurate global metrics
            if confusion_matrix.mat is not None:
                cm_tensor = torch.tensor(confusion_matrix.mat, dtype=torch.float32, device=self.device)
            else:
                # Handle case where no samples were processed
                cm_tensor = torch.zeros((self.config.data.num_classes, self.config.data.num_classes), 
                                      dtype=torch.float32, device=self.device)
            
            gathered_cm = gather_tensor(cm_tensor)
            
            if is_main_process():
                world_size = get_world_size()
                
                # Extract individual process stats
                all_avg_losses = gathered_stats[0::4]
                all_cuda_mem = gathered_stats[1::4]
                all_total_losses = gathered_stats[2::4]
                all_sample_counts = gathered_stats[3::4]
                
                # Sum confusion matrices to get global confusion matrix
                num_classes = self.config.data.num_classes
                global_cm = gathered_cm.view(world_size, num_classes, num_classes).sum(dim=0)
                
                # Compute global metrics from the summed confusion matrix
                confusion_matrix.mat = global_cm.cpu().numpy().astype(np.int64)
                global_metrics = confusion_matrix.compute()
                
                # Compute global average loss (weighted by sample count)
                total_samples = all_sample_counts.sum().item()
                global_loss = all_total_losses.sum().item() / total_samples if total_samples > 0 else 0
                
                # Compute individual process metrics for variance analysis
                process_metrics = []
                for rank in range(world_size):
                    rank_cm = gathered_cm.view(world_size, num_classes, num_classes)[rank].cpu().numpy().astype(np.int64)
                    rank_cm_obj = ConfusionMatrix(num_classes, self.config.data.ignore_index)
                    rank_cm_obj.mat = rank_cm
                    if rank_cm.sum() > 0:  # Only compute if there are samples
                        rank_metrics = rank_cm_obj.compute()
                        process_metrics.append(rank_metrics)
                    else:
                        # Handle empty confusion matrix
                        process_metrics.append({
                            'pixel_accuracy': 0.0,
                            'mean_iou': 0.0,
                            'mean_accuracy': 0.0,
                            'fw_iou': 0.0,
                            'iou': np.zeros(num_classes)
                        })
                
                # Log individual process metrics
                for rank in range(world_size):
                    self.writer.add_scalar(f'Loss/val_rank_{rank}', all_avg_losses[rank].item(), epoch)
                    self.writer.add_scalar(f'Memory/val_cuda_memory_rank_{rank}', all_cuda_mem[rank].item(), epoch)
                    self.writer.add_scalar(f'Metrics/pixel_accuracy_rank_{rank}', process_metrics[rank]['pixel_accuracy'], epoch)
                    self.writer.add_scalar(f'Metrics/mean_iou_rank_{rank}', process_metrics[rank]['mean_iou'], epoch)
                    self.writer.add_scalar(f'Samples/val_count_rank_{rank}', all_sample_counts[rank].item(), epoch)
                
                # Log global (accurate) metrics
                self.writer.add_scalar('Loss/val_global', global_loss, epoch)
                self.writer.add_scalar('Metrics/pixel_accuracy_global', global_metrics['pixel_accuracy'], epoch)
                self.writer.add_scalar('Metrics/mean_accuracy_global', global_metrics['mean_accuracy'], epoch)
                self.writer.add_scalar('Metrics/mean_iou_global', global_metrics['mean_iou'], epoch)
                self.writer.add_scalar('Metrics/fw_iou_global', global_metrics['fw_iou'], epoch)
                
                # Log variance across processes
                process_miou = [m['mean_iou'] for m in process_metrics]
                process_pixel_acc = [m['pixel_accuracy'] for m in process_metrics]
                
                if len(process_miou) > 1:
                    miou_tensor = torch.tensor(process_miou)
                    pixel_acc_tensor = torch.tensor(process_pixel_acc)
                    
                    self.writer.add_scalar('Metrics/mean_iou_std', miou_tensor.std().item(), epoch)
                    self.writer.add_scalar('Metrics/mean_iou_min', miou_tensor.min().item(), epoch)
                    self.writer.add_scalar('Metrics/mean_iou_max', miou_tensor.max().item(), epoch)
                    
                    # Log histograms
                    try:
                        self.writer.add_histogram('Distributions/val_loss_per_rank', all_avg_losses.cpu(), epoch)
                        self.writer.add_histogram('Distributions/val_miou_per_rank', miou_tensor, epoch)
                        self.writer.add_histogram('Distributions/val_pixel_acc_per_rank', pixel_acc_tensor, epoch)
                    except Exception as e: 
                        print(e)
                # Log global per-class IoU
                class_names = self.config.data.class_names if hasattr(self.config.data, 'class_names') else None
                for i, iou in enumerate(global_metrics['iou']):
                    class_name = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
                    self.writer.add_scalar(f'IoU_per_class_global/{class_name}', iou, epoch)
                
                # Create visualization comparing global vs per-process IoU
                if len(process_metrics) > 1:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
                    
                    # Left plot: Global IoU per class
                    num_classes = len(global_metrics['iou'])
                    class_names_plot = class_names if class_names else [f"Class {i}" for i in range(num_classes)]
                    
                    ax1.bar(np.arange(num_classes), global_metrics['iou'])
                    ax1.set_xticks(np.arange(num_classes))
                    ax1.set_xticklabels(class_names_plot, rotation=45, ha='right')
                    ax1.set_ylim(0, 1.0)
                    ax1.set_ylabel('IoU')
                    ax1.set_title(f'Global IoU per Class (Epoch {epoch+1})\nTotal Samples: {total_samples}')
                    
                    # Right plot: IoU variance across processes
                    x = np.arange(num_classes)
                    width = 0.8 / world_size
                    
                    for rank in range(world_size):
                        if rank < len(process_metrics):
                            offset = (rank - world_size/2 + 0.5) * width
                            rank_iou = process_metrics[rank].get('iou', np.zeros(num_classes))
                            ax2.bar(x + offset, rank_iou, width, 
                                   label=f'Rank {rank} ({all_sample_counts[rank].int()} samples)', alpha=0.8)
                    
                    ax2.set_xticks(x)
                    ax2.set_xticklabels(class_names_plot, rotation=45, ha='right')
                    ax2.set_ylim(0, 1.0)
                    ax2.set_ylabel('IoU')
                    ax2.set_title(f'IoU per Class by Process (Epoch {epoch+1})')
                    ax2.legend()
                    
                    plt.tight_layout()
                    
                    # Save to TensorBoard
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150)
                    buf.seek(0)
                    comparison_img = Image.open(buf)
                    plt.close(fig)
                    self.writer.add_image('Charts/IoU_global_vs_processes', 
                                        np.array(comparison_img).transpose(2, 0, 1), epoch)
                    buf.close()
                
                # Print detailed results
                print(f"Validation Epoch: [{epoch + 1}] - Global metrics from {total_samples} samples across {world_size} processes")
                print(f"  Global Loss: {global_loss:.4f}")
                print(f"  Global Pixel Accuracy: {global_metrics['pixel_accuracy']:.4f}")
                print(f"  Global Mean Accuracy: {global_metrics['mean_accuracy']:.4f}")
                print(f"  Global Mean IoU: {global_metrics['mean_iou']:.4f}")
                print(f"  Global Freq Weighted IoU: {global_metrics['fw_iou']:.4f}")
                
                if len(process_miou) > 1:
                    print(f"  Process mIoU variance: std={torch.tensor(process_miou).std():.4f}, "
                          f"min={min(process_miou):.4f}, max={max(process_miou):.4f}")
                
                print("  Global IoU per class:")
                for i, iou in enumerate(global_metrics['iou']):
                    class_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
                    print(f"    {class_name}: {iou:.4f}")
                
                # Return global metrics for checkpoint saving
                return {
                    'val_loss': global_loss,
                    'pixel_accuracy': global_metrics['pixel_accuracy'],
                    'mean_accuracy': global_metrics['mean_accuracy'],
                    'mean_iou': global_metrics['mean_iou'],
                    'fw_iou': global_metrics['fw_iou']
                }
            else:
                # Non-main processes return dummy metrics (won't be used for checkpointing)
                return {
                    'val_loss': loss_meter.avg,
                    'pixel_accuracy': 0.0,
                    'mean_accuracy': 0.0,
                    'mean_iou': 0.0,
                    'fw_iou': 0.0
                }
        else:
            # Single process - compute metrics normally
            metrics = confusion_matrix.compute()
            
            if is_main_process() and self.writer:
                self.writer.add_scalar('Loss/val', loss_meter.avg, epoch)
                self.writer.add_scalar('Metrics/pixel_accuracy', metrics['pixel_accuracy'], epoch)
                self.writer.add_scalar('Metrics/mean_accuracy', metrics['mean_accuracy'], epoch)
                self.writer.add_scalar('Metrics/mean_iou', metrics['mean_iou'], epoch)
                self.writer.add_scalar('Metrics/fw_iou', metrics['fw_iou'], epoch)
                
                # Print detailed results
                print(f"Validation Epoch: [{epoch + 1}]")
                print(f"  Loss: {loss_meter.avg:.4f}")
                print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
                print(f"  Mean Accuracy: {metrics['mean_accuracy']:.4f}")
                print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
                print(f"  Freq Weighted IoU: {metrics['fw_iou']:.4f}")

                # Log per-class IoU
                print("  IoU per class:")
                class_names = self.config.data.class_names if hasattr(self.config.data, 'class_names') else None
                for i, iou in enumerate(metrics['iou']):
                    class_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
                    print(f"    {class_name}: {iou:.4f}")

                    self.writer.add_scalar(f'IoU_per_class/{class_name}', iou, epoch)
            
            return {
                'val_loss': loss_meter.avg,
                'pixel_accuracy': metrics['pixel_accuracy'],
                'mean_accuracy': metrics['mean_accuracy'],
                'mean_iou': metrics['mean_iou'],
                'fw_iou': metrics['fw_iou']
            }
    
    def log_segmentation_sample(self, image, prediction, target, epoch, tag):
        """
        Log segmentation sample to TensorBoard (only on main process).
        
        Args:
            image: Input image tensor [C, H, W]
            prediction: Prediction tensor [C, H, W]
            target: Target tensor [H, W]
            epoch: Current epoch
            tag: Sample tag
        """
        if not is_main_process() or not self.writer:
            return
            
        # Create a figure with 3 subplots: input, prediction, ground truth
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Denormalize image
        mean = torch.tensor(self.config.data.mean).view(3, 1, 1).to(image.device)
        std = torch.tensor(self.config.data.std).view(3, 1, 1).to(image.device)
        image_denorm = image * std + mean
        image_np = image_denorm.cpu().numpy().transpose(1, 2, 0)
        image_np = np.clip(image_np, 0, 1)
        
        # Get target mask
        target_mask = target.cpu().numpy()
        target_height, target_width = target_mask.shape

        # Convert to PIL Image for resizing
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        image_resized = image_pil.resize((target_width, target_height), Image.BILINEAR)
        image_np = np.array(image_resized).astype(np.float32) / 255.0
        
        # Get predicted class indices
        if prediction.dim() == 3:  # [C, H, W]
            predicted_mask = torch.argmax(prediction, dim=0).cpu().numpy()
        else:  # [H, W]
            predicted_mask = prediction.cpu().numpy()
        
        
        
        # Show original image
        ax1.imshow(image_np)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Create a colored prediction mask
        if self.class_colors is not None:
            # Create RGB prediction using class colors
            pred_colored = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3))
            for class_idx in range(len(self.class_colors)):
                class_mask = predicted_mask == class_idx
                pred_colored[class_mask] = self.class_colors[class_idx].cpu().numpy()
                
            # Show prediction as overlay
            ax2.imshow(image_np)
            ax2.imshow(pred_colored, alpha=0.5)
            ax2.set_title("Prediction Overlay")
            ax2.axis('off')
            
            # Create RGB ground truth using class colors
            target_colored = np.zeros((target_mask.shape[0], target_mask.shape[1], 3))
            for class_idx in range(len(self.class_colors)):
                class_mask = target_mask == class_idx
                target_colored[class_mask] = self.class_colors[class_idx].cpu().numpy()
                
            # Show ground truth as overlay
            ax3.imshow(image_np)
            ax3.imshow(target_colored, alpha=0.5)
            ax3.set_title("Ground Truth Overlay")
            ax3.axis('off')
        else:
            # Fallback if no color map is provided
            ax2.imshow(predicted_mask, cmap='nipy_spectral')
            ax2.set_title("Prediction")
            ax2.axis('off')
            
            ax3.imshow(target_mask, cmap='nipy_spectral')
            ax3.set_title("Ground Truth")
            ax3.axis('off')
        
        # Save plot to TensorBoard
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        sample_img = Image.open(buf)
        self.writer.add_image(f'Samples/{tag}', np.array(sample_img).transpose(2, 0, 1), epoch)
        plt.close(fig)
        buf.close()
        
        # Also add the segmentation prediction as image
        if self.class_colors is not None:
            self.writer.add_image(
                f'Masks/{tag}_pred', 
                torch.from_numpy(pred_colored.transpose(2, 0, 1)), 
                epoch
            )
            self.writer.add_image(
                f'Masks/{tag}_gt', 
                torch.from_numpy(target_colored.transpose(2, 0, 1)), 
                epoch
            )

    def save_checkpoint(self, epoch: int, is_best: bool = False, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Save a checkpoint of the model (only on main process).
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            metrics: Dictionary of validation metrics to save
        """
        if not is_main_process():
            return
            
        # Create checkpoint directory
        checkpoint_dir = self.output_dir / "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Get model state dict (handle DDP or DP)
        if hasattr(self.model, 'module'):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        # Create checkpoint with comprehensive information
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_miou': self.best_miou,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'config': self.config,
            'metrics_history': self.metrics_history,
            'use_amp': self.use_amp,
        }
        
        # Add scheduler state if available
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Add current metrics if provided
        if metrics:
            checkpoint['current_metrics'] = metrics

        # Format checkpoint path
        if False: ## Set to True to save all checkpoints for all epoches
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(checkpoint, checkpoint_path)
        # If this is the best model, create a copy
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model checkpoint to {best_path}")
            
            # Also save a text file with the best metrics for quick reference
            if metrics:
                best_metrics_path = self.output_dir / "best_metrics.txt"
                with open(best_metrics_path, 'w') as f:
                    f.write(f"Best model (epoch {epoch + 1}):\n")
                    for k, v in metrics.items():
                        f.write(f"{k}: {v}\n")
                    f.write(f"use_amp: {self.use_amp}\n")
        
        # Save latest checkpoint (for easy resuming)
        latest_path = checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)

class MultiDatasetTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        deep_supervision = self.config.model.decoder.deep_supervision
        no_object_weight = self.config.model.decoder.no_object_weight

        # loss weights
        class_weight = self.config.model.decoder.class_weight
        dice_weight = self.config.model.decoder.dice_weight
        mask_weight = self.config.model.decoder.mask_weight

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=self.config.model.decoder.train_num_points,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = self.config.model.decoder.dec_layers
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        
        losses = ["labels", "masks"]
        
        self.criterion = {}

        for t in self.config.tasks:
            self.criterion[t] = SetCriterion(
                self.config.num_class_dict[t],
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight[t],
                losses=losses,
                num_points=self.config.model.decoder.train_num_points,
                oversample_ratio=self.config.model.decoder.oversample_ratio,
                importance_sample_ratio=self.config.model.decoder.importance_sample_ratio,
            ).to(self.device)
        
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            # 'mean_iou': [],
            # 'pixel_accuracy': [],
            'learning_rate': []
        }

        for t in self.config.tasks:
            self.metrics_history[f'mean_iou_{t}'] = []
            self.metrics_history[f'pixel_accuracy_{t}'] = []
            
        # if self.config.enable_ema:
        if getattr(self.config, 'enable_ema', False):
            self.ema_model = EMA(model=self.model, decay=self.config.ema_decay, device=self.device, )
            if is_main_process():
                print("Using EMA weights for pseudo-labeling")
            if self.resume_checkpoint:
                checkpoint = torch.load(self.resume_checkpoint, map_location=self.device)
                self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
        else:
            self.ema_model = None
            
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        loss_meter = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        cuda_mem = AverageMeter()
        
        start = time.time()
        
        # Only show progress bar on main process
        if is_main_process():
            pbar = tqdm(self.train_loader)
            pbar.set_description(f'[TRAINING] Epoch {epoch + 1}')
        else:
            pbar = self.train_loader

        total_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(pbar):
            # Measure data loading time
            data_time.update(time.time() - start)
            # if batch_idx > 5:
            #     break
            # Get data
            if isinstance(batch, dict):
                inputs = batch['image'].to(self.device)
                targets = batch['target'].to(self.device)
            else:  # Assume tuple (inputs, targets)
                inputs, targets, ids = batch
                inputs = inputs.to(self.device)
                if not isinstance(targets, list) and not isinstance(targets, dict):
                    targets = targets.to(self.device)
                
                ### Temp workaround
                targets_temp = {self.config.tasks[0]: [], self.config.tasks[1]: []}
                for i, id in enumerate(ids):
                    targets_temp[self.config.tasks[id]].append(targets[i])
                targets = targets_temp
                ids = ids.to(self.device)
            # Zero gradients
            self.optimizer.zero_grad()
            
            if self.use_amp and self.scaler is not None:
                # Mixed precision training
                with autocast():
                    _, outputs = self.model(inputs)
                    if self.config.data.task_type == 'semantic_segmentation':
                        # outputs = torch.nn.functional.interpolate(
                        #     outputs, size=targets.shape[1:], 
                        #     mode="bilinear", align_corners=False
                        # )
                        # loss = self.criterion(outputs, targets)
                        raise NotImplementedError
                    elif self.config.data.task_type == 'instance_segmentation':
                        # losses = 0.0
                        losses = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
                        
                        # Dataset-sepcific loss with GT supervision
                        for t_id, t in enumerate(self.config.tasks):
                            # loss = self.criterion[t](outputs[t], targets[t])
                            
                            ### Temp workaround
                            outputs_t = outputs[t]
                            outputs_temp = {'pred_logits': outputs_t['pred_logits'][ids == t_id], 'pred_masks': outputs_t['pred_masks'][ids == t_id] }
                            outputs_temp['aux_outputs'] = []
                            for aux_i in outputs_t['aux_outputs']:
                                outputs_temp['aux_outputs'].append({"pred_logits": aux_i['pred_logits'][ids == t_id], "pred_masks": aux_i['pred_masks'][ids == t_id]})
                            if len(targets[t]) == 0:
                                # loss = 0.0
                                loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
                            else:
                                loss = self.criterion[t](outputs_temp, targets[t])

                            if isinstance(loss, dict):
                                for k in list(loss.keys()):
                                    if k in self.criterion[t].weight_dict:
                                        loss[k] *= self.criterion[t].weight_dict[k]
                                    else:
                                        # remove this loss if not specified in `weight_dict`
                                        loss.pop(k)
                                loss_dict = loss.copy()
                                loss = sum(loss_dict.values())
                            losses = losses + loss
                # Scale loss to prevent gradient underflow
                scaled_loss = self.scaler.scale(losses)
                scaled_loss.backward()
                
                # Gradient clipping (if configured)
                if hasattr(self.config.training, 'grad_clip_val') and self.config.training.grad_clip_val:
                    # Unscale before clipping to get true gradient norms
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.grad_clip_val
                    )
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                _, outputs = self.model(inputs)
                if self.config.data.task_type == 'semantic_segmentation':
                    # outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[1:], mode="bilinear", align_corners=False)
                    # loss = self.criterion(outputs, targets)
                    raise NotImplementedError
                elif self.config.data.task_type == 'instance_segmentation':
                    # losses = 0.0
                    losses = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
                    for t_id, t in enumerate(self.config.tasks):
                        ### Temp workaround
                        outputs_t = outputs[t]
                        outputs_temp = {'pred_logits': outputs_t['pred_logits'][ids == t_id], 'pred_masks': outputs_t['pred_masks'][ids == t_id] }
                        outputs_temp['aux_outputs'] = []
                        for aux_i in outputs_t['aux_outputs']:
                            outputs_temp['aux_outputs'].append({"pred_logits": aux_i['pred_logits'][ids == t_id], "pred_masks": aux_i['pred_masks'][ids == t_id]})
                        if len(targets[t]) == 0:
                            # loss = 0.0
                            loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
                        else:
                            loss = self.criterion[t](outputs_temp, targets[t])

                        if isinstance(loss, dict):
                            for k in list(loss.keys()):
                                if k in self.criterion[t].weight_dict:
                                    loss[k] *= self.criterion[t].weight_dict[k]
                                else:
                                    # remove this loss if not specified in `weight_dict`
                                    loss.pop(k)
                            loss_dict = loss.copy()
                            loss = sum(loss_dict.values())
                        losses = losses + loss

                losses.backward()
                
                # Gradient clipping if configured
                if hasattr(self.config.training, 'grad_clip_val') and self.config.training.grad_clip_val:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.grad_clip_val
                    )
                
                # Update weights
                self.optimizer.step()
            
            self.ema_model.update(self.model.module if is_distributed() else self.model)
            
            # For testing the output of EMA
            if is_main_process():
                ## Class-agnostic mask visualization
                if True:
                    self.model.eval()
                    with torch.no_grad():
                        outputs_superset, _ = self.model(inputs)
                        masks_sup = outputs_superset["pred_masks"]
                        binary_mask_for_vis = masks_sup[0].sigmoid()
                        num_queries = binary_mask_for_vis.size(0)
                        nrow = int(math.ceil(math.sqrt(num_queries)))
                        grid_vis = make_grid(
                            binary_mask_for_vis.unsqueeze(1), nrow=nrow, padding=2
                        )
                        # grid_vis: [1, H_total, W_total]
                        # grid_vis_np = grid_vis.cpu().numpy()  # [H_total, W_total], float32 in [0,1]
                        # print(grid_vis_np.shape)
                        # grid_vis_np = (grid_vis_np * 255).astype(np.uint8)  # Convert to uint8
                        # Convert to 3-channel grayscale for cv2.imshow
                        img = cv2.cvtColor(np.array(torchvision.transforms.ToPILImage()(grid_vis)), cv2.COLOR_RGB2BGR)
                        # img.show()
                        # grid_vis_np = cv2.cvtColor(grid_vis_np, cv2.COLOR_GRAY2BGR)
                        cv2.imshow("Class-agnostic masks", img)
                        k = cv2.waitKey(1) & 0xFF        # non-blocking
                        if k in (27, ord('q')):          # ESC or q to quit
                            break
                if True:
                    with self.ema_model.apply_to(self.model.module if is_distributed() else self.model):
                        self.model.eval()
                        with torch.no_grad():
                            outputs_superset_ema, _ = self.model(inputs)
                            logits_sup = outputs_superset_ema["pred_logits"]
                            masks_sup = outputs_superset_ema["pred_masks"]
                            masks_sup = F.interpolate(
                                masks_sup,
                                size=inputs.shape[2:],  # (H, W)
                                mode="bilinear",
                                align_corners=False,
                            )
                            if is_distributed():
                                # outputs_map = self.model.module.decoder.semantic_inference(logits_sup[0], masks_sup[0])
                                outputs_map = self.model.module.decoder.semantic_inference(logits_sup[0], masks_sup[0], False)
                            else:
                                # outputs_map = self.model.decoder.semantic_inference(logits_sup[0], masks_sup[0])
                                outputs_map = self.model.decoder.semantic_inference(logits_sup[0], masks_sup[0], False)
                            # print(outputs_map.shape)
                            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                            fig.suptitle("Unified Prediction", fontsize=16)
                            mean = torch.tensor(self.config.data.mean).view(3, 1, 1).to(inputs.device)
                            std = torch.tensor(self.config.data.std).view(3, 1, 1).to(inputs.device)
                            image_denorm = inputs[0] * std + mean
                            image_np = image_denorm.cpu().numpy().transpose(1, 2, 0)
                            image_np = np.clip(image_np, 0, 1)
                            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
                            # image_resized = image_pil.resize(img.size, Image.BILINEAR)
                            image_np = np.array(image_pil).astype(np.float32) / 255.0
                            if outputs_map.dim() == 3:  # [C, H, W]
                                predicted_mask = torch.argmax(outputs_map, dim=0).cpu().numpy()
                            else:  # [H, W]
                                predicted_mask = outputs_map.cpu().numpy()
                            ax1.imshow(image_np)
                            # print(self.config.tasks[ids[0].item()])
                            ax1.set_title("Original Image")
                            ax1.axis('off')

                            unified_label_colors = self.config.data.get_manual_superset_color_and_names()['sup_colors']
                            unified_class_names = self.config.data.get_manual_superset_color_and_names()['sup_names']
                            # Find duplicates
                            color_to_indices = {}
                            for idx, color in enumerate(unified_label_colors):
                                color_to_indices.setdefault(color, []).append(idx)

                            class_colors = torch.tensor(unified_label_colors, dtype=torch.float32) / 255.0

                            if outputs_map.dim() == 3:  # [C, H, W]
                                predicted_mask = torch.argmax(outputs_map, dim=0).cpu().numpy()
                            else:  # [H, W]
                                predicted_mask = outputs_map.cpu().numpy()

                            pred_colored = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3))
                            for class_idx in range(len(class_colors)):
                                class_mask = predicted_mask == class_idx
                                pred_colored[class_mask] = class_colors[class_idx].cpu().numpy()

                            ax2.imshow(image_np)
                            ax2.imshow(pred_colored, alpha=0.5)
                            ax2.set_title("Prediction Overlay")
                            ax2.axis('off')

                            ax3.imshow(pred_colored)
                            ax3.set_title("Predicted Mask")
                            ax3.axis('off')

                            # Legend as a fourth subplot in the bottom row

                            # Change layout to 2 rows: 1st row for images, 2nd row for legend (spanning all columns)
                            fig.clf()  # Clear the current figure to redefine the layout
                            gs = fig.add_gridspec(2, 3, height_ratios=[10, 1])

                            # Redraw the three images in the first row
                            ax1 = fig.add_subplot(gs[0, 0])
                            ax1.imshow(image_np)
                            ax1.set_title("Original Image {name}".format(name=self.config.tasks[ids[0].item()]))
                            ax1.axis('off')

                            ax2 = fig.add_subplot(gs[0, 1])
                            ax2.imshow(image_np)
                            ax2.imshow(pred_colored, alpha=0.5)
                            ax2.set_title("Prediction Overlay {name}".format(name=self.config.tasks[ids[0].item()]))
                            ax2.axis('off')

                            ax3 = fig.add_subplot(gs[0, 2])
                            ax3.imshow(pred_colored)
                            ax3.set_title("Predicted Mask {name}".format(name=self.config.tasks[ids[0].item()]))
                            ax3.axis('off')

                            # Legend in the bottom row, spanning all columns
                            # Move the legend down by increasing the bottom margin and adjusting subplot spacing
                            ax4 = fig.add_subplot(gs[1, :])
                            for idx, (name, color) in enumerate(zip(unified_class_names, unified_label_colors)):
                                ax4.bar(idx, 1, color=np.array(color)/255, edgecolor='k', width=1)
                            ax4.set_xticks(range(len(unified_class_names)))
                            ax4.set_xticklabels(unified_class_names, rotation=90, ha='center', fontsize=10)
                            ax4.set_yticks([])
                            ax4.set_title('Legend', fontsize=12)
                            ax4.set_xlim(-0.5, len(unified_class_names)-0.5)
                            ax4.tick_params(axis='x', length=0)
                            # plt.tight_layout(rect=[0, 0.08, 1, 1])  # Increase bottom margin (0.08)
                            plt.tight_layout()
                            plt.subplots_adjust(hspace=0.2)         # Increase vertical space between rows
                            # plt.show()
                            # Render figure to RGB array
                            canvas = FigureCanvas(fig)
                            canvas.draw()
                            w, h = fig.canvas.get_width_height()
                            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)

                            plt.close(fig)
                            # Show in OpenCV (convert RGB→BGR)
                            cv2.imshow("EMA Prediction on Supersets", cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
                            k = cv2.waitKey(1) & 0xFF        # non-blocking
                            if k in (27, ord('q')):          # ESC or q to quit
                                break
                            # cv2.destroyAllWindows()
                            
                            
                    self.model.train()
            
            loss = losses
            # Update metrics
            loss_meter.update(loss.item(), inputs.size(0))
            batch_time.update(time.time() - start)
            if torch.cuda.is_available():
                cuda_mem.update(torch.cuda.max_memory_allocated(device=None) / (1024 * 1024 * 1024))
            
            # Log batch-level metrics only from main process (less frequently)
            if is_main_process() and self.writer and batch_idx % 10 == 0:
                global_step = epoch * total_batches + batch_idx
                self.writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_Rate', current_lr, global_step)

                if self.use_amp and self.scaler is not None:
                    self.writer.add_scalar('AMP/loss_scale', self.scaler.get_scale(), global_step)
            # Update progress bar only on main process
            if is_main_process():
                monitor = {
                    'Loss': f'{loss_meter.val:.4f} ({loss_meter.avg:.4f})',
                    'CUDA': f'{cuda_mem.val:.2f} ({cuda_mem.avg:.2f})'
                }
                if self.use_amp and self.scaler is not None:
                    monitor['Scale'] = f'{self.scaler.get_scale():.0f}'
                pbar.set_postfix(monitor)

            start = time.time()
        
        # Log epoch-level metrics from all processes
        if self.writer:
            if hasattr(self, 'rank_prefix'):
                self.writer.add_scalar(f'{self.rank_prefix}Loss/train_epoch', loss_meter.avg, epoch)
                self.writer.add_scalar(f'{self.rank_prefix}Memory/avg_cuda_memory_gb', cuda_mem.avg, epoch)
            else:
                self.writer.add_scalar('Loss/train_epoch', loss_meter.avg, epoch)
                self.writer.add_scalar('Memory/avg_cuda_memory_gb', cuda_mem.avg, epoch)
        
        # Gather metrics from all processes
        if is_distributed():
            # Prepare metrics to gather
            process_metrics = torch.tensor([
                loss_meter.avg,
                cuda_mem.avg,
                batch_time.avg,
                data_time.avg,
                loss_meter.sum,   # Total loss across all batches
                loss_meter.count  # Total samples processed
            ], device=self.device)
            
            # Gather metrics from all processes to rank 0
            gathered_metrics = gather_tensor(process_metrics)
            
            if is_main_process():
                # Extract gathered metrics
                world_size = get_world_size()
                all_avg_losses = gathered_metrics[0::6]  # Every 6th element starting from 0
                all_cuda_mem = gathered_metrics[1::6]
                all_batch_times = gathered_metrics[2::6]
                all_data_times = gathered_metrics[3::6]
                all_total_losses = gathered_metrics[4::6]
                all_sample_counts = gathered_metrics[5::6]
                
                # Compute global weighted average loss
                total_samples = all_sample_counts.sum().item()
                global_loss = all_total_losses.sum().item() / total_samples if total_samples > 0 else 0
                
                # Log individual process metrics
                for rank in range(world_size):
                    self.writer.add_scalar(f'Loss/train_rank_{rank}', all_avg_losses[rank].item(), epoch)
                    self.writer.add_scalar(f'Memory/cuda_memory_rank_{rank}', all_cuda_mem[rank].item(), epoch)
                    self.writer.add_scalar(f'Timing/batch_time_rank_{rank}', all_batch_times[rank].item(), epoch)
                    self.writer.add_scalar(f'Timing/data_time_rank_{rank}', all_data_times[rank].item(), epoch)
                    self.writer.add_scalar(f'Samples/train_count_rank_{rank}', all_sample_counts[rank].item(), epoch)
                
                # Log aggregated metrics
                self.writer.add_scalar('Loss/train_global', global_loss, epoch)
                self.writer.add_scalar('Loss/train_avg_across_ranks', all_avg_losses.mean().item(), epoch)
                self.writer.add_scalar('Loss/train_min', all_avg_losses.min().item(), epoch)
                self.writer.add_scalar('Loss/train_max', all_avg_losses.max().item(), epoch)
                self.writer.add_scalar('Loss/train_std', all_avg_losses.std().item(), epoch)
                
                self.writer.add_scalar('Memory/cuda_memory_avg', all_cuda_mem.mean().item(), epoch)
                self.writer.add_scalar('Memory/cuda_memory_max', all_cuda_mem.max().item(), epoch)
                
                self.writer.add_scalar('Timing/batch_time_avg', all_batch_times.mean().item(), epoch)
                self.writer.add_scalar('Timing/data_time_avg', all_data_times.mean().item(), epoch)
                
                try:
                    # Log histograms for better visualization
                    self.writer.add_histogram('Distributions/loss_per_rank', all_avg_losses.cpu(), epoch)
                    self.writer.add_histogram('Distributions/memory_per_rank', all_cuda_mem.cpu(), epoch)
                except Exception as e: 
                    print(e)
                return {'train_loss': global_loss}
            else:
                return {'train_loss': loss_meter.avg}
        else:
            # Single process
            if is_main_process() and self.writer:
                self.writer.add_scalar('Loss/train', loss_meter.avg, epoch)
                self.writer.add_scalar('Memory/cuda_memory', cuda_mem.avg, epoch)
            
            return {'train_loss': loss_meter.avg}

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        loss_meter = AverageMeter()
        cuda_mem = AverageMeter()

        # Initialize confusion matrix for IoU calculation
        confusion_matrix = {}
        for t in self.config.tasks:
            confusion_matrix[t] = ConfusionMatrix(self.config.num_class_dict[t], self.config.data.ignore_index)
        # confusion_matrix = ConfusionMatrix(self.config.data.num_classes, self.config.data.ignore_index)
        
        # Create storage for visualizing a few samples
        num_samples_to_visualize = min(4, self.config.training.batch_size)
        # samples_visualized = 0
        samples_visualized = {self.config.tasks[0]: 0, self.config.tasks[1]: 0}

        with torch.no_grad():
            # Only show progress bar on main process
            if is_main_process():
                pbar = tqdm(self.val_loader)
                pbar.set_description(f'[VALIDATION] Epoch {epoch + 1}')
            else:
                pbar = self.val_loader

            for batch_idx, batch in enumerate(pbar):
                # Get data
                if isinstance(batch, dict):
                    inputs = batch['image'].to(self.device)
                    targets = batch['target'].to(self.device)
                else:  # Assume tuple (inputs, targets)
                    inputs, targets, ids = batch
                    inputs = inputs.to(self.device)
                    if not isinstance(targets, list) and not isinstance(targets, dict):
                        targets = targets.to(self.device)
                    ### Temp workaround
                    targets_temp = {self.config.tasks[0]: [], self.config.tasks[1]: []}
                    for i, id in enumerate(ids):
                        targets_temp[self.config.tasks[id]].append(targets[i])
                    targets = targets_temp
                    ids = ids.to(self.device)

                # print(f"Rank: {get_rank()}: ids: {ids}, targets bdd: {len(targets[self.config.tasks[0]])}, targets bdd: {len(targets[self.config.tasks[1]])}")
                
                # Run forward pass
                _, outputs = self.model(inputs)
                # print(targets)
                if self.config.data.task_type == 'semantic_segmentation':
                    # outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[1:], mode="bilinear", align_corners=False)
                    # loss = self.criterion_eval(outputs, targets)
                    raise NotImplementedError
                elif self.config.data.task_type == 'instance_segmentation':
                    losses = 0.0
                    outputs_for_eval = {}
                    for t_id, t in enumerate(self.config.tasks):
                        # loss = self.criterion[t](outputs[t], targets[t])
                        ### Temp workaround
                        outputs_t = outputs[t]
                        outputs_temp = {'pred_logits': outputs_t['pred_logits'][ids == t_id], 'pred_masks': outputs_t['pred_masks'][ids == t_id] }
                        outputs_temp['aux_outputs'] = []
                        for aux_i in outputs_t['aux_outputs']:
                            outputs_temp['aux_outputs'].append({"pred_logits": aux_i['pred_logits'][ids == t_id], "pred_masks": aux_i['pred_masks'][ids == t_id]})
                        outputs_for_eval[t] = outputs_temp
                        
                        if len(targets[t]) == 0:
                            loss = 0.0
                        else:
                            loss = self.criterion[t](outputs_temp, targets[t])

                        if isinstance(loss, dict):
                            for k in list(loss.keys()):
                                if k in self.criterion[t].weight_dict:
                                    loss[k] *= self.criterion[t].weight_dict[k]
                                else:
                                    # remove this loss if not specified in `weight_dict`
                                    loss.pop(k)
                            loss_dict = loss.copy()
                            loss = sum(loss_dict.values())
                        losses += loss
                    outputs = outputs_for_eval
                    loss = losses
                # Update metrics
                loss_meter.update(loss.item(), inputs.size(0))
                if torch.cuda.is_available():
                    cuda_mem.update(torch.cuda.max_memory_allocated(device=None) / (1024 * 1024 * 1024))
                
                # Update confusion matrix for IoU calculation
                outputs_for_eval = {}
                targets_for_eval = {}
                if isinstance(outputs, dict) and isinstance(targets, dict):
                    ## process mask2former head outputs and targets into [B, C, H, W] and [B, H, W]
                    for t in self.config.tasks:
                        if len(targets[t]) != 0:
                            target_hw = targets[t][0]['masks'].shape[1:]
                            mask_cls_results = outputs[t]["pred_logits"]
                            mask_pred_results = outputs[t]["pred_masks"]
                            # print("mask_cls_results:", mask_cls_results.shape)
                            # print("mask_pred_results", mask_pred_results.shape)
                            # print("target_hw: ", target_hw)
                            # upsample masks
                            mask_pred_results = F.interpolate(
                                mask_pred_results,
                                size=target_hw,
                                mode="bilinear",
                                align_corners=False,
                            )

                            outputs_map = []
                            for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):
                                # semantic segmentation inference
                                if is_distributed():
                                    # semmap = self.model.module.decoder.semantic_inference(mask_cls_result, mask_pred_result)
                                    semmap = self.model.module.decoder.semantic_inference(mask_cls_result, mask_pred_result, False)
                                else:
                                    # semmap = self.model.decoder.semantic_inference(mask_cls_result, mask_pred_result)
                                    semmap = self.model.decoder.semantic_inference(mask_cls_result, mask_pred_result, False)
                                outputs_map.append(semmap)
                            outputs_map = torch.stack(outputs_map)
                            targets_map = self.val_loader.dataset.instances_to_semantic(targets=targets[t], ignore_index=self.config.data.ignore_index)
                        
                            confusion_matrix[t].update(outputs_map, targets_map)

                            # print(f"Rank: {get_rank()}: Task: {t}, outputs_map: {outputs_map.shape}, targets_map: {targets_map.shape}")
                            # print(t)
                            # print(outputs_map.shape)
                            # print(targets_map.shape)

                            outputs_for_eval[t] = outputs_map
                            targets_for_eval[t] = targets_map
                            
                    outputs = outputs_for_eval
                    targets = targets_for_eval
                    
                # Update progress bar only on main process
                if is_main_process():
                    monitor = {
                        'Loss': f'{loss_meter.val:.4f} ({loss_meter.avg:.4f})',
                        'CUDA': f'{cuda_mem.val:.2f} ({cuda_mem.avg:.2f})'
                    }
                    pbar.set_postfix(monitor)

                # Visualize samples only on main process
                if is_main_process() and self.writer:
                    # try:
                    # if samples_visualized < num_samples_to_visualize:
                        # sample_idx = np.random.randint(0, inputs.shape[0])
                    for t_id, t in enumerate(self.config.tasks):
                        # if len(targets[t]) != 0:
                        if samples_visualized[t] < num_samples_to_visualize:
                            if t in targets.keys():
                                inputs_t = inputs[ids == t_id]
                                sample_idx = np.random.randint(0, inputs_t.shape[0])
                                label_colors_list, _ = self.config.data.parse_color_and_names(t)
                                self.log_segmentation_sample(
                                    inputs_t[sample_idx], 
                                    outputs[t][sample_idx], 
                                    targets[t][sample_idx],
                                    epoch, 
                                    f"sample_{t}_{samples_visualized[t]}",
                                    torch.tensor(label_colors_list, dtype=torch.float32) / 255.0,
                                )
                                samples_visualized[t] += 1
                        # samples_visualized += 1
                    # except Exception as e: 
                    #     # print(e)
                    #     print(f"[WARNING] Fail to log image to tensorboard")
                    #     print(e)

        # Gather metrics from all processes
        if is_distributed():
            # print(f"Rank: {get_rank()} finish epoch...", flush=True)
            # dist.barrier()
            synchronize()
            # Gather basic loss and memory stats
            process_stats = torch.tensor([
                loss_meter.avg,
                cuda_mem.avg,
                loss_meter.sum,   # Total loss
                loss_meter.count  # Total samples
            ], device=self.device, dtype=torch.float32)
            
            # if process_stats.is_cuda:
            #     torch.cuda.synchronize(process_stats.device)
            
            # Force the WORLD group explicitly
            # group = dist.group.WORLD
            # print(f"[global {get_rank()}] group_rank={dist.get_rank(group)} "
            #                 f"group_world={dist.get_world_size(group)}", flush=True)

            # bufs = [torch.empty_like(process_stats) for _ in range(get_world_size())]
            # dist.all_gather(bufs, process_stats, group=group)
            # gathered_stats = torch.stack(bufs, dim=0)

            # print(f"[global {get_rank()}] gathered shape {tuple(gathered_stats.shape)}", flush=True)
            # if get_rank() == 0:
            #     print(gathered_stats, flush=True)

            # print(f"Rank: {get_rank()} before process_stats {process_stats}", flush=True)
            gathered_stats = gather_tensor(process_stats)
            # print(f"Rank: {get_rank()} after gathered_stats {gathered_stats}", flush=True)
            # # print(gathered_stats.shape)

            # if is_main_process():
            #     print(gathered_stats, flush=True)

            # dist.barrier()

            # gathered_cm = {
            #     self.config.tasks[0]: torch.zeros((get_world_size(), self.config.num_class_dict[self.config.tasks[0]], self.config.num_class_dict[self.config.tasks[0]])),
            #     self.config.tasks[1]: torch.zeros((get_world_size(), self.config.num_class_dict[self.config.tasks[1]], self.config.num_class_dict[self.config.tasks[1]])),
            # }

            # Gather confusion matrices to CPU (root-only) and compute global/per-rank metrics on rank 0
            # gathered_cm = {}
            # for t in self.config.tasks:
            #     num_classes = self.config.num_class_dict[t]

            #     # Always make a (C, C) CPU numpy array, even if this rank saw no samples
            #     if confusion_matrix[t].mat is not None:
            #         cm_np = confusion_matrix[t].mat.astype(np.int64, copy=False)
            #     else:
            #         cm_np = np.zeros((num_classes, num_classes), dtype=np.int64)

            #     # Root only will receive a list; others pass None
            #     obj_list = [None for _ in range(get_world_size())] if is_main_process() else None

            #     # CPU object gather to rank 0 (works regardless of NCCL/Gloo)
            #     dist.gather_object(cm_np, obj_list, dst=0)

            #     # On rank 0: stack into [world_size, C, C] tensor for later use
            #     if is_main_process():
            #         gathered_cm[t] = torch.stack(
            #             [torch.as_tensor(a, dtype=torch.float64) for a in obj_list], dim=0
            #         )  # shape: [ws, C, C]
            

            ## This stuck
            ## Gather and sum confusion matrices for accurate global metrics
            gathered_cm = {}

            # task_groups = {}
            # for t in self.config.tasks:
            #     # This assumes all ranks participate in all tasks
            #     task_groups[t] = dist.new_group()

            for t in self.config.tasks:
                # dist.barrier()
                synchronize()
                # print(f"Rank: {get_rank()} inside for t in self.config.tasks")
                if confusion_matrix[t].mat is not None:
                    cm_tensor = torch.tensor(confusion_matrix[t].mat, dtype=torch.float32, device=self.device)
                else:
                    # Handle case where no samples were processed
                    cm_tensor = torch.zeros((self.config.num_class_dict[t], self.config.num_class_dict[t]), 
                                        dtype=torch.float32, device=self.device)
                # print(f"Rank {get_rank()} at task {t}: cm_tensor.shape: {cm_tensor.shape}")
                # gathered_cm_t = gather_tensor_stack(cm_tensor, group=task_groups[t])
                # gathered_cm_t = gather_variable_tensor(cm_tensor)
                gathered_cm_t = gather_tensor(cm_tensor)
                gathered_cm[t] = gathered_cm_t


            if is_main_process():
                world_size = get_world_size()
                # print(f"line 2226: {world_size}")
                # Extract individual process stats
                all_avg_losses = gathered_stats[0::4]
                all_cuda_mem = gathered_stats[1::4]
                all_total_losses = gathered_stats[2::4]
                all_sample_counts = gathered_stats[3::4]
                
                process_metrics_task = {}

                global_metrics_task = {}
                for t in self.config.tasks:
                    # Sum confusion matrices to get global confusion matrix
                    num_classes = self.config.num_class_dict[t]
                    global_cm = gathered_cm[t].view(world_size, num_classes, num_classes).sum(dim=0)
                    
                    # Compute global metrics from the summed confusion matrix
                    confusion_matrix[t].mat = global_cm.cpu().numpy().astype(np.int64)
                    global_metrics_task[t] = confusion_matrix[t].compute()

                    process_metrics = []
                    for rank in range(world_size):
                        rank_cm = gathered_cm[t].view(world_size, num_classes, num_classes)[rank].cpu().numpy().astype(np.int64)
                        rank_cm_obj = ConfusionMatrix(num_classes, self.config.data.ignore_index)
                        rank_cm_obj.mat = rank_cm
                        if rank_cm.sum() > 0:  # Only compute if there are samples
                            rank_metrics = rank_cm_obj.compute()
                            process_metrics.append(rank_metrics)
                        else:
                            # Handle empty confusion matrix
                            process_metrics.append({
                                'pixel_accuracy': 0.0,
                                'mean_iou': 0.0,
                                'mean_accuracy': 0.0,
                                'fw_iou': 0.0,
                                'iou': np.zeros(num_classes)
                            })
                    process_metrics_task[t] = process_metrics

                # Compute global average loss (weighted by sample count)
                total_samples = all_sample_counts.sum().item()
                global_loss = all_total_losses.sum().item() / total_samples if total_samples > 0 else 0
                
                # Log individual process metrics
                for rank in range(world_size):
                    self.writer.add_scalar(f'Loss/val_rank_{rank}', all_avg_losses[rank].item(), epoch)
                    self.writer.add_scalar(f'Memory/val_cuda_memory_rank_{rank}', all_cuda_mem[rank].item(), epoch)

                    for t in self.config.tasks:
                        self.writer.add_scalar(f'Metrics/pixel_accuracy_rank_{rank}_task_{t}', process_metrics_task[t][rank]['pixel_accuracy'], epoch)
                        self.writer.add_scalar(f'Metrics/mean_iou_rank_{rank}_task_{t}', process_metrics_task[t][rank]['mean_iou'], epoch)
                    self.writer.add_scalar(f'Samples/val_count_rank_{rank}', all_sample_counts[rank].item(), epoch)
                
                # Log global (accurate) metrics
                self.writer.add_scalar('Loss/val_global', global_loss, epoch)
                
                try:
                    self.writer.add_histogram('Distributions/val_loss_per_rank', all_avg_losses.cpu(), epoch)
                except Exception as e: 
                    print(e)
                print(f"Validation Epoch: [{epoch + 1}] - Global metrics from {total_samples} samples across {world_size} processes")
                print(f"  Global Loss: {global_loss:.4f}")
                for t in self.config.tasks:
                    global_metrics = global_metrics_task[t]
                    self.writer.add_scalar(f'Metrics/pixel_accuracy_global_task_{t}', global_metrics['pixel_accuracy'], epoch)
                    self.writer.add_scalar(f'Metrics/mean_accuracy_global_task_{t}', global_metrics['mean_accuracy'], epoch)
                    self.writer.add_scalar(f'Metrics/mean_iou_global_task_{t}', global_metrics['mean_iou'], epoch)
                    self.writer.add_scalar(f'Metrics/fw_iou_global_task_{t}', global_metrics['fw_iou'], epoch)
                
                    # Log variance across processes
                    process_metrics = process_metrics_task[t]

                    process_miou = [m['mean_iou'] for m in process_metrics]
                    process_pixel_acc = [m['pixel_accuracy'] for m in process_metrics]
                
                    if len(process_miou) > 1:
                        miou_tensor = torch.tensor(process_miou)
                        pixel_acc_tensor = torch.tensor(process_pixel_acc)
                        
                        self.writer.add_scalar(f'Metrics/mean_iou_std_task_{t}', miou_tensor.std().item(), epoch)
                        self.writer.add_scalar(f'Metrics/mean_iou_min_task_{t}', miou_tensor.min().item(), epoch)
                        self.writer.add_scalar(f'Metrics/mean_iou_max_task_{t}', miou_tensor.max().item(), epoch)
                        
                        # Log histograms
                        try:
                            # self.writer.add_histogram('Distributions/val_loss_per_rank', all_avg_losses.cpu(), epoch)
                            self.writer.add_histogram(f'Distributions/val_miou_per_rank_task_{t}', miou_tensor, epoch)
                            self.writer.add_histogram(f'Distributions/val_pixel_acc_per_rank_task_{t}', pixel_acc_tensor, epoch)
                        except Exception as e: 
                            print(e)
                    # Log global per-class IoU
                    _, class_names = self.config.data.parse_color_and_names(t)
                    # class_names = self.config.data.class_names if hasattr(self.config.data, 'class_names') else None
                    global_metrics = global_metrics_task[t]
                    for i, iou in enumerate(global_metrics['iou']):
                        class_name = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
                        self.writer.add_scalar(f'IoU_per_class_global_task_{t}/{class_name}', iou, epoch)
                
                    # Create visualization comparing global vs per-process IoU
                    if len(process_metrics) > 1:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
                        
                        # Left plot: Global IoU per class
                        num_classes = len(global_metrics['iou'])
                        class_names_plot = class_names if class_names else [f"Class {i}" for i in range(num_classes)]
                        
                        ax1.bar(np.arange(num_classes), global_metrics['iou'])
                        ax1.set_xticks(np.arange(num_classes))
                        ax1.set_xticklabels(class_names_plot, rotation=45, ha='right')
                        ax1.set_ylim(0, 1.0)
                        ax1.set_ylabel('IoU')
                        ax1.set_title(f'Task: {t} - Global IoU per Class (Epoch {epoch+1})\nTotal Samples: {total_samples}')
                        
                        # Right plot: IoU variance across processes
                        x = np.arange(num_classes)
                        width = 0.8 / world_size
                        
                        for rank in range(world_size):
                            if rank < len(process_metrics):
                                offset = (rank - world_size/2 + 0.5) * width
                                rank_iou = process_metrics[rank].get('iou', np.zeros(num_classes))
                                ax2.bar(x + offset, rank_iou, width, 
                                    label=f'Rank {rank} ({all_sample_counts[rank].int()} samples)', alpha=0.8)
                        
                        ax2.set_xticks(x)
                        ax2.set_xticklabels(class_names_plot, rotation=45, ha='right')
                        ax2.set_ylim(0, 1.0)
                        ax2.set_ylabel('IoU')
                        ax2.set_title(f'Task: {t} - IoU per Class by Process (Epoch {epoch+1})')
                        ax2.legend()
                        
                        plt.tight_layout()
                        
                        # Save to TensorBoard
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=150)
                        buf.seek(0)
                        comparison_img = Image.open(buf)
                        plt.close(fig)
                        self.writer.add_image(f'Charts/IoU_global_vs_processes_task_{t}', 
                                            np.array(comparison_img).transpose(2, 0, 1), epoch)
                        buf.close()
                
                    # Print detailed results
                    
                    print(f"  Task {t} - Global Pixel Accuracy: {global_metrics['pixel_accuracy']:.4f}")
                    print(f"  Task {t} - Global Mean Accuracy: {global_metrics['mean_accuracy']:.4f}")
                    print(f"  Task {t} - Global Mean IoU: {global_metrics['mean_iou']:.4f}")
                    print(f"  Task {t} - Global Freq Weighted IoU: {global_metrics['fw_iou']:.4f}")
                    
                    if len(process_miou) > 1:
                        print(f"  Task {t} - Process mIoU variance: std={torch.tensor(process_miou).std():.4f}, "
                            f"min={min(process_miou):.4f}, max={max(process_miou):.4f}")
                    
                    print(f"  Task {t} - Global IoU per class:")
                    for i, iou in enumerate(global_metrics['iou']):
                        class_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
                        print(f"    {class_name}: {iou:.4f}")
                
                output_metrics = {'val_loss': global_loss}

                for t in self.config.tasks:
                    output_metrics[f'pixel_accuracy_{t}'] = global_metrics_task[t]['pixel_accuracy']
                    output_metrics[f'mean_accuracy_{t}'] = global_metrics_task[t]['mean_accuracy']
                    output_metrics[f'mean_iou_{t}'] = global_metrics_task[t]['mean_iou']
                    output_metrics[f'fw_iou_{t}'] = global_metrics_task[t]['fw_iou']
                # Return global metrics for checkpoint saving
                return output_metrics
            else:
                # Non-main processes return dummy metrics (won't be used for checkpointing)
                return {
                    'val_loss': loss_meter.avg,
                    'pixel_accuracy': 0.0,
                    'mean_accuracy': 0.0,
                    'mean_iou': 0.0,
                    'fw_iou': 0.0
                }
        else:
            # Single process - compute metrics normally
            output_metrics = {'val_loss': loss_meter.avg}
            if is_main_process() and self.writer:
                self.writer.add_scalar('Loss/val', loss_meter.avg, epoch)
                print(f"Validation Epoch: [{epoch + 1}]")
                print(f"  Loss: {loss_meter.avg:.4f}")
            for t in self.config.tasks:
                metrics = confusion_matrix[t].compute()
            
                if is_main_process() and self.writer:
                    self.writer.add_scalar(f'Metrics/pixel_accuracy_task_{t}', metrics['pixel_accuracy'], epoch)
                    self.writer.add_scalar(f'Metrics/mean_accuracy_task_{t}', metrics['mean_accuracy'], epoch)
                    self.writer.add_scalar(f'Metrics/mean_iou_task_{t}', metrics['mean_iou'], epoch)
                    self.writer.add_scalar(f'Metrics/fw_iou_task_{t}', metrics['fw_iou'], epoch)
                    
                    # Print detailed results
                    print(f"  Task {t} - Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
                    print(f"  Task {t} - Mean Accuracy: {metrics['mean_accuracy']:.4f}")
                    print(f"  Task {t} - Mean IoU: {metrics['mean_iou']:.4f}")
                    print(f"  Task {t} - Freq Weighted IoU: {metrics['fw_iou']:.4f}")

                    # Log per-class IoU
                    print(f"  Task {t} - IoU per class:")
                    # class_names = self.config.data.class_names if hasattr(self.config.data, 'class_names') else None
                    _, class_names = self.config.data.parse_color_and_names(t)
                    for i, iou in enumerate(metrics['iou']):
                        class_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
                        print(f"    {class_name}: {iou:.4f}")

                        self.writer.add_scalar(f'IoU_per_class_task_{t}/{class_name}', iou, epoch)
                    
                output_metrics[f'pixel_accuracy_{t}'] = metrics['pixel_accuracy']
                output_metrics[f'mean_accuracy_{t}'] = metrics['mean_accuracy']
                output_metrics[f'mean_iou_{t}'] = metrics['mean_iou']
                output_metrics[f'fw_iou_{t}'] = metrics['fw_iou']
            
            return output_metrics
    
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Returns:
            Dictionary with training statistics
        """
        if is_main_process():
            print(f"Starting training from epoch {self.start_epoch + 1}")
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config.training.epochs):
            # Set epoch for distributed sampler
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            # Training step
            train_metrics = self.train_epoch(epoch)
            
            # Only update history on main process
            if is_main_process():
                self.metrics_history['train_loss'].append(train_metrics['train_loss'])
                
                # Current learning rate
                lr = self.optimizer.param_groups[0]['lr']
                self.metrics_history['learning_rate'].append(lr)
            
            # Validation step
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.validate_epoch(epoch)
                
                # Update metrics history on main process
                if is_main_process():
                    self.metrics_history['val_loss'].append(val_metrics.get('val_loss', 0))

                    mean_miou_across_task = 0.0
                    num_tasks = 0
                    for t in self.config.tasks:
                        self.metrics_history[f'mean_iou_{t}'].append(val_metrics.get(f'mean_iou_{t}', 0))
                        self.metrics_history[f'pixel_accuracy_{t}'].append(val_metrics.get(f'pixel_accuracy_{t}', 0))
                        mean_miou_across_task += val_metrics.get(f'mean_iou_{t}', 0)
                        num_tasks += 1
                    # Check for best model (using mIoU)
                    current_miou = mean_miou_across_task / num_tasks
                    if current_miou > self.best_miou:
                        self.best_miou = current_miou
                        self.best_val_loss = val_metrics.get('val_loss', float('inf'))
                        self.best_epoch = epoch
                        self.best_metrics = val_metrics.copy()
                        
                        # Save best model checkpoint with metrics
                        self.save_checkpoint(epoch, is_best=True, metrics=val_metrics)
                    
                    # Also keep track of best loss
                    if val_metrics.get('val_loss', float('inf')) < self.best_val_loss and epoch != self.best_epoch:
                        self.best_val_loss = val_metrics.get('val_loss', float('inf'))

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('val_loss', 0))
                else:
                    self.scheduler.step()
            
            if is_main_process():
                self.save_checkpoint(epoch, metrics=val_metrics)
            
            # Print epoch summary on main process
            if is_main_process():
                lr = self.optimizer.param_groups[0]['lr']
                val_loss_str = f"{val_metrics.get('val_loss', 'N/A'):.4f}" if 'val_loss' in val_metrics else 'N/A'
                miou_str = f"{current_miou:.4f}"
                print(f"Epoch {epoch + 1}/{self.config.training.epochs} - "
                      f"Train loss: {train_metrics['train_loss']:.4f}, "
                      f"Val loss: {val_loss_str}, "
                      f"Mean mIoU across tasks: {miou_str}, "
                      f"LR: {lr:.6f}")
        
        # Save metrics history plot
        if is_main_process():
            self._save_metrics_plot(self.metrics_history)
        
        # Calculate training time
        total_time = time.time() - start_time
        
        # Return final statistics
        return {
            'total_epochs': self.config.training.epochs,
            'resumed_from_epoch': self.start_epoch + 1,
            'best_val_loss': self.best_val_loss,
            'best_miou': self.best_miou,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'training_time': total_time,
            'metrics_history': self.metrics_history
        }
    
    def log_segmentation_sample(self, image, prediction, target, epoch, tag, class_colors=None):
        """
        Log segmentation sample to TensorBoard (only on main process).
        
        Args:
            image: Input image tensor [C, H, W]
            prediction: Prediction tensor [C, H, W]
            target: Target tensor [H, W]
            epoch: Current epoch
            tag: Sample tag
        """
        if not is_main_process() or not self.writer:
            return
            
        # Create a figure with 3 subplots: input, prediction, ground truth
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Denormalize image
        mean = torch.tensor(self.config.data.mean).view(3, 1, 1).to(image.device)
        std = torch.tensor(self.config.data.std).view(3, 1, 1).to(image.device)
        image_denorm = image * std + mean
        image_np = image_denorm.cpu().numpy().transpose(1, 2, 0)
        image_np = np.clip(image_np, 0, 1)
        
        # Get target mask
        target_mask = target.cpu().numpy()
        # print(target_mask.shape)
        target_height, target_width = target_mask.shape

        # Convert to PIL Image for resizing
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        image_resized = image_pil.resize((target_width, target_height), Image.BILINEAR)
        image_np = np.array(image_resized).astype(np.float32) / 255.0
        
        # Get predicted class indices
        if prediction.dim() == 3:  # [C, H, W]
            predicted_mask = torch.argmax(prediction, dim=0).cpu().numpy()
        else:  # [H, W]
            predicted_mask = prediction.cpu().numpy()
        
        
        
        # Show original image
        ax1.imshow(image_np)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Create a colored prediction mask
        if class_colors is not None:
            # Create RGB prediction using class colors
            pred_colored = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3))
            for class_idx in range(len(class_colors)):
                class_mask = predicted_mask == class_idx
                pred_colored[class_mask] = class_colors[class_idx].cpu().numpy()
                
            # Show prediction as overlay
            ax2.imshow(image_np)
            ax2.imshow(pred_colored, alpha=0.5)
            ax2.set_title("Prediction Overlay")
            ax2.axis('off')
            
            # Create RGB ground truth using class colors
            target_colored = np.zeros((target_mask.shape[0], target_mask.shape[1], 3))
            for class_idx in range(len(class_colors)):
                class_mask = target_mask == class_idx
                target_colored[class_mask] = class_colors[class_idx].cpu().numpy()
                
            # Show ground truth as overlay
            ax3.imshow(image_np)
            ax3.imshow(target_colored, alpha=0.5)
            ax3.set_title("Ground Truth Overlay")
            ax3.axis('off')
        else:
            # Fallback if no color map is provided
            ax2.imshow(predicted_mask, cmap='nipy_spectral')
            ax2.set_title("Prediction")
            ax2.axis('off')
            
            ax3.imshow(target_mask, cmap='nipy_spectral')
            ax3.set_title("Ground Truth")
            ax3.axis('off')
        
        # Save plot to TensorBoard
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        sample_img = Image.open(buf)
        self.writer.add_image(f'Samples/{tag}', np.array(sample_img).transpose(2, 0, 1), epoch)
        plt.close(fig)
        buf.close()
        
        # Also add the segmentation prediction as image
        if class_colors is not None:
            self.writer.add_image(
                f'Masks/{tag}_pred', 
                torch.from_numpy(pred_colored.transpose(2, 0, 1)), 
                epoch
            )
            self.writer.add_image(
                f'Masks/{tag}_gt', 
                torch.from_numpy(target_colored.transpose(2, 0, 1)), 
                epoch
            )

    def save_checkpoint(self, epoch: int, is_best: bool = False, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Save a checkpoint of the model (only on main process).
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            metrics: Dictionary of validation metrics to save
        """
        if not is_main_process():
            return
            
        # Create checkpoint directory
        checkpoint_dir = self.output_dir / "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Get model state dict (handle DDP or DP)
        if hasattr(self.model, 'module'):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        # Create checkpoint with comprehensive information
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_miou': self.best_miou,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'config': self.config,
            'metrics_history': self.metrics_history,
            'use_amp': self.use_amp,
        }
        
        # Add scheduler state if available
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Add current metrics if provided
        if metrics:
            checkpoint['current_metrics'] = metrics

        if self.ema_model:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()
            
        # if self.sup_pseudo_enable_class_mask:
        # checkpoint['sup_pseudo_enable_class_mask'] = self.sup_pseudo_enable_class_mask
        # print(checkpoint['sup_pseudo_enable_class_mask'])   
        # Format checkpoint path
        if False: ## Set to True to save all checkpoints for all epoches
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(checkpoint, checkpoint_path)
        # If this is the best model, create a copy
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model checkpoint to {best_path}")
            
            # Also save a text file with the best metrics for quick reference
            if metrics:
                best_metrics_path = self.output_dir / "best_metrics.txt"
                with open(best_metrics_path, 'w') as f:
                    f.write(f"Best model (epoch {epoch + 1}):\n")
                    for k, v in metrics.items():
                        f.write(f"{k}: {v}\n")
                    f.write(f"use_amp: {self.use_amp}\n")
        
        # Save latest checkpoint (for easy resuming)
        latest_path = checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)   

class MultiDatasetCrossSupTrainer(MultiDatasetTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        deep_supervision = self.config.model.decoder.deep_supervision
        no_object_weight = self.config.model.decoder.no_object_weight
        # print(no_object_weight)
        # loss weights
        class_weight = self.config.model.decoder.class_weight
        dice_weight = self.config.model.decoder.dice_weight
        mask_weight = self.config.model.decoder.mask_weight

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=self.config.model.decoder.train_num_points,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = self.config.model.decoder.dec_layers
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        
        losses = ["labels", "masks"]
        
        self.criterion = {}

        for t in self.config.tasks:
            self.criterion[t] = SetCriterion(
                self.config.num_class_dict[t],
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight[t],
                losses=losses,
                num_points=self.config.model.decoder.train_num_points,
                oversample_ratio=self.config.model.decoder.oversample_ratio,
                importance_sample_ratio=self.config.model.decoder.importance_sample_ratio,
            ).to(self.device)
        
        self.criterion['pseudo'] = SetCriterion(
                self.config.data.num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight['pseudo'],
                losses=losses,
                num_points=self.config.model.decoder.train_num_points,
                oversample_ratio=self.config.model.decoder.oversample_ratio,
                importance_sample_ratio=self.config.model.decoder.importance_sample_ratio,
            ).to(self.device)
        
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            # 'mean_iou': [],
            # 'pixel_accuracy': [],
            'learning_rate': []
        }

        for t in self.config.tasks:
            self.metrics_history[f'mean_iou_{t}'] = []
            self.metrics_history[f'pixel_accuracy_{t}'] = []
            
        # if self.config.enable_ema:
        if getattr(self.config, 'enable_ema', False):
            self.ema_model = EMA(model=self.model, decay=self.config.ema_decay, device=self.device, )
            if is_main_process():
                print("Using EMA weights for pseudo-labeling")
            if self.resume_checkpoint:
                checkpoint = torch.load(self.resume_checkpoint, map_location=self.device)
                self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
        else:
            self.ema_model = None
        
        self.bdd_pseudo_enable_class_indices = []
        self.sup_pseudo_enable_class_mask = torch.zeros((self.config.data.num_classes, 1))
        if self.resume_checkpoint:
            checkpoint = torch.load(self.resume_checkpoint, map_location=self.device)
            self.sup_pseudo_enable_class_mask = checkpoint.get('sup_pseudo_enable_class_mask', np.zeros((self.config.data.num_classes, 1)))
            if is_main_process():
                sup_names = self.config.data.get_manual_superset_color_and_names()['sup_names']
                print("Load enable pseudo label class mask from checkpoints")
                # print(self.sup_pseudo_enable_class_mask)
                sup_pseudo_enable_class_name_indices = torch.nonzero(torch.from_numpy(self.sup_pseudo_enable_class_mask).squeeze(-1), as_tuple=True)[0]
                print("Enable superset class names are: ", [sup_names[i] for i in sup_pseudo_enable_class_name_indices])
        
        self.kl_loss = nn.KLDivLoss(reduction="none")
        self.config.pseudo_label_confidence_score = 0.9
        
    @torch.no_grad()
    def get_pseudo_probs(self, input_w, targets):
        with self.ema_model.apply_to(self.model.module if is_distributed() else self.model):
            self.model.eval()
            _, outputs_w = self.model(input_w)
            
        self.model.train()
        logits_w = outputs_w['bdd100k']["pred_logits"]
        masks_w = outputs_w['bdd100k']["pred_masks"]
        
        masks_w = F.interpolate(
            masks_w,
            size=input_w.shape[2:],  # (H, W)
            mode="bilinear",
            align_corners=False,
        )
        
        mask_probs = F.softmax(logits_w, dim=-1)[..., :-1]
        mask_pred = masks_w.sigmoid()
        probs_pseudo = torch.einsum('bqc,bqhw->bchw', mask_probs, mask_pred)
        
        scores, pseudo_labels = torch.max(probs_pseudo, dim=1)
        
        valid_mask = torch.ones_like(scores)
        
        # pseudo_labels = []
        bdd_pseudo_enable_class_mask = self.config.dataset_prediction_mapping['bdd100k'] @ self.sup_pseudo_enable_class_mask
        for i in range(len(logits_w)):
            # Only keep pseudo-labels for enabled superset classes, set others to ignore_index
            # pseudo_label_masked = pseudo_label.clone()
            for c in range(bdd_pseudo_enable_class_mask.shape[0]):
                if bdd_pseudo_enable_class_mask[c, 0] == 0:
                    valid_mask[pseudo_labels == c] = 0
            valid_mask[scores < self.config.pseudo_label_confidence_score] = 0
            has_gt_target = targets[i]['masks'].sum(dim=0) != 0
            # print(has_gt_target.shape)
            # print(pseudo_label_masked.shape)
            valid_mask[i][has_gt_target] = 0
            # pseudo_labels.append(pseudo_label_masked)

        
        # pseudo_labels = torch.stack(pseudo_labels)
        
        if is_main_process():
            if True:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle("Unified Prediction", fontsize=16)
                mean = torch.tensor(self.config.data.mean).view(3, 1, 1).to(input_w.device)
                std = torch.tensor(self.config.data.std).view(3, 1, 1).to(input_w.device)
                image_denorm = input_w[0] * std + mean
                image_np = image_denorm.cpu().numpy().transpose(1, 2, 0)
                image_np = np.clip(image_np, 0, 1)
                image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
                # image_resized = image_pil.resize(img.size, Image.BILINEAR)
                image_np = np.array(image_pil).astype(np.float32) / 255.0
                
                pseudo_labels[~valid_mask.bool()] = 255
                predicted_mask = pseudo_labels[0].cpu().numpy()
                ax1.imshow(image_np)
                # print(self.config.tasks[ids[0].item()])
                ax1.set_title("Original Image")
                ax1.axis('off')

                unified_label_colors, unified_class_names = self.config.data.parse_color_and_names(task='bdd100k')
                
                # unified_label_colors = self.config.data.get_manual_superset_color_and_names()['sup_colors']
                # unified_class_names = self.config.data.get_manual_superset_color_and_names()['sup_names']
                # Find duplicates
                color_to_indices = {}
                for idx, color in enumerate(unified_label_colors):
                    color_to_indices.setdefault(color, []).append(idx)

                class_colors = torch.tensor(unified_label_colors, dtype=torch.float32) / 255.0

                pred_colored = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3))
                for class_idx in range(len(class_colors)):
                    class_mask = predicted_mask == class_idx
                    pred_colored[class_mask] = class_colors[class_idx].cpu().numpy()

                ax2.imshow(image_np)
                ax2.imshow(pred_colored, alpha=0.5)
                ax2.set_title("Pseudo-Label Overlay")
                ax2.axis('off')

                ax3.imshow(pred_colored)
                ax3.set_title("Pseudo-Label Mask")
                ax3.axis('off')

                # Legend as a fourth subplot in the bottom row

                # Change layout to 2 rows: 1st row for images, 2nd row for legend (spanning all columns)
                fig.clf()  # Clear the current figure to redefine the layout
                gs = fig.add_gridspec(2, 3, height_ratios=[10, 1])

                # Redraw the three images in the first row
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(image_np)
                ax1.set_title("Original Image")
                ax1.axis('off')

                ax2 = fig.add_subplot(gs[0, 1])
                ax2.imshow(image_np)
                ax2.imshow(pred_colored, alpha=0.5)
                ax2.set_title("Pseudo-Label Overlay")
                ax2.axis('off')

                ax3 = fig.add_subplot(gs[0, 2])
                ax3.imshow(pred_colored)
                ax3.set_title("Pseudo-Label Mask")
                ax3.axis('off')

                # Legend in the bottom row, spanning all columns
                # Move the legend down by increasing the bottom margin and adjusting subplot spacing
                ax4 = fig.add_subplot(gs[1, :])
                for idx, (name, color) in enumerate(zip(unified_class_names, unified_label_colors)):
                    ax4.bar(idx, 1, color=np.array(color)/255, edgecolor='k', width=1)
                ax4.set_xticks(range(len(unified_class_names)))
                ax4.set_xticklabels(unified_class_names, rotation=90, ha='center', fontsize=10)
                ax4.set_yticks([])
                ax4.set_title('Legend', fontsize=12)
                ax4.set_xlim(-0.5, len(unified_class_names)-0.5)
                ax4.tick_params(axis='x', length=0)
                # plt.tight_layout(rect=[0, 0.08, 1, 1])  # Increase bottom margin (0.08)
                plt.tight_layout()
                plt.subplots_adjust(hspace=0.2)         # Increase vertical space between rows
                # plt.show()
                # Render figure to RGB array
                canvas = FigureCanvas(fig)
                canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)

                plt.close(fig)
                # Show in OpenCV (convert RGB→BGR)
                cv2.imshow("Pseudo Labels from EMA", cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
                k = cv2.waitKey(1) & 0xFF        # non-blocking
                if k in (27, ord('q')):          # ESC or q to quit
                    raise KeyboardInterrupt
                
        valid_mask = valid_mask.unsqueeze(1).expand(probs_pseudo.shape)
        
        return probs_pseudo, valid_mask.bool()
        
    @torch.no_grad()
    def get_pseudo_labels(self, input_w, targets):
        with self.ema_model.apply_to(self.model.module if is_distributed() else self.model):
            self.model.eval()
            outputs_w_full_ema, _ = self.model(input_w)
            
        self.model.train()
        logits_w = outputs_w_full_ema["pred_logits"]
        masks_w = outputs_w_full_ema["pred_masks"]
        masks_w = F.interpolate(
            masks_w,
            size=input_w.shape[2:],  # (H, W)
            mode="bilinear",
            align_corners=False,
        )
        
        pseudo_labels = []
        
        for i in range(len(logits_w)):
            if is_distributed():
                pseudo_label = self.model.module.decoder.semantic_inference(logits_w[i], masks_w[i])
            else:
                pseudo_label = self.model.decoder.semantic_inference(logits_w[i], masks_w[i])
            # pseudo_label = torch.argmax(pseudo_label, dim=1)
            scores, pseudo_label = torch.max(pseudo_label, dim=0)
            # Only keep pseudo-labels for enabled superset classes, set others to ignore_index
            pseudo_label_masked = pseudo_label.clone()
            for c in range(self.sup_pseudo_enable_class_mask.shape[0]):
                if self.sup_pseudo_enable_class_mask[c, 0] == 0:
                    pseudo_label_masked[pseudo_label == c] = self.config.data.ignore_index
            pseudo_label_masked[scores < self.config.pseudo_label_confidence_score] = self.config.data.ignore_index
            has_gt_target = targets[i]['masks'].sum(dim=0) != 0
            # print(has_gt_target.shape)
            # print(pseudo_label_masked.shape)
            pseudo_label_masked[has_gt_target] = self.config.data.ignore_index
            pseudo_labels.append(pseudo_label_masked)
            
        pseudo_labels = torch.stack(pseudo_labels)
        
        if is_main_process():
            if True:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle("Unified Prediction", fontsize=16)
                mean = torch.tensor(self.config.data.mean).view(3, 1, 1).to(input_w.device)
                std = torch.tensor(self.config.data.std).view(3, 1, 1).to(input_w.device)
                image_denorm = input_w[0] * std + mean
                image_np = image_denorm.cpu().numpy().transpose(1, 2, 0)
                image_np = np.clip(image_np, 0, 1)
                image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
                # image_resized = image_pil.resize(img.size, Image.BILINEAR)
                image_np = np.array(image_pil).astype(np.float32) / 255.0
                predicted_mask = pseudo_labels[0].cpu().numpy()
                ax1.imshow(image_np)
                # print(self.config.tasks[ids[0].item()])
                ax1.set_title("Original Image")
                ax1.axis('off')

                unified_label_colors = self.config.data.get_manual_superset_color_and_names()['sup_colors']
                unified_class_names = self.config.data.get_manual_superset_color_and_names()['sup_names']
                # Find duplicates
                color_to_indices = {}
                for idx, color in enumerate(unified_label_colors):
                    color_to_indices.setdefault(color, []).append(idx)

                class_colors = torch.tensor(unified_label_colors, dtype=torch.float32) / 255.0

                pred_colored = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3))
                for class_idx in range(len(class_colors)):
                    class_mask = predicted_mask == class_idx
                    pred_colored[class_mask] = class_colors[class_idx].cpu().numpy()

                ax2.imshow(image_np)
                ax2.imshow(pred_colored, alpha=0.5)
                ax2.set_title("Pseudo-Label Overlay")
                ax2.axis('off')

                ax3.imshow(pred_colored)
                ax3.set_title("Pseudo-Label Mask")
                ax3.axis('off')

                # Legend as a fourth subplot in the bottom row

                # Change layout to 2 rows: 1st row for images, 2nd row for legend (spanning all columns)
                fig.clf()  # Clear the current figure to redefine the layout
                gs = fig.add_gridspec(2, 3, height_ratios=[10, 1])

                # Redraw the three images in the first row
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(image_np)
                ax1.set_title("Original Image")
                ax1.axis('off')

                ax2 = fig.add_subplot(gs[0, 1])
                ax2.imshow(image_np)
                ax2.imshow(pred_colored, alpha=0.5)
                ax2.set_title("Pseudo-Label Overlay")
                ax2.axis('off')

                ax3 = fig.add_subplot(gs[0, 2])
                ax3.imshow(pred_colored)
                ax3.set_title("Pseudo-Label Mask")
                ax3.axis('off')

                # Legend in the bottom row, spanning all columns
                # Move the legend down by increasing the bottom margin and adjusting subplot spacing
                ax4 = fig.add_subplot(gs[1, :])
                for idx, (name, color) in enumerate(zip(unified_class_names, unified_label_colors)):
                    ax4.bar(idx, 1, color=np.array(color)/255, edgecolor='k', width=1)
                ax4.set_xticks(range(len(unified_class_names)))
                ax4.set_xticklabels(unified_class_names, rotation=90, ha='center', fontsize=10)
                ax4.set_yticks([])
                ax4.set_title('Legend', fontsize=12)
                ax4.set_xlim(-0.5, len(unified_class_names)-0.5)
                ax4.tick_params(axis='x', length=0)
                # plt.tight_layout(rect=[0, 0.08, 1, 1])  # Increase bottom margin (0.08)
                plt.tight_layout()
                plt.subplots_adjust(hspace=0.2)         # Increase vertical space between rows
                # plt.show()
                # Render figure to RGB array
                canvas = FigureCanvas(fig)
                canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)

                plt.close(fig)
                # Show in OpenCV (convert RGB→BGR)
                cv2.imshow("Pseudo Labels from EMA", cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
                k = cv2.waitKey(1) & 0xFF        # non-blocking
                if k in (27, ord('q')):          # ESC or q to quit
                    raise KeyboardInterrupt
        
        return pseudo_labels
    
    def compute_loss(self, input_s, input_w, targets, ids):
        outputs_s_full, outputs_s = self.model(input_s)    
        if self.config.data.task_type == 'semantic_segmentation':
            raise NotImplementedError
        
        elif self.config.data.task_type == 'instance_segmentation':
            losses = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
            # Dataset-sepcific loss with GT supervision
            for t_id, t in enumerate(self.config.tasks):
                
                ### Temp workaround
                outputs_t = outputs_s[t]
                outputs_temp = {'pred_logits': outputs_t['pred_logits'][ids == t_id], 'pred_masks': outputs_t['pred_masks'][ids == t_id] }
                outputs_temp['aux_outputs'] = []
                for aux_i in outputs_t['aux_outputs']:
                    outputs_temp['aux_outputs'].append({"pred_logits": aux_i['pred_logits'][ids == t_id], "pred_masks": aux_i['pred_masks'][ids == t_id]})
                if len(targets[t]) == 0:
                    # loss = 0.0
                    loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
                else:
                    loss = self.criterion[t](outputs_temp, targets[t])

                if isinstance(loss, dict):
                    for k in list(loss.keys()):
                        if k in self.criterion[t].weight_dict:
                            loss[k] *= self.criterion[t].weight_dict[k]
                        else:
                            # remove this loss if not specified in `weight_dict`
                            loss.pop(k)
                    loss_dict = loss.copy()
                    loss = sum(loss_dict.values())
                losses = losses + loss
        # print(f"Rank {get_rank()} computed supervised loss at batch {batch_idx}")
        if self.sup_pseudo_enable_class_mask.sum() != 0:
            # print("Compute Pseudo Label Loss...")
            # Pseudo-label loss
            if (ids == 1).sum() > 0:
                # With binary mask
                # pseudo_labels = self.get_pseudo_labels(input_w=input_w[ids == 1], targets=targets['r2s100k'])
                # pseudo_target = []
                # valid = torch.zeros(pseudo_labels.shape[0]).bool()
                # for i in range(pseudo_labels.shape[0]):
                #     labels, masks = self.train_loader.dataset.semantic_to_instances(pseudo_labels[i].detach().cpu().numpy())
                #     if len(labels) == 0:
                #         continue
                #     masks = torch.stack([torch.from_numpy(mask) for mask in masks])
                #     labels = torch.tensor(labels, dtype=torch.long)
                #     pseudo_target.append({
                #         "labels": labels,
                #         "masks": masks,
                #     })
                #     valid[i] = True

                # if valid.sum() > 0: ## Skip when no valid pseudo label masks within the batch
                #     ### Temp workaround
                #     outputs_t = outputs_s_full
                #     outputs_temp = {'pred_logits': outputs_t['pred_logits'][ids == 1][valid], 'pred_masks': outputs_t['pred_masks'][ids == 1][valid] }
                #     outputs_temp['aux_outputs'] = []
                #     for aux_i in outputs_t['aux_outputs']:
                #         outputs_temp['aux_outputs'].append({"pred_logits": aux_i['pred_logits'][ids == 1][valid], "pred_masks": aux_i['pred_masks'][ids == 1][valid]})
                #     loss_pseudo = self.criterion['pseudo'](outputs_temp, pseudo_target)
                #     if isinstance(loss_pseudo, dict):
                #         for k in list(loss_pseudo.keys()):
                #             if k in self.criterion['pseudo'].weight_dict:
                #                 loss_pseudo[k] *= self.criterion['pseudo'].weight_dict[k]
                #             else:
                #                 # remove this loss if not specified in `weight_dict`
                #                 loss_pseudo.pop(k)
                #         loss_dict = loss_pseudo.copy()
                #         loss_pseudo = sum(loss_dict.values())
                # else:
                #     dummy = torch.zeros(1, device=self.device)
                #     if is_distributed():
                #         torch.distributed.all_reduce(dummy, op=torch.distributed.ReduceOp.SUM)
                #     loss_pseudo = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
                
                # With distribution
                probs_pseudo, valid_mask  = self.get_pseudo_probs(input_w=input_w[ids == 1], targets=targets['r2s100k'])
                if valid_mask.sum() > 0:
                    probs_pseudo = (probs_pseudo / probs_pseudo.sum(1, keepdim=True).clamp_min(1e-8)).detach() ## Normalize
                    # print(probs_pseudo.shape, valid_mask.shape)
                    outputs_s_bdd = outputs_s['bdd100k']
                    logits_s_bdd = outputs_s_bdd["pred_logits"][ids == 1]
                    masks_s_bdd = outputs_s_bdd["pred_masks"][ids == 1]
                    
                    masks_s_bdd = F.interpolate(
                        masks_s_bdd,
                        size=input_s.shape[2:],  # (H, W)
                        mode="bilinear",
                        align_corners=False,
                    )
                    
                    probs_pred = F.softmax(logits_s_bdd, dim=-1)[..., :-1]
                    mask_pred = masks_s_bdd.sigmoid()
                    probs_pred = torch.einsum('bqc,bqhw->bchw', probs_pred, mask_pred)
                    probs_pred = (probs_pred / probs_pred.sum(1, keepdim=True).clamp_min(1e-8)) ## Normalize
                    
                    loss_pseudo = self.kl_loss(torch.log(probs_pred.clamp_min(1e-8)), probs_pseudo)[valid_mask].mean()
                    # loss_pseudo = 0.0
                    # print(loss_pseudo)
                else:
                    dummy = torch.zeros(1, device=self.device)
                    if is_distributed():
                        torch.distributed.all_reduce(dummy, op=torch.distributed.ReduceOp.SUM)
                    loss_pseudo = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
            else:
                dummy = torch.zeros(1, device=self.device)
                if is_distributed():
                    torch.distributed.all_reduce(dummy, op=torch.distributed.ReduceOp.SUM)
                loss_pseudo = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
        else:
            dummy = torch.zeros(1, device=self.device)
            if is_distributed():
                torch.distributed.all_reduce(dummy, op=torch.distributed.ReduceOp.SUM)
            loss_pseudo = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
                     
        losses = losses + self.config.pseudo_loss_weight * loss_pseudo
        return losses

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        loss_meter = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        cuda_mem = AverageMeter()
        
        start = time.time()
        
        # Only show progress bar on main process
        if is_main_process():
            pbar = tqdm(self.train_loader)
            pbar.set_description(f'[TRAINING] Epoch {epoch + 1}')
        else:
            pbar = self.train_loader

        total_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(pbar):
            # if batch_idx > 5:
            #     break
            # Measure data loading time
            data_time.update(time.time() - start)
            
            input_w, input_s, targets, ids = batch
            input_w = input_w.to(self.device)
            input_s = input_s.to(self.device)
            if not isinstance(targets, list) and not isinstance(targets, dict):
                targets = targets.to(self.device)
            
            ### Temp workaround
            targets_temp = {self.config.tasks[0]: [], self.config.tasks[1]: []}
            for i, id in enumerate(ids):
                targets_temp[self.config.tasks[id]].append(targets[i])
            targets = targets_temp
            ids = ids.to(self.device)
            # Zero gradients
            self.optimizer.zero_grad()
            
            if self.use_amp and self.scaler is not None:
                # Mixed precision training
                with autocast():
                    losses = self.compute_loss(input_s=input_s, input_w=input_w, targets=targets, ids=ids)
                # Scale loss to prevent gradient underflow
                scaled_loss = self.scaler.scale(losses)
                scaled_loss.backward()
                
                # Gradient clipping (if configured)
                if hasattr(self.config.training, 'grad_clip_val') and self.config.training.grad_clip_val:
                    # Unscale before clipping to get true gradient norms
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.grad_clip_val
                    )
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                losses = self.compute_loss(input_s=input_s, input_w=input_w, targets=targets, ids=ids)
                losses.backward()
                
                # Gradient clipping if configured
                if hasattr(self.config.training, 'grad_clip_val') and self.config.training.grad_clip_val:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.grad_clip_val
                    )
                
                # Update weights
                self.optimizer.step()
            
            self.ema_model.update(self.model.module if is_distributed() else self.model)
            
            # For testing the output of EMA
            if is_main_process():
                if True:
                    with self.ema_model.apply_to(self.model.module if is_distributed() else self.model):
                        self.model.eval()
                        with torch.no_grad():
                            outputs_superset_ema, _ = self.model(input_w)
                            logits_sup = outputs_superset_ema["pred_logits"]
                            masks_sup = outputs_superset_ema["pred_masks"]
                            masks_sup = F.interpolate(
                                masks_sup,
                                size=input_w.shape[2:],  # (H, W)
                                mode="bilinear",
                                align_corners=False,
                            )
                            if is_distributed():
                                outputs_map = self.model.module.decoder.semantic_inference(logits_sup[0], masks_sup[0])
                            else:
                                outputs_map = self.model.decoder.semantic_inference(logits_sup[0], masks_sup[0])
                            # print(outputs_map.shape)
                            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                            fig.suptitle("Unified Prediction", fontsize=16)
                            mean = torch.tensor(self.config.data.mean).view(3, 1, 1).to(input_w.device)
                            std = torch.tensor(self.config.data.std).view(3, 1, 1).to(input_w.device)
                            image_denorm = input_w[0] * std + mean
                            image_np = image_denorm.cpu().numpy().transpose(1, 2, 0)
                            image_np = np.clip(image_np, 0, 1)
                            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
                            # image_resized = image_pil.resize(img.size, Image.BILINEAR)
                            image_np = np.array(image_pil).astype(np.float32) / 255.0
                            if outputs_map.dim() == 3:  # [C, H, W]
                                predicted_mask = torch.argmax(outputs_map, dim=0).cpu().numpy()
                            else:  # [H, W]
                                predicted_mask = outputs_map.cpu().numpy()
                            ax1.imshow(image_np)
                            # print(self.config.tasks[ids[0].item()])
                            ax1.set_title("Original Image")
                            ax1.axis('off')

                            unified_label_colors = self.config.data.get_manual_superset_color_and_names()['sup_colors']
                            unified_class_names = self.config.data.get_manual_superset_color_and_names()['sup_names']
                            # Find duplicates
                            color_to_indices = {}
                            for idx, color in enumerate(unified_label_colors):
                                color_to_indices.setdefault(color, []).append(idx)

                            class_colors = torch.tensor(unified_label_colors, dtype=torch.float32) / 255.0

                            if outputs_map.dim() == 3:  # [C, H, W]
                                predicted_mask = torch.argmax(outputs_map, dim=0).cpu().numpy()
                            else:  # [H, W]
                                predicted_mask = outputs_map.cpu().numpy()

                            pred_colored = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3))
                            for class_idx in range(len(class_colors)):
                                class_mask = predicted_mask == class_idx
                                pred_colored[class_mask] = class_colors[class_idx].cpu().numpy()

                            ax2.imshow(image_np)
                            ax2.imshow(pred_colored, alpha=0.5)
                            ax2.set_title("Prediction Overlay")
                            ax2.axis('off')

                            ax3.imshow(pred_colored)
                            ax3.set_title("Predicted Mask")
                            ax3.axis('off')

                            # Legend as a fourth subplot in the bottom row

                            # Change layout to 2 rows: 1st row for images, 2nd row for legend (spanning all columns)
                            fig.clf()  # Clear the current figure to redefine the layout
                            gs = fig.add_gridspec(2, 3, height_ratios=[10, 1])

                            # Redraw the three images in the first row
                            ax1 = fig.add_subplot(gs[0, 0])
                            ax1.imshow(image_np)
                            ax1.set_title("Original Image {name}".format(name=self.config.tasks[ids[0].item()]))
                            ax1.axis('off')

                            ax2 = fig.add_subplot(gs[0, 1])
                            ax2.imshow(image_np)
                            ax2.imshow(pred_colored, alpha=0.5)
                            ax2.set_title("Prediction Overlay {name}".format(name=self.config.tasks[ids[0].item()]))
                            ax2.axis('off')

                            ax3 = fig.add_subplot(gs[0, 2])
                            ax3.imshow(pred_colored)
                            ax3.set_title("Predicted Mask {name}".format(name=self.config.tasks[ids[0].item()]))
                            ax3.axis('off')

                            # Legend in the bottom row, spanning all columns
                            # Move the legend down by increasing the bottom margin and adjusting subplot spacing
                            ax4 = fig.add_subplot(gs[1, :])
                            for idx, (name, color) in enumerate(zip(unified_class_names, unified_label_colors)):
                                ax4.bar(idx, 1, color=np.array(color)/255, edgecolor='k', width=1)
                            ax4.set_xticks(range(len(unified_class_names)))
                            ax4.set_xticklabels(unified_class_names, rotation=90, ha='center', fontsize=10)
                            ax4.set_yticks([])
                            ax4.set_title('Legend', fontsize=12)
                            ax4.set_xlim(-0.5, len(unified_class_names)-0.5)
                            ax4.tick_params(axis='x', length=0)
                            # plt.tight_layout(rect=[0, 0.08, 1, 1])  # Increase bottom margin (0.08)
                            plt.tight_layout()
                            plt.subplots_adjust(hspace=0.2)         # Increase vertical space between rows
                            # plt.show()
                            # Render figure to RGB array
                            canvas = FigureCanvas(fig)
                            canvas.draw()
                            w, h = fig.canvas.get_width_height()
                            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)

                            plt.close(fig)
                            # Show in OpenCV (convert RGB→BGR)
                            cv2.imshow("EMA Prediction on Supersets", cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
                            k = cv2.waitKey(1) & 0xFF        # non-blocking
                            if k in (27, ord('q')):          # ESC or q to quit
                                break
                            # cv2.destroyAllWindows()
                            
                    self.model.train()
            loss = losses
            # Update metrics
            loss_meter.update(loss.item(), input_w.size(0))
            batch_time.update(time.time() - start)
            if torch.cuda.is_available():
                cuda_mem.update(torch.cuda.max_memory_allocated(device=None) / (1024 * 1024 * 1024))
            
            # Log batch-level metrics only from main process (less frequently)
            if is_main_process() and self.writer and batch_idx % 10 == 0:
                global_step = epoch * total_batches + batch_idx
                self.writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_Rate', current_lr, global_step)

                if self.use_amp and self.scaler is not None:
                    self.writer.add_scalar('AMP/loss_scale', self.scaler.get_scale(), global_step)
            # Update progress bar only on main process
            if is_main_process():
                monitor = {
                    'Loss': f'{loss_meter.val:.4f} ({loss_meter.avg:.4f})',
                    'CUDA': f'{cuda_mem.val:.2f} ({cuda_mem.avg:.2f})'
                }
                if self.use_amp and self.scaler is not None:
                    monitor['Scale'] = f'{self.scaler.get_scale():.0f}'
                pbar.set_postfix(monitor)

            start = time.time()
        
        # Log epoch-level metrics from all processes
        if self.writer:
            if hasattr(self, 'rank_prefix'):
                self.writer.add_scalar(f'{self.rank_prefix}Loss/train_epoch', loss_meter.avg, epoch)
                self.writer.add_scalar(f'{self.rank_prefix}Memory/avg_cuda_memory_gb', cuda_mem.avg, epoch)
            else:
                self.writer.add_scalar('Loss/train_epoch', loss_meter.avg, epoch)
                self.writer.add_scalar('Memory/avg_cuda_memory_gb', cuda_mem.avg, epoch)
        
        # Gather metrics from all processes
        if is_distributed():
            # Prepare metrics to gather
            process_metrics = torch.tensor([
                loss_meter.avg,
                cuda_mem.avg,
                batch_time.avg,
                data_time.avg,
                loss_meter.sum,   # Total loss across all batches
                loss_meter.count  # Total samples processed
            ], device=self.device)
            
            # Gather metrics from all processes to rank 0
            gathered_metrics = gather_tensor(process_metrics)
            
            if is_main_process():
                # Extract gathered metrics
                world_size = get_world_size()
                all_avg_losses = gathered_metrics[0::6]  # Every 6th element starting from 0
                all_cuda_mem = gathered_metrics[1::6]
                all_batch_times = gathered_metrics[2::6]
                all_data_times = gathered_metrics[3::6]
                all_total_losses = gathered_metrics[4::6]
                all_sample_counts = gathered_metrics[5::6]
                
                # Compute global weighted average loss
                total_samples = all_sample_counts.sum().item()
                global_loss = all_total_losses.sum().item() / total_samples if total_samples > 0 else 0
                
                # Log individual process metrics
                for rank in range(world_size):
                    self.writer.add_scalar(f'Loss/train_rank_{rank}', all_avg_losses[rank].item(), epoch)
                    self.writer.add_scalar(f'Memory/cuda_memory_rank_{rank}', all_cuda_mem[rank].item(), epoch)
                    self.writer.add_scalar(f'Timing/batch_time_rank_{rank}', all_batch_times[rank].item(), epoch)
                    self.writer.add_scalar(f'Timing/data_time_rank_{rank}', all_data_times[rank].item(), epoch)
                    self.writer.add_scalar(f'Samples/train_count_rank_{rank}', all_sample_counts[rank].item(), epoch)
                
                # Log aggregated metrics
                self.writer.add_scalar('Loss/train_global', global_loss, epoch)
                self.writer.add_scalar('Loss/train_avg_across_ranks', all_avg_losses.mean().item(), epoch)
                self.writer.add_scalar('Loss/train_min', all_avg_losses.min().item(), epoch)
                self.writer.add_scalar('Loss/train_max', all_avg_losses.max().item(), epoch)
                self.writer.add_scalar('Loss/train_std', all_avg_losses.std().item(), epoch)
                
                self.writer.add_scalar('Memory/cuda_memory_avg', all_cuda_mem.mean().item(), epoch)
                self.writer.add_scalar('Memory/cuda_memory_max', all_cuda_mem.max().item(), epoch)
                
                self.writer.add_scalar('Timing/batch_time_avg', all_batch_times.mean().item(), epoch)
                self.writer.add_scalar('Timing/data_time_avg', all_data_times.mean().item(), epoch)
                
                try:
                    # Log histograms for better visualization
                    self.writer.add_histogram('Distributions/loss_per_rank', all_avg_losses.cpu(), epoch)
                    self.writer.add_histogram('Distributions/memory_per_rank', all_cuda_mem.cpu(), epoch)
                except Exception as e: 
                    print(e)
                return {'train_loss': global_loss}
            else:
                return {'train_loss': loss_meter.avg}
        else:
            # Single process
            if is_main_process() and self.writer:
                self.writer.add_scalar('Loss/train', loss_meter.avg, epoch)
                self.writer.add_scalar('Memory/cuda_memory', cuda_mem.avg, epoch)
            
            return {'train_loss': loss_meter.avg}

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        loss_meter = AverageMeter()
        cuda_mem = AverageMeter()

        # Initialize confusion matrix for IoU calculation
        confusion_matrix = {}
        for t in self.config.tasks:
            confusion_matrix[t] = ConfusionMatrix(self.config.num_class_dict[t], self.config.data.ignore_index)
        # confusion_matrix = ConfusionMatrix(self.config.data.num_classes, self.config.data.ignore_index)
        
        # Create storage for visualizing a few samples
        num_samples_to_visualize = min(4, self.config.training.batch_size)
        # samples_visualized = 0
        samples_visualized = {self.config.tasks[0]: 0, self.config.tasks[1]: 0}

        with torch.no_grad():
            # Only show progress bar on main process
            if is_main_process():
                pbar = tqdm(self.val_loader)
                pbar.set_description(f'[VALIDATION] Epoch {epoch + 1}')
            else:
                pbar = self.val_loader

            for batch_idx, batch in enumerate(pbar):
                # Get data
                # if batch_idx > 10:
                #     break
                inputs, _, targets, ids = batch
                inputs = inputs.to(self.device)
                if not isinstance(targets, list) and not isinstance(targets, dict):
                    targets = targets.to(self.device)
                ### Temp workaround
                targets_temp = {self.config.tasks[0]: [], self.config.tasks[1]: []}
                for i, id in enumerate(ids):
                    targets_temp[self.config.tasks[id]].append(targets[i])
                targets = targets_temp
                ids = ids.to(self.device)

                # print(f"Rank: {get_rank()}: ids: {ids}, targets bdd: {len(targets[self.config.tasks[0]])}, targets bdd: {len(targets[self.config.tasks[1]])}")
                
                # Run forward pass
                _, outputs = self.model(inputs)
                # print(targets)
                if self.config.data.task_type == 'semantic_segmentation':
                    # outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[1:], mode="bilinear", align_corners=False)
                    # loss = self.criterion_eval(outputs, targets)
                    raise NotImplementedError
                elif self.config.data.task_type == 'instance_segmentation':
                    losses = 0.0
                    outputs_for_eval = {}
                    for t_id, t in enumerate(self.config.tasks):
                        # loss = self.criterion[t](outputs[t], targets[t])
                        ### Temp workaround
                        outputs_t = outputs[t]
                        outputs_temp = {'pred_logits': outputs_t['pred_logits'][ids == t_id], 'pred_masks': outputs_t['pred_masks'][ids == t_id] }
                        outputs_temp['aux_outputs'] = []
                        for aux_i in outputs_t['aux_outputs']:
                            outputs_temp['aux_outputs'].append({"pred_logits": aux_i['pred_logits'][ids == t_id], "pred_masks": aux_i['pred_masks'][ids == t_id]})
                        outputs_for_eval[t] = outputs_temp
                        
                        if len(targets[t]) == 0:
                            loss = 0.0
                        else:
                            loss = self.criterion[t](outputs_temp, targets[t])

                        if isinstance(loss, dict):
                            for k in list(loss.keys()):
                                if k in self.criterion[t].weight_dict:
                                    loss[k] *= self.criterion[t].weight_dict[k]
                                else:
                                    # remove this loss if not specified in `weight_dict`
                                    loss.pop(k)
                            loss_dict = loss.copy()
                            loss = sum(loss_dict.values())
                        losses += loss
                    outputs = outputs_for_eval
                    loss = losses
                # Update metrics
                loss_meter.update(loss.item(), inputs.size(0))
                if torch.cuda.is_available():
                    cuda_mem.update(torch.cuda.max_memory_allocated(device=None) / (1024 * 1024 * 1024))
                
                # Update confusion matrix for IoU calculation
                outputs_for_eval = {}
                targets_for_eval = {}
                if isinstance(outputs, dict) and isinstance(targets, dict):
                    ## process mask2former head outputs and targets into [B, C, H, W] and [B, H, W]
                    for t in self.config.tasks:
                        if len(targets[t]) != 0:
                            target_hw = targets[t][0]['masks'].shape[1:]
                            mask_cls_results = outputs[t]["pred_logits"]
                            mask_pred_results = outputs[t]["pred_masks"]
                            # print("mask_cls_results:", mask_cls_results.shape)
                            # print("mask_pred_results", mask_pred_results.shape)
                            # print("target_hw: ", target_hw)
                            # upsample masks
                            mask_pred_results = F.interpolate(
                                mask_pred_results,
                                size=target_hw,
                                mode="bilinear",
                                align_corners=False,
                            )

                            outputs_map = []
                            for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):
                                # semantic segmentation inference
                                if is_distributed():
                                    semmap = self.model.module.decoder.semantic_inference(mask_cls_result, mask_pred_result)
                                else:
                                    semmap = self.model.decoder.semantic_inference(mask_cls_result, mask_pred_result)
                                outputs_map.append(semmap)
                            outputs_map = torch.stack(outputs_map)
                            targets_map = self.val_loader.dataset.instances_to_semantic(targets=targets[t], ignore_index=self.config.data.ignore_index)
                        
                            confusion_matrix[t].update(outputs_map, targets_map)

                            # print(f"Rank: {get_rank()}: Task: {t}, outputs_map: {outputs_map.shape}, targets_map: {targets_map.shape}")
                            # print(t)
                            # print(outputs_map.shape)
                            # print(targets_map.shape)

                            outputs_for_eval[t] = outputs_map
                            targets_for_eval[t] = targets_map
                            
                    outputs = outputs_for_eval
                    targets = targets_for_eval
                    
                # Update progress bar only on main process
                if is_main_process():
                    monitor = {
                        'Loss': f'{loss_meter.val:.4f} ({loss_meter.avg:.4f})',
                        'CUDA': f'{cuda_mem.val:.2f} ({cuda_mem.avg:.2f})'
                    }
                    pbar.set_postfix(monitor)

                # Visualize samples only on main process
                if is_main_process() and self.writer:
                    # try:
                    # if samples_visualized < num_samples_to_visualize:
                        # sample_idx = np.random.randint(0, inputs.shape[0])
                    for t_id, t in enumerate(self.config.tasks):
                        # if len(targets[t]) != 0:
                        if samples_visualized[t] < num_samples_to_visualize:
                            if t in targets.keys():
                                inputs_t = inputs[ids == t_id]
                                sample_idx = np.random.randint(0, inputs_t.shape[0])
                                label_colors_list, _ = self.config.data.parse_color_and_names(t)
                                self.log_segmentation_sample(
                                    inputs_t[sample_idx], 
                                    outputs[t][sample_idx], 
                                    targets[t][sample_idx],
                                    epoch, 
                                    f"sample_{t}_{samples_visualized[t]}",
                                    torch.tensor(label_colors_list, dtype=torch.float32) / 255.0,
                                )
                                samples_visualized[t] += 1
                        # samples_visualized += 1
                    # except Exception as e: 
                    #     # print(e)
                    #     print(f"[WARNING] Fail to log image to tensorboard")
                    #     print(e)

        # Gather metrics from all processes
        if is_distributed():
            # print(f"Rank: {get_rank()} finish epoch...", flush=True)
            # dist.barrier()
            synchronize()
            # Gather basic loss and memory stats
            process_stats = torch.tensor([
                loss_meter.avg,
                cuda_mem.avg,
                loss_meter.sum,   # Total loss
                loss_meter.count  # Total samples
            ], device=self.device, dtype=torch.float32)
            
            # if process_stats.is_cuda:
            #     torch.cuda.synchronize(process_stats.device)
            
            # Force the WORLD group explicitly
            # group = dist.group.WORLD
            # print(f"[global {get_rank()}] group_rank={dist.get_rank(group)} "
            #                 f"group_world={dist.get_world_size(group)}", flush=True)

            # bufs = [torch.empty_like(process_stats) for _ in range(get_world_size())]
            # dist.all_gather(bufs, process_stats, group=group)
            # gathered_stats = torch.stack(bufs, dim=0)

            # print(f"[global {get_rank()}] gathered shape {tuple(gathered_stats.shape)}", flush=True)
            # if get_rank() == 0:
            #     print(gathered_stats, flush=True)

            # print(f"Rank: {get_rank()} before process_stats {process_stats}", flush=True)
            gathered_stats = gather_tensor(process_stats)
            # print(f"Rank: {get_rank()} after gathered_stats {gathered_stats}", flush=True)
            # # print(gathered_stats.shape)

            # if is_main_process():
            #     print(gathered_stats, flush=True)

            # dist.barrier()

            # gathered_cm = {
            #     self.config.tasks[0]: torch.zeros((get_world_size(), self.config.num_class_dict[self.config.tasks[0]], self.config.num_class_dict[self.config.tasks[0]])),
            #     self.config.tasks[1]: torch.zeros((get_world_size(), self.config.num_class_dict[self.config.tasks[1]], self.config.num_class_dict[self.config.tasks[1]])),
            # }

            # Gather confusion matrices to CPU (root-only) and compute global/per-rank metrics on rank 0
            # gathered_cm = {}
            # for t in self.config.tasks:
            #     num_classes = self.config.num_class_dict[t]

            #     # Always make a (C, C) CPU numpy array, even if this rank saw no samples
            #     if confusion_matrix[t].mat is not None:
            #         cm_np = confusion_matrix[t].mat.astype(np.int64, copy=False)
            #     else:
            #         cm_np = np.zeros((num_classes, num_classes), dtype=np.int64)

            #     # Root only will receive a list; others pass None
            #     obj_list = [None for _ in range(get_world_size())] if is_main_process() else None

            #     # CPU object gather to rank 0 (works regardless of NCCL/Gloo)
            #     dist.gather_object(cm_np, obj_list, dst=0)

            #     # On rank 0: stack into [world_size, C, C] tensor for later use
            #     if is_main_process():
            #         gathered_cm[t] = torch.stack(
            #             [torch.as_tensor(a, dtype=torch.float64) for a in obj_list], dim=0
            #         )  # shape: [ws, C, C]
            

            ## This stuck
            ## Gather and sum confusion matrices for accurate global metrics
            gathered_cm = {}

            # task_groups = {}
            # for t in self.config.tasks:
            #     # This assumes all ranks participate in all tasks
            #     task_groups[t] = dist.new_group()

            for t in self.config.tasks:
                # dist.barrier()
                synchronize()
                # print(f"Rank: {get_rank()} inside for t in self.config.tasks")
                if confusion_matrix[t].mat is not None:
                    cm_tensor = torch.tensor(confusion_matrix[t].mat, dtype=torch.float32, device=self.device)
                else:
                    # Handle case where no samples were processed
                    cm_tensor = torch.zeros((self.config.num_class_dict[t], self.config.num_class_dict[t]), 
                                        dtype=torch.float32, device=self.device)
                # print(f"Rank {get_rank()} at task {t}: cm_tensor.shape: {cm_tensor.shape}")
                # gathered_cm_t = gather_tensor_stack(cm_tensor, group=task_groups[t])
                # gathered_cm_t = gather_variable_tensor(cm_tensor)
                gathered_cm_t = gather_tensor(cm_tensor)
                gathered_cm[t] = gathered_cm_t


            if is_main_process():
                world_size = get_world_size()
                # print(f"line 2226: {world_size}")
                # Extract individual process stats
                all_avg_losses = gathered_stats[0::4]
                all_cuda_mem = gathered_stats[1::4]
                all_total_losses = gathered_stats[2::4]
                all_sample_counts = gathered_stats[3::4]
                
                process_metrics_task = {}

                global_metrics_task = {}
                for t in self.config.tasks:
                    # Sum confusion matrices to get global confusion matrix
                    num_classes = self.config.num_class_dict[t]
                    global_cm = gathered_cm[t].view(world_size, num_classes, num_classes).sum(dim=0)
                    
                    # Compute global metrics from the summed confusion matrix
                    confusion_matrix[t].mat = global_cm.cpu().numpy().astype(np.int64)
                    global_metrics_task[t] = confusion_matrix[t].compute()

                    process_metrics = []
                    for rank in range(world_size):
                        rank_cm = gathered_cm[t].view(world_size, num_classes, num_classes)[rank].cpu().numpy().astype(np.int64)
                        rank_cm_obj = ConfusionMatrix(num_classes, self.config.data.ignore_index)
                        rank_cm_obj.mat = rank_cm
                        if rank_cm.sum() > 0:  # Only compute if there are samples
                            rank_metrics = rank_cm_obj.compute()
                            process_metrics.append(rank_metrics)
                        else:
                            # Handle empty confusion matrix
                            process_metrics.append({
                                'pixel_accuracy': 0.0,
                                'mean_iou': 0.0,
                                'mean_accuracy': 0.0,
                                'fw_iou': 0.0,
                                'iou': np.zeros(num_classes)
                            })
                    process_metrics_task[t] = process_metrics

                # Compute global average loss (weighted by sample count)
                total_samples = all_sample_counts.sum().item()
                global_loss = all_total_losses.sum().item() / total_samples if total_samples > 0 else 0
                
                # Log individual process metrics
                for rank in range(world_size):
                    self.writer.add_scalar(f'Loss/val_rank_{rank}', all_avg_losses[rank].item(), epoch)
                    self.writer.add_scalar(f'Memory/val_cuda_memory_rank_{rank}', all_cuda_mem[rank].item(), epoch)

                    for t in self.config.tasks:
                        self.writer.add_scalar(f'Metrics/pixel_accuracy_rank_{rank}_task_{t}', process_metrics_task[t][rank]['pixel_accuracy'], epoch)
                        self.writer.add_scalar(f'Metrics/mean_iou_rank_{rank}_task_{t}', process_metrics_task[t][rank]['mean_iou'], epoch)
                    self.writer.add_scalar(f'Samples/val_count_rank_{rank}', all_sample_counts[rank].item(), epoch)
                
                # Log global (accurate) metrics
                self.writer.add_scalar('Loss/val_global', global_loss, epoch)
                
                try:
                    self.writer.add_histogram('Distributions/val_loss_per_rank', all_avg_losses.cpu(), epoch)
                except Exception as e: 
                    print(e)
                print(f"Validation Epoch: [{epoch + 1}] - Global metrics from {total_samples} samples across {world_size} processes")
                print(f"  Global Loss: {global_loss:.4f}")
                for t in self.config.tasks:
                    global_metrics = global_metrics_task[t]
                    self.writer.add_scalar(f'Metrics/pixel_accuracy_global_task_{t}', global_metrics['pixel_accuracy'], epoch)
                    self.writer.add_scalar(f'Metrics/mean_accuracy_global_task_{t}', global_metrics['mean_accuracy'], epoch)
                    self.writer.add_scalar(f'Metrics/mean_iou_global_task_{t}', global_metrics['mean_iou'], epoch)
                    self.writer.add_scalar(f'Metrics/fw_iou_global_task_{t}', global_metrics['fw_iou'], epoch)
                
                    # Log variance across processes
                    process_metrics = process_metrics_task[t]

                    process_miou = [m['mean_iou'] for m in process_metrics]
                    process_pixel_acc = [m['pixel_accuracy'] for m in process_metrics]
                
                    if len(process_miou) > 1:
                        miou_tensor = torch.tensor(process_miou)
                        pixel_acc_tensor = torch.tensor(process_pixel_acc)
                        
                        self.writer.add_scalar(f'Metrics/mean_iou_std_task_{t}', miou_tensor.std().item(), epoch)
                        self.writer.add_scalar(f'Metrics/mean_iou_min_task_{t}', miou_tensor.min().item(), epoch)
                        self.writer.add_scalar(f'Metrics/mean_iou_max_task_{t}', miou_tensor.max().item(), epoch)
                        
                        # Log histograms
                        try:
                            # self.writer.add_histogram('Distributions/val_loss_per_rank', all_avg_losses.cpu(), epoch)
                            self.writer.add_histogram(f'Distributions/val_miou_per_rank_task_{t}', miou_tensor, epoch)
                            self.writer.add_histogram(f'Distributions/val_pixel_acc_per_rank_task_{t}', pixel_acc_tensor, epoch)
                        except Exception as e: 
                            print(e)
                    # Log global per-class IoU
                    _, class_names = self.config.data.parse_color_and_names(t)
                    # class_names = self.config.data.class_names if hasattr(self.config.data, 'class_names') else None
                    global_metrics = global_metrics_task[t]
                    for i, iou in enumerate(global_metrics['iou']):
                        class_name = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
                        self.writer.add_scalar(f'IoU_per_class_global_task_{t}/{class_name}', iou, epoch)
                
                    # Create visualization comparing global vs per-process IoU
                    if len(process_metrics) > 1:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
                        
                        # Left plot: Global IoU per class
                        num_classes = len(global_metrics['iou'])
                        class_names_plot = class_names if class_names else [f"Class {i}" for i in range(num_classes)]
                        
                        ax1.bar(np.arange(num_classes), global_metrics['iou'])
                        ax1.set_xticks(np.arange(num_classes))
                        ax1.set_xticklabels(class_names_plot, rotation=45, ha='right')
                        ax1.set_ylim(0, 1.0)
                        ax1.set_ylabel('IoU')
                        ax1.set_title(f'Task: {t} - Global IoU per Class (Epoch {epoch+1})\nTotal Samples: {total_samples}')
                        
                        # Right plot: IoU variance across processes
                        x = np.arange(num_classes)
                        width = 0.8 / world_size
                        
                        for rank in range(world_size):
                            if rank < len(process_metrics):
                                offset = (rank - world_size/2 + 0.5) * width
                                rank_iou = process_metrics[rank].get('iou', np.zeros(num_classes))
                                ax2.bar(x + offset, rank_iou, width, 
                                    label=f'Rank {rank} ({all_sample_counts[rank].int()} samples)', alpha=0.8)
                        
                        ax2.set_xticks(x)
                        ax2.set_xticklabels(class_names_plot, rotation=45, ha='right')
                        ax2.set_ylim(0, 1.0)
                        ax2.set_ylabel('IoU')
                        ax2.set_title(f'Task: {t} - IoU per Class by Process (Epoch {epoch+1})')
                        ax2.legend()
                        
                        plt.tight_layout()
                        
                        # Save to TensorBoard
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=150)
                        buf.seek(0)
                        comparison_img = Image.open(buf)
                        plt.close(fig)
                        self.writer.add_image(f'Charts/IoU_global_vs_processes_task_{t}', 
                                            np.array(comparison_img).transpose(2, 0, 1), epoch)
                        buf.close()
                
                    # Print detailed results
                    
                    print(f"  Task {t} - Global Pixel Accuracy: {global_metrics['pixel_accuracy']:.4f}")
                    print(f"  Task {t} - Global Mean Accuracy: {global_metrics['mean_accuracy']:.4f}")
                    print(f"  Task {t} - Global Mean IoU: {global_metrics['mean_iou']:.4f}")
                    print(f"  Task {t} - Global Freq Weighted IoU: {global_metrics['fw_iou']:.4f}")
                    
                    if len(process_miou) > 1:
                        print(f"  Task {t} - Process mIoU variance: std={torch.tensor(process_miou).std():.4f}, "
                            f"min={min(process_miou):.4f}, max={max(process_miou):.4f}")
                    
                    print(f"  Task {t} - Global IoU per class:")
                    
                    if t == 'bdd100k':
                        self.bdd_pseudo_enable_class_indices = []
                    for i, iou in enumerate(global_metrics['iou']):
                        class_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
                        if t == 'bdd100k':
                            if iou > self.config.pseudo_label_threshold and class_names[i] != 'road':
                                self.bdd_pseudo_enable_class_indices.append(i)
                        print(f"    {class_name}: {iou:.4f}")
                    if t == 'bdd100k':
                        arr = np.array(self.bdd_pseudo_enable_class_indices)
                        encoded_arr = np.zeros((self.config.num_class_dict['bdd100k']), dtype=int)
                        encoded_arr[arr] = 1
                        self.sup_pseudo_enable_class_mask = self.config.dataset_prediction_mapping['bdd100k'].T @ encoded_arr.reshape((self.config.num_class_dict['bdd100k'], 1))
                    
                        if global_metrics['mean_iou'] < self.config.pseudo_label_miou_threshold:
                            self.sup_pseudo_enable_class_mask = torch.zeros((self.config.data.num_classes, 1))

                output_metrics = {'val_loss': global_loss}

                for t in self.config.tasks:
                    output_metrics[f'pixel_accuracy_{t}'] = global_metrics_task[t]['pixel_accuracy']
                    output_metrics[f'mean_accuracy_{t}'] = global_metrics_task[t]['mean_accuracy']
                    output_metrics[f'mean_iou_{t}'] = global_metrics_task[t]['mean_iou']
                    output_metrics[f'fw_iou_{t}'] = global_metrics_task[t]['fw_iou']

                

                # Return global metrics for checkpoint saving
                return output_metrics
            else:
                # Non-main processes return dummy metrics (won't be used for checkpointing)
                return {
                    'val_loss': loss_meter.avg,
                    'pixel_accuracy': 0.0,
                    'mean_accuracy': 0.0,
                    'mean_iou': 0.0,
                    'fw_iou': 0.0
                }
        else:
            # Single process - compute metrics normally
            output_metrics = {'val_loss': loss_meter.avg}
            if is_main_process() and self.writer:
                self.writer.add_scalar('Loss/val', loss_meter.avg, epoch)
                print(f"Validation Epoch: [{epoch + 1}]")
                print(f"  Loss: {loss_meter.avg:.4f}")
            for t in self.config.tasks:
                metrics = confusion_matrix[t].compute()
                # print(metrics)
                if is_main_process() and self.writer:
                    self.writer.add_scalar(f'Metrics/pixel_accuracy_task_{t}', metrics['pixel_accuracy'], epoch)
                    self.writer.add_scalar(f'Metrics/mean_accuracy_task_{t}', metrics['mean_accuracy'], epoch)
                    self.writer.add_scalar(f'Metrics/mean_iou_task_{t}', metrics['mean_iou'], epoch)
                    self.writer.add_scalar(f'Metrics/fw_iou_task_{t}', metrics['fw_iou'], epoch)
                    
                    # Print detailed results
                    print(f"  Task {t} - Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
                    print(f"  Task {t} - Mean Accuracy: {metrics['mean_accuracy']:.4f}")
                    print(f"  Task {t} - Mean IoU: {metrics['mean_iou']:.4f}")
                    print(f"  Task {t} - Freq Weighted IoU: {metrics['fw_iou']:.4f}")

                    # Log per-class IoU
                    print(f"  Task {t} - IoU per class:")
                    # class_names = self.config.data.class_names if hasattr(self.config.data, 'class_names') else None
                    _, class_names = self.config.data.parse_color_and_names(t)
                    
                    if t == 'bdd100k':
                        self.bdd_pseudo_enable_class_indices = []
                        
                    for i, iou in enumerate(metrics['iou']):
                        class_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
                        if t == 'bdd100k':
                            if iou > self.config.pseudo_label_threshold and class_names[i] != 'road':
                                self.bdd_pseudo_enable_class_indices.append(i)
                        print(f"    {class_name}: {iou:.4f}")

                        self.writer.add_scalar(f'IoU_per_class_task_{t}/{class_name}', iou, epoch)
                    
                    if t == 'bdd100k':
                        arr = np.array(self.bdd_pseudo_enable_class_indices)
                        encoded_arr = np.zeros((self.config.num_class_dict['bdd100k']), dtype=int)
                        encoded_arr[arr] = 1
                        self.sup_pseudo_enable_class_mask = self.config.dataset_prediction_mapping['bdd100k'].T @ encoded_arr.reshape((self.config.num_class_dict['bdd100k'], 1))
                        # print(self.sup_pseudo_enable_class_mask)
                        if metrics['mean_iou'] < self.config.pseudo_label_miou_threshold:
                            self.sup_pseudo_enable_class_mask = torch.zeros((self.config.data.num_classes, 1))
                            
                output_metrics[f'pixel_accuracy_{t}'] = metrics['pixel_accuracy']
                output_metrics[f'mean_accuracy_{t}'] = metrics['mean_accuracy']
                output_metrics[f'mean_iou_{t}'] = metrics['mean_iou']
                output_metrics[f'fw_iou_{t}'] = metrics['fw_iou']
            
            return output_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Save a checkpoint of the model (only on main process).
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            metrics: Dictionary of validation metrics to save
        """
        if not is_main_process():
            return
            
        # Create checkpoint directory
        checkpoint_dir = self.output_dir / "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Get model state dict (handle DDP or DP)
        if hasattr(self.model, 'module'):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        # Create checkpoint with comprehensive information
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_miou': self.best_miou,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'config': self.config,
            'metrics_history': self.metrics_history,
            'use_amp': self.use_amp,
        }
        
        # Add scheduler state if available
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Add current metrics if provided
        if metrics:
            checkpoint['current_metrics'] = metrics

        if self.ema_model:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()
            
        # if self.sup_pseudo_enable_class_mask:
        checkpoint['sup_pseudo_enable_class_mask'] = self.sup_pseudo_enable_class_mask
        # print(checkpoint['sup_pseudo_enable_class_mask'])   
        # Format checkpoint path
        if True: ## Set to True to save all checkpoints for all epoches
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(checkpoint, checkpoint_path)
        # If this is the best model, create a copy
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model checkpoint to {best_path}")
            
            # Also save a text file with the best metrics for quick reference
            if metrics:
                best_metrics_path = self.output_dir / "best_metrics.txt"
                with open(best_metrics_path, 'w') as f:
                    f.write(f"Best model (epoch {epoch + 1}):\n")
                    for k, v in metrics.items():
                        f.write(f"{k}: {v}\n")
                    f.write(f"use_amp: {self.use_amp}\n")
        
        # Save latest checkpoint (for easy resuming)
        latest_path = checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)