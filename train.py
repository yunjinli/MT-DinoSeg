import os
import random
import numpy as np
import torch
torch.hub.set_dir('/home/phd_li/.cache/torch/hub')
os.environ['HF_HOME'] = '/home/phd_li/.cache/huggingface'
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from torchinfo import summary
from config import load_config_from_args, ModelRegistry, EncoderRegistry, DecoderRegistry, Config
from models import create_model
from data import get_data_loaders
from trainer import Trainer, MultiDatasetTrainer, MultiDatasetCrossSupTrainer
from utility.distributed import (
    setup_distributed, 
    cleanup_distributed, 
    is_main_process,
    get_rank,
    get_world_size,
    init_seeds
)
import argparse
from typing import Optional
import signal
import sys

def signal_handler(signum, frame):
    """Handle termination signals for graceful shutdown."""
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    cleanup_distributed()
    sys.exit(0)

def load_checkpoint_config(checkpoint_path: str) -> Optional[Config]:
    """Load configuration from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'config' in checkpoint:
            return checkpoint['config']
        else:
            print(f"Warning: No config found in checkpoint {checkpoint_path}")
            return None
    except Exception as e:
        print(f"Error loading config from checkpoint: {e}")
        return None

def main():
    """Main training function."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup distributed training
    device, rank, world_size, local_rank = setup_distributed()
    
    try:
        config, args = load_config_from_args()

        # Handle resume options
        resume_training = getattr(args, 'resume', False)
        resume_from = getattr(args, 'resume_from', None)
        resume_run_id = getattr(args, 'resume_run_id', None)

        # Check if we need to find a checkpoint to resume from
        if resume_training or resume_run_id:
            # If resuming from a specific run, find the latest checkpoint
            if resume_run_id and not resume_from:
                experiment_dir = Path(f"outputs/{config.experiment_name}")
                run_dir = experiment_dir / f"run_{resume_run_id}"
                if run_dir.exists():
                    checkpoints_dir = run_dir / "checkpoints"
                    if checkpoints_dir.exists():
                        latest_checkpoint = checkpoints_dir / "latest.pth"
                        if latest_checkpoint.exists():
                            resume_from = str(latest_checkpoint)
                            resume_training = True
                        else:
                            if is_main_process():
                                print(f"No checkpoint found in {checkpoints_dir}")
                            else:
                                if is_main_process():
                                    print(f"No checkpoints directory found for run {resume_run_id}")
                else:
                    if is_main_process():
                        print(f"Run {resume_run_id} not found")
        
        # Load configuration
        if resume_from:
            # Load config from checkpoint
            if is_main_process():
                print(f"Loading configuration from checkpoint: {resume_from}")
            config = load_checkpoint_config(resume_from)
            if config is None:
                if is_main_process():
                    print("Failed to load config from checkpoint, falling back to command line config")
                config, _ = load_config_from_args()
            else:
                if is_main_process():
                    print("Successfully loaded original configuration from checkpoint")
                # Apply any command line overrides
                if getattr(args, 'learning_rate', None) is not None:
                    config.training.optimizer.learning_rate = args.learning_rate
                    if is_main_process():
                        print(f"Overriding learning rate to {args.learning_rate}")
                if getattr(args, 'batch_size', None) is not None:
                    config.training.batch_size = args.batch_size
                    if is_main_process():
                        print(f"Overriding batch size to {args.batch_size}")
                if getattr(args, 'epochs', None) is not None:
                    config.training.epochs = args.epochs
                    if is_main_process():
                        print(f"Overriding epochs to {args.epochs}")
                if getattr(args, 'freeze', False):
                    config.model.encoder.freeze = True
                    if is_main_process():
                        print("Overriding encoder freeze to True")
        else:
            # Load config normally
            config, _ = load_config_from_args()

        # Extract run_id from checkpoint path if resuming
        run_id = getattr(args, 'run_id', None)
        if resume_from and not run_id:
            checkpoint_path = Path(resume_from)
            # Extract run_id from path (e.g., .../run_20250518_123045/checkpoints/latest.pth)
            run_dir = checkpoint_path.parent.parent
            if run_dir.name.startswith("run_"):
                run_id = run_dir.name[4:]  # Remove "run_" prefix
                if is_main_process():
                    print(f"Extracted run_id from checkpoint path: {run_id}")
        
        # Generate run_id if not resuming and not provided
        if not resume_training and run_id is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{timestamp}_rank{rank}"
        
        # Set random seeds for reproducibility (different for each rank)
        init_seeds(config.seed, rank)

        if is_main_process():
            print(f"Starting experiment: {config.experiment_name}")
            print(f"Starting experiment: {config.model.name}")
            print(f"Distributed training: {world_size} GPU(s), rank {rank}")
            if resume_training or resume_from:
                print(f"Resuming training from: {resume_from or 'latest checkpoint'}")
                print(f"Using original configuration from checkpoint")
            else:
                print(f"Run ID: {run_id}")

            print(f"Encoder: {config.model.encoder.name}, Freeze: {config.model.encoder.freeze}")
            print(f"Decoder: {config.model.decoder.name}, Output dim: {config.model.decoder.num_classes}")
            print(f"Dataset: {config.data.dataset_name}, Number of classes: {config.data.num_classes}")
            print(f"Training for {config.training.epochs} epochs with lr={config.training.optimizer.learning_rate}")
        
        # Create the model
        model = create_model(config)
        model = model.to(device)
        
        # Print model summary only on main process
        if is_main_process():
            try:
                summary(model, depth=8)
                print("Model created successfully!")
                print(f"Encoder output dimension: {model.encoder.get_output_dim()}")
                print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
                print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            except Exception as e:
                print(f"Could not print model summary: {e}")

        # Wrap model with DDP if distributed
        if world_size > 1:
            # Find unused parameters automatically
            model = DDP(
                model, 
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=True,  # All parameters are used
                gradient_as_bucket_view=True   # Optimize gradient memory layout
            )
            if is_main_process():
                print("Model wrapped with DistributedDataParallel")

        # Get data loaders with distributed support
        # train_loader, _, val_loader = get_data_loaders(
        train_loader, val_loader, _ = get_data_loaders(
            config, 
            distributed=(world_size > 1),
            rank=rank,
            world_size=world_size
        )
        
        # Print training info only on main process
        if is_main_process():
            print(f"Training configuration:")
            print(f"- Experiment: {config.experiment_name}")
            print(f"- Model: {config.model.encoder.name} encoder with {config.model.decoder.name}")
            print(f"- Device: {device}")
            print(f"- World size: {world_size}")
            print(f"- Rank: {rank}")
            print(f"- Batch size per GPU: {config.training.batch_size}")
            print(f"- Effective batch size: {config.training.batch_size * world_size}")
            print(f"- Learning rate: {config.training.optimizer.learning_rate}")
            print(f"- Epochs: {config.training.epochs}")
            print(f"- Encoder freeze: {config.model.encoder.freeze}")
            print(f"- Number of training samples: {len(train_loader.dataset)}")
            print(f"- Number of validation samples: {len(val_loader.dataset)}")
        # Create trainer
        if config.multidataset:
            if config.cross_task_supervision:
                trainer = MultiDatasetCrossSupTrainer(
                    model=model,
                    config=config,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    run_id=run_id,
                    resume_from=resume_from,
                    resume_training=resume_training,
                    rank=rank,
                    world_size=world_size,
                )
            else:
                trainer = MultiDatasetTrainer(
                    model=model,
                    config=config,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    run_id=run_id,
                    resume_from=resume_from,
                    resume_training=resume_training,
                    rank=rank,
                    world_size=world_size
                )
        else:
            trainer = Trainer(
                model=model,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                run_id=run_id,
                resume_from=resume_from,
                resume_training=resume_training,
                rank=rank,
                world_size=world_size
            )
        
        # Run training
        train_stats = trainer.train()
        
        # Print final results only on main process
        if is_main_process():
            print(f"Training complete!")
            if train_stats.get('resumed_from_epoch', 1) > 1:
                print(f"Resumed from epoch {train_stats['resumed_from_epoch']}")
            print(f"Best validation loss: {train_stats['best_val_loss']:.4f} (epoch {train_stats['best_epoch'] + 1})")
            print(f"Best mean IoU: {train_stats['best_miou']:.4f} (epoch {train_stats['best_epoch'] + 1})")
            print(f"Training time: {train_stats['training_time'] / 60:.2f} minutes")

            # If there are final metrics, report them
            if 'final_metrics' in train_stats and train_stats['final_metrics']:
                print("\nFinal metrics:")
                metrics = train_stats['final_metrics']
                print(f"  Loss: {metrics.get('val_loss', 'N/A'):.4f}")
                print(f"  Pixel Accuracy: {metrics.get('pixel_accuracy', 'N/A'):.4f}")
                print(f"  Mean IoU: {metrics.get('mean_iou', 'N/A'):.4f}")
    
    finally:
        # Clean up distributed training
        cleanup_distributed()

if __name__ == "__main__":
    main()