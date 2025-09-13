import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler
from typing import Tuple, Optional, Dict, Any, Union

from config.base import Config, DataConfig
from .dataset import get_dataset

def get_data_loaders(
    config: Config,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config: Complete configuration
        distributed: Whether to use distributed sampling
        rank: Process rank for distributed training
        world_size: Total number of processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = get_dataset(config.data, split="train")
    val_dataset = get_dataset(config.data, split="val")
    test_dataset = get_dataset(config.data, split="test")
    
    # Create samplers
    train_sampler = None
    val_sampler = None
    test_sampler = None
    
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.seed,
            drop_last=True,
        )
        
        if val_dataset is not None:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=True,
            )
        
        if test_dataset is not None:
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
    # else:
    #     train_sampler = None


    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        prefetch_factor=config.data.prefetch_factor,
        # drop_last=config.data.drop_last,
        drop_last=True,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            sampler=val_sampler,
            # shuffle=True,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers,
            prefetch_factor=config.data.prefetch_factor,
            # drop_last=False,
            drop_last=True,
            collate_fn=val_dataset.collate_fn
        )
    
    test_loader = None
    if test_dataset is not None:
       test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers,
            prefetch_factor=config.data.prefetch_factor,
            drop_last=False,
            collate_fn=test_dataset.collate_fn
        )
    
    return train_loader, val_loader, test_loader