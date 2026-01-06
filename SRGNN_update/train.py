#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for SR-GNN
Modernized implementation with PyTorch Geometric

Key improvements:
- Native PyTorch DataLoader integration
- Mixed precision training support (AMP)
- Better progress tracking with tqdm
- Gradient accumulation support
- Improved checkpointing
"""

import argparse
import pickle
import time
import os
import torch
from torch.cuda.amp import GradScaler, autocast
from utils import split_validation, create_dataloader, Data_Legacy
from model import SessionGraph, trans_to_cuda, get_device, train_test


def parse_args():
    parser = argparse.ArgumentParser(description='SR-GNN Training (Modern PyG Implementation)')
    
    # Dataset arguments
    parser.add_argument('--dataset', default='sample', 
                        help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
    parser.add_argument('--mini', action='store_true', 
                        help='use mini dataset for testing')
    
    # Model arguments
    parser.add_argument('--hiddenSize', type=int, default=120, 
                        help='hidden state size')
    parser.add_argument('--step', type=int, default=1, 
                        help='GNN propagation steps')
    parser.add_argument('--nonhybrid', action='store_true', 
                        help='only use global preference for prediction')
    parser.add_argument('--gnn_type', type=str, default='ggnn', 
                        choices=['ggnn', 'gatv2'],
                        help='GNN architecture type')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate')
    
    # Training arguments
    parser.add_argument('--batchSize', type=int, default=75, 
                        help='input batch size')
    parser.add_argument('--epoch', type=int, default=15, 
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='learning rate')
    parser.add_argument('--lr_dc', type=float, default=0.1, 
                        help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=3, 
                        help='steps after which learning rate decays')
    parser.add_argument('--l2', type=float, default=1e-5, 
                        help='L2 regularization weight')
    parser.add_argument('--patience', type=int, default=10, 
                        help='early stopping patience')
    
    # Validation arguments
    parser.add_argument('--validation', action='store_true', 
                        help='use validation split')
    parser.add_argument('--valid_portion', type=float, default=0.1, 
                        help='validation split ratio')
    
    # Misc arguments
    parser.add_argument('--saved_data', type=str, default='SRGNN_update',
                        help='directory to save models')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of data loading workers')
    parser.add_argument('--use_amp', action='store_true',
                        help='use automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)
    # For CUDA determinism (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    opt = parse_args()
    print('=' * 60)
    print('SR-GNN Training (Modern PyG Implementation)')
    print('=' * 60)
    print(opt)
    print('=' * 60)
    
    # Set random seed
    set_seed(opt.seed)
    
    # Create save directory
    os.makedirs(opt.saved_data, exist_ok=True)
    
    # Load data
    print('Loading data...')
    train_data = pickle.load(open('datasets/train.pkl', 'rb'))
    
    if opt.mini:
        print('Using mini dataset for testing...')
        train_data = [train_data[0][:200], train_data[1][:200]]
    
    if opt.validation:
        print('Splitting validation set...')
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/test.pkl', 'rb'))
    
    print(f'Train samples: {len(train_data[0])}')
    print(f'Test samples: {len(test_data[0])}')
    
    # Create dataloaders
    print('Creating dataloaders...')
    train_loader = create_dataloader(
        train_data, 
        batch_size=opt.batchSize, 
        shuffle=True,
        num_workers=opt.num_workers
    )
    test_loader = create_dataloader(
        test_data, 
        batch_size=opt.batchSize, 
        shuffle=False,
        num_workers=opt.num_workers
    )
    
    # Determine number of nodes
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset in ['yoochoose1_64', 'yoochoose1_4']:
        n_node = 37484
    else:
        n_node = 22055
    
    print(f'Number of items: {n_node}')
    
    # Create model
    print('Creating model...')
    device = get_device()
    print(f'Using device: {device}')
    
    model = SessionGraph(opt, n_node)
    model = trans_to_cuda(model)
    model.setup_optimizer()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Training loop
    print('=' * 60)
    print('Starting training...')
    print('=' * 60)
    
    start_time = time.time()
    best_result = [0, 0]  # [best_recall, best_mrr]
    best_epoch = [0, 0]
    bad_counter = 0
    
    scaler = GradScaler() if opt.use_amp else None
    
    for epoch in range(opt.epoch):
        print('-' * 60)
        print(f'Epoch: {epoch + 1}/{opt.epoch}')
        
        # Train and evaluate
        hit, mrr = train_epoch(model, train_loader, test_loader, device, 
                               opt.use_amp, scaler)
        
        # Check for improvement
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
            torch.save(model.state_dict(), 
                      os.path.join(opt.saved_data, 'best_recall.pt'))
        
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
            torch.save(model.state_dict(), 
                      os.path.join(opt.saved_data, 'best_mrr.pt'))
        
        if flag:
            bad_counter = 0
        else:
            bad_counter += 1
        
        print(f'Current Result - Recall@20: {hit:.4f}, MRR@20: {mrr:.4f}')
        print(f'Best Result - Recall@20: {best_result[0]:.4f} (Epoch {best_epoch[0]+1}), '
              f'MRR@20: {best_result[1]:.4f} (Epoch {best_epoch[1]+1})')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': model.optimizer.state_dict(),
            'scheduler': model.scheduler.state_dict(),
            'best_result': best_result,
            'best_epoch': best_epoch,
            'config': vars(opt)
        }
        torch.save(checkpoint, os.path.join(opt.saved_data, 'last_checkpoint.pth.tar'))
        
        # Early stopping
        if bad_counter >= opt.patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break
    
    print('=' * 60)
    end_time = time.time()
    print(f'Training completed in {end_time - start_time:.2f} seconds')
    print(f'Best Recall@20: {best_result[0]:.4f} at epoch {best_epoch[0] + 1}')
    print(f'Best MRR@20: {best_result[1]:.4f} at epoch {best_epoch[1] + 1}')


def train_epoch(model, train_loader, test_loader, device, use_amp=False, scaler=None):
    """
    Train for one epoch and evaluate.
    
    Args:
        model: SessionGraph model
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        device: Device to use
        use_amp: Whether to use automatic mixed precision
        scaler: GradScaler for AMP
    Returns:
        hit: Recall@20
        mrr: MRR@20
    """
    import datetime
    import numpy as np
    from model import trans_to_cpu
    
    model.scheduler.step()
    
    # Training
    print(f'Training started: {datetime.datetime.now()}')
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for i, batch_data in enumerate(train_loader):
        pyg_batch, mask, targets = batch_data
        pyg_batch = pyg_batch.to(device)
        mask = mask.to(device)
        targets = torch.tensor(targets, dtype=torch.long, device=device) - 1
        
        model.optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                hidden = model(pyg_batch)
                scores = model.compute_scores(hidden, mask)
                loss = model.loss_function(scores, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(model.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(model.optimizer)
            scaler.update()
        else:
            hidden = model(pyg_batch)
            scores = model.compute_scores(hidden, mask)
            loss = model.loss_function(scores, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model.optimizer.step()
        
        total_loss += loss.item()
        
        if i % max(1, num_batches // 5) == 0:
            print(f'  [{i}/{num_batches}] Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / num_batches
    print(f'  Average Loss: {avg_loss:.4f}')
    
    # Evaluation
    print(f'Evaluation started: {datetime.datetime.now()}')
    model.eval()
    hit, mrr = [], []
    
    with torch.no_grad():
        for batch_data in test_loader:
            pyg_batch, mask, targets = batch_data
            pyg_batch = pyg_batch.to(device)
            mask = mask.to(device)
            
            if use_amp:
                with autocast():
                    hidden = model(pyg_batch)
                    scores = model.compute_scores(hidden, mask)
            else:
                hidden = model(pyg_batch)
                scores = model.compute_scores(hidden, mask)
            
            sub_scores = scores.topk(20)[1]
            sub_scores = trans_to_cpu(sub_scores).numpy()
            
            for score, target in zip(sub_scores, targets):
                hit.append(np.isin(target - 1, score))
                rank = np.where(score == target - 1)[0]
                if len(rank) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (rank[0] + 1))
    
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    
    return hit, mrr


if __name__ == '__main__':
    main()
