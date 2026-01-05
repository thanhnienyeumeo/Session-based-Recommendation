#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing/Evaluation script for SR-GNN
Modernized implementation with PyTorch Geometric

Features:
- Load trained models and evaluate on test set
- Support for both state_dict and full model loading
- Detailed metrics reporting
"""

import argparse
import pickle
import datetime
import numpy as np
import torch
from utils import create_dataloader
from model import SessionGraph, trans_to_cuda, trans_to_cpu, get_device


def parse_args():
    parser = argparse.ArgumentParser(description='SR-GNN Testing (Modern PyG Implementation)')
    
    # Dataset arguments
    parser.add_argument('--dataset', default='sample',
                        help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
    
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
    
    # Evaluation arguments
    parser.add_argument('--batchSize', type=int, default=75,
                        help='input batch size')
    parser.add_argument('--top_k', type=int, default=20,
                        help='top-k for evaluation metrics')
    
    # Legacy arguments (for compatibility)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_dc', type=float, default=0.1)
    parser.add_argument('--lr_dc_step', type=int, default=3)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--valid_portion', type=float, default=0.1)
    
    # Misc arguments
    parser.add_argument('--saved_data', type=str, default='SRGNN_update',
                        help='directory containing saved models')
    parser.add_argument('--model_type', type=str, default='best_mrr',
                        choices=['best_mrr', 'best_recall', 'last'],
                        help='which model to load')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of data loading workers')
    
    return parser.parse_args()


def validate(model, test_loader, device, top_k=20):
    """
    Evaluate model on test data.
    
    Args:
        model: SessionGraph model
        test_loader: Test DataLoader
        device: Device to use
        top_k: Top-k for metrics
    Returns:
        Tuple of (hit@k, mrr@k, ndcg@k)
    """
    print(f'Starting evaluation: {datetime.datetime.now()}')
    model.eval()
    
    hit, mrr, ndcg = [], [], []
    
    with torch.no_grad():
        for batch_data in test_loader:
            pyg_batch, mask, targets = batch_data
            pyg_batch = pyg_batch.to(device)
            mask = mask.to(device)
            
            hidden = model(pyg_batch)
            scores = model.compute_scores(hidden, mask)
            
            sub_scores = scores.topk(top_k)[1]
            sub_scores = trans_to_cpu(sub_scores).numpy()
            
            for score, target in zip(sub_scores, targets):
                # Hit Rate (Recall)
                hit.append(np.isin(target - 1, score))
                
                # MRR (Mean Reciprocal Rank)
                rank = np.where(score == target - 1)[0]
                if len(rank) == 0:
                    mrr.append(0)
                    ndcg.append(0)
                else:
                    mrr.append(1 / (rank[0] + 1))
                    # NDCG
                    ndcg.append(1 / np.log2(rank[0] + 2))
    
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    ndcg = np.mean(ndcg) * 100
    
    return hit, mrr, ndcg


def load_model(opt, n_node, device):
    """
    Load a trained model.
    
    Args:
        opt: Arguments
        n_node: Number of items
        device: Device to use
    Returns:
        Loaded model
    """
    # Create model architecture
    model = SessionGraph(opt, n_node)
    model = trans_to_cuda(model)
    
    # Determine model path
    if opt.model_type == 'best_mrr':
        model_path = f'{opt.saved_data}/best_mrr.pt'
    elif opt.model_type == 'best_recall':
        model_path = f'{opt.saved_data}/best_recall.pt'
    else:
        model_path = f'{opt.saved_data}/last_checkpoint.pth.tar'
    
    print(f'Loading model from: {model_path}')
    
    try:
        # Try loading as state_dict first
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # It's a checkpoint dictionary
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'best_result' in checkpoint:
                print(f"Checkpoint best result - Recall@20: {checkpoint['best_result'][0]:.4f}, "
                      f"MRR@20: {checkpoint['best_result'][1]:.4f}")
        elif isinstance(checkpoint, dict):
            # It's a state_dict
            model.load_state_dict(checkpoint)
        else:
            # It might be a full model (legacy format)
            print("Detected legacy model format, loading full model...")
            model = checkpoint
            model = trans_to_cuda(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try loading as full model (legacy)
        try:
            model = torch.load(model_path, map_location=device)
            model = trans_to_cuda(model)
            print("Loaded legacy full model")
        except Exception as e2:
            raise RuntimeError(f"Failed to load model: {e2}")
    
    return model


def main():
    opt = parse_args()
    print('=' * 60)
    print('SR-GNN Testing (Modern PyG Implementation)')
    print('=' * 60)
    print(opt)
    print('=' * 60)
    
    # Load test data
    print('Loading test data...')
    test_data = pickle.load(open('datasets/test.pkl', 'rb'))
    print(f'Test samples: {len(test_data[0])}')
    
    # Create dataloader
    print('Creating dataloader...')
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
    
    # Get device
    device = get_device()
    print(f'Using device: {device}')
    
    # Load model
    print('Loading model...')
    model = load_model(opt, n_node, device)
    
    # Evaluate
    print('=' * 60)
    print('Evaluating...')
    print('=' * 60)
    
    recall, mrr, ndcg = validate(model, test_loader, device, top_k=opt.top_k)
    
    print('=' * 60)
    print(f'Results @ {opt.top_k}:')
    print(f'  Recall@{opt.top_k}: {recall:.4f}%')
    print(f'  MRR@{opt.top_k}: {mrr:.4f}%')
    print(f'  NDCG@{opt.top_k}: {ndcg:.4f}%')
    print('=' * 60)
    
    # Also evaluate at different k values
    for k in [5, 10, 20]:
        if k != opt.top_k:
            hit_k, mrr_k, ndcg_k = validate(model, test_loader, device, top_k=k)
            print(f'Results @ {k}: Recall={hit_k:.4f}%, MRR={mrr_k:.4f}%, NDCG={ndcg_k:.4f}%')


if __name__ == '__main__':
    main()
