#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SR-GNN: Session-based Recommendation with Graph Neural Networks
Modernized implementation using PyTorch Geometric (2024+)

Original paper: Wu et al., "Session-based Recommendation with Graph Neural Networks", AAAI 2019
"""

import math
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, GATv2Conv, MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch


class ModernGNN(nn.Module):
    """
    Modern GNN implementation using PyTorch Geometric's optimized layers.
    Supports multiple GNN variants: GGNN (original), GAT, and GATv2.
    """
    def __init__(self, hidden_size, num_layers=1, gnn_type='ggnn', heads=4, dropout=0.1):
        super(ModernGNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        
        if gnn_type == 'ggnn':
            # GatedGraphConv is the modern equivalent of the original SR-GNN's GNN
            # It uses GRU-like gating mechanism with optimized sparse operations
            self.gnn = GatedGraphConv(
                out_channels=hidden_size,
                num_layers=num_layers,
                aggr='add'
            )
        elif gnn_type == 'gatv2':
            # GATv2 is a more powerful attention-based GNN
            self.gnn_layers = nn.ModuleList()
            for i in range(num_layers):
                self.gnn_layers.append(
                    GATv2Conv(
                        in_channels=hidden_size,
                        out_channels=hidden_size // heads,
                        heads=heads,
                        dropout=dropout,
                        concat=True,
                        add_self_loops=False
                    )
                )
            self.dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, hidden_size]
            edge_index: Edge indices [2, num_edges] in COO format
        Returns:
            Updated node features [num_nodes, hidden_size]
        """
        if self.gnn_type == 'ggnn':
            return self.gnn(x, edge_index)
        else:
            for i, layer in enumerate(self.gnn_layers):
                x = layer(x, edge_index)
                if i < len(self.gnn_layers) - 1:
                    x = F.elu(x)
                    x = self.dropout(x)
            return x


class SessionGraph(nn.Module):
    """
    Session-based Recommendation with Graph Neural Networks.
    
    Modernized implementation with:
    - PyTorch Geometric for efficient GNN operations
    - Sparse edge representation instead of dense adjacency matrices
    - Support for multiple GNN architectures
    - Mixed precision training support
    - Improved attention mechanism
    """
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.gnn_type = getattr(opt, 'gnn_type', 'ggnn')
        self.dropout_rate = getattr(opt, 'dropout', 0.1)
        
        # Item embedding layer
        self.embedding = nn.Embedding(self.n_node, self.hidden_size, padding_idx=0)
        
        # Modern GNN layers
        self.gnn = ModernGNN(
            hidden_size=self.hidden_size,
            num_layers=opt.step,
            gnn_type=self.gnn_type,
            dropout=self.dropout_rate
        )
        
        # Attention mechanism for session representation
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Loss function
        self.loss_function = nn.CrossEntropyLoss()
        
        # Optimizer and scheduler (will be set up after model is on device)
        self.optimizer = None
        self.scheduler = None
        self._opt = opt
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, param in self.named_parameters():
            if 'embedding' in name:
                param.data.uniform_(-stdv, stdv)
            elif 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def setup_optimizer(self):
        """Set up optimizer and scheduler after model is moved to device."""
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self._opt.lr,
            weight_decay=self._opt.l2
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self._opt.lr_dc_step,
            gamma=self._opt.lr_dc
        )
    
    def compute_scores(self, hidden, mask):
        """
        Compute prediction scores using attention mechanism.
        
        Args:
            hidden: Session item embeddings [batch_size, seq_len, hidden_size]
            mask: Valid item mask [batch_size, seq_len]
        Returns:
            scores: Prediction scores [batch_size, n_items]
        """
        # Get the last item representation (target item)
        seq_lens = mask.sum(dim=1) - 1
        batch_indices = torch.arange(mask.shape[0], device=hidden.device)
        ht = hidden[batch_indices, seq_lens]  # [batch_size, hidden_size]
        
        # Soft attention over all session items
        q1 = self.linear_one(ht).unsqueeze(1)  # [batch_size, 1, hidden_size]
        q2 = self.linear_two(hidden)  # [batch_size, seq_len, hidden_size]
        alpha = self.linear_three(torch.sigmoid(q1 + q2))  # [batch_size, seq_len, 1]
        
        # Apply mask and compute weighted sum
        mask_float = mask.unsqueeze(-1).float()
        a = torch.sum(alpha * hidden * mask_float, dim=1)  # [batch_size, hidden_size]
        
        # Hybrid representation
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], dim=1))
        
        # Apply layer normalization and dropout
        a = self.layer_norm(a)
        a = self.dropout(a)
        
        # Compute scores against all items (excluding padding)
        b = self.embedding.weight[1:]  # [n_items, hidden_size]
        scores = torch.matmul(a, b.transpose(0, 1))  # [batch_size, n_items]
        
        return scores
    
    def forward(self, data):
        """
        Forward pass through the model.
        
        Args:
            data: PyG Batch object containing:
                - x: Node indices [num_nodes]
                - edge_index: Edge indices [2, num_edges]
                - batch: Batch assignment [num_nodes]
                - sequence: Original sequence indices for each session
                - mask: Valid item mask
        Returns:
            hidden: Session item embeddings [batch_size, seq_len, hidden_size]
        """
        # Get node embeddings
        x = self.embedding(data.x)  # [num_nodes, hidden_size]
        
        # Apply GNN
        x = self.gnn(x, data.edge_index)  # [num_nodes, hidden_size]
        
        # Convert back to dense batch format
        hidden, _ = to_dense_batch(x, data.batch)  # [batch_size, max_nodes, hidden_size]
        
        # Reorder according to original sequence
        batch_size = data.num_graphs
        seq_hidden = []
        
        for i in range(batch_size):
            # Get node mask for this graph
            node_mask = (data.batch == i)
            graph_hidden = x[node_mask]  # [num_nodes_i, hidden_size]
            
            # Get unique items for this graph
            graph_items = data.x[node_mask]
            
            # Get the sequence for this graph
            seq = data.sequence[i]
            
            # Map sequence items to their node indices
            seq_emb = []
            for item in seq:
                if item == 0:
                    seq_emb.append(torch.zeros(self.hidden_size, device=x.device))
                else:
                    # Find the index of this item in graph_items
                    idx = (graph_items == item).nonzero(as_tuple=True)[0]
                    if len(idx) > 0:
                        seq_emb.append(graph_hidden[idx[0]])
                    else:
                        seq_emb.append(self.embedding(torch.tensor(item, device=x.device)))
            
            seq_hidden.append(torch.stack(seq_emb))
        
        hidden = torch.stack(seq_hidden)  # [batch_size, seq_len, hidden_size]
        
        return hidden


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def trans_to_cuda(variable):
    """Transfer variable to the best available device."""
    device = get_device()
    if isinstance(variable, nn.Module):
        return variable.to(device)
    elif isinstance(variable, torch.Tensor):
        return variable.to(device)
    else:
        return variable


def trans_to_cpu(variable):
    """Transfer variable to CPU."""
    if isinstance(variable, torch.Tensor):
        return variable.cpu()
    else:
        return variable


def forward(model, batch_data):
    """
    Forward pass for a batch of data.
    
    Args:
        model: SessionGraph model
        batch_data: Tuple of (pyg_batch, mask, targets)
    Returns:
        targets: Ground truth targets
        scores: Prediction scores
    """
    pyg_batch, mask, targets = batch_data
    device = get_device()
    
    pyg_batch = pyg_batch.to(device)
    mask = mask.to(device)
    
    hidden = model(pyg_batch)
    scores = model.compute_scores(hidden, mask)
    
    return targets, scores


def train_test(model, train_data, test_data):
    """
    Train for one epoch and evaluate on test data.
    
    Args:
        model: SessionGraph model
        train_data: Training DataLoader
        test_data: Test DataLoader
    Returns:
        hit: Recall@20
        mrr: MRR@20
    """
    device = get_device()
    model.scheduler.step()
    
    print('Start training:', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    
    for i, batch_data in enumerate(train_data):
        model.optimizer.zero_grad()
        
        targets, scores = forward(model, batch_data)
        targets = torch.tensor(targets, dtype=torch.long, device=device) - 1
        
        loss = model.loss_function(scores, targets)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        model.optimizer.step()
        total_loss += loss.item()
        
        if i % max(1, len(train_data) // 5) == 0:
            print(f'[{i}/{len(train_data)}] Loss: {loss.item():.4f}')
    
    print(f'\tTotal Loss: {total_loss:.3f}')
    
    print('Start predicting:', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    
    with torch.no_grad():
        for batch_data in test_data:
            targets, scores = forward(model, batch_data)
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
