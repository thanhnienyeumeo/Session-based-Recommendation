#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for SR-GNN
Modernized implementation using PyTorch Geometric data structures

Key improvements over original:
- Uses PyG Data objects with sparse edge_index format
- Efficient graph construction with vectorized operations
- Native PyG DataLoader with proper batching
- Memory-efficient sparse representations
"""

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional


def build_graph(train_data: List[List[int]]) -> dict:
    """
    Build a global graph from training sequences.
    
    Args:
        train_data: List of item sequences
    Returns:
        Dictionary with edge information
    """
    edge_dict = {}
    
    for seq in train_data:
        for i in range(len(seq) - 1):
            edge = (seq[i], seq[i + 1])
            edge_dict[edge] = edge_dict.get(edge, 0) + 1
    
    # Normalize weights
    in_degree = {}
    for (src, dst), weight in edge_dict.items():
        in_degree[dst] = in_degree.get(dst, 0) + weight
    
    normalized_edges = {}
    for (src, dst), weight in edge_dict.items():
        normalized_edges[(src, dst)] = weight / in_degree[dst] if in_degree[dst] > 0 else 0
    
    return normalized_edges


def split_validation(train_set: Tuple, valid_portion: float = 0.1) -> Tuple:
    """
    Split training set into train and validation sets.
    
    Args:
        train_set: Tuple of (sequences, targets)
        valid_portion: Fraction of data to use for validation
    Returns:
        Tuple of ((train_x, train_y), (valid_x, valid_y))
    """
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    
    # Shuffle indices
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def sequence_to_graph(sequence: List[int]) -> Data:
    """
    Convert a session sequence to a PyG graph.
    
    Each unique item becomes a node, and consecutive items form directed edges.
    Uses sparse edge_index format which is much more memory efficient than
    dense adjacency matrices.
    
    Args:
        sequence: List of item IDs in the session
    Returns:
        PyG Data object with:
            - x: Node features (item IDs)
            - edge_index: COO format edge indices
            - edge_weight: Normalized edge weights
    """
    # Get unique items while preserving order
    items = []
    seen = set()
    for item in sequence:
        if item not in seen and item != 0:
            items.append(item)
            seen.add(item)
    
    if len(items) == 0:
        # Handle empty sequence
        return Data(
            x=torch.tensor([0], dtype=torch.long),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_weight=torch.zeros(0, dtype=torch.float)
        )
    
    # Create item to index mapping
    item_to_idx = {item: idx for idx, item in enumerate(items)}
    
    # Build edges from consecutive items
    edge_dict = {}
    for i in range(len(sequence) - 1):
        if sequence[i] == 0 or sequence[i + 1] == 0:
            continue
        src_idx = item_to_idx[sequence[i]]
        dst_idx = item_to_idx[sequence[i + 1]]
        edge_dict[(src_idx, dst_idx)] = edge_dict.get((src_idx, dst_idx), 0) + 1
    
    # Compute edge weights (normalized by in-degree)
    if len(edge_dict) > 0:
        # Calculate in-degrees
        in_degree = {}
        for (src, dst), weight in edge_dict.items():
            in_degree[dst] = in_degree.get(dst, 0) + weight
        
        # Build edge tensors with normalized weights
        edges_src = []
        edges_dst = []
        edge_weights = []
        
        for (src, dst), weight in edge_dict.items():
            # Add forward edge
            edges_src.append(src)
            edges_dst.append(dst)
            edge_weights.append(weight / in_degree[dst] if in_degree[dst] > 0 else 0)
        
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)
    
    # Node features are item IDs
    x = torch.tensor(items, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)


class SessionDataset(Dataset):
    """
    PyTorch Dataset for session-based recommendation.
    
    Each sample contains:
    - A session graph (PyG Data object)
    - The original sequence (for attention ordering)
    - A mask indicating valid positions
    - The target item
    """
    def __init__(self, data: Tuple, shuffle: bool = False):
        """
        Args:
            data: Tuple of (sequences, targets)
            shuffle: Whether to shuffle data on each epoch
        """
        self.sequences = data[0]
        self.targets = np.asarray(data[1])
        self.shuffle = shuffle
        
        # Compute max sequence length for padding
        self.max_len = max(len(seq) for seq in self.sequences)
        
        # Pre-compute graphs for efficiency
        self.graphs = [sequence_to_graph(seq) for seq in self.sequences]
        
        # Pre-compute padded sequences and masks
        self.padded_sequences = []
        self.masks = []
        for seq in self.sequences:
            padded = seq + [0] * (self.max_len - len(seq))
            mask = [1] * len(seq) + [0] * (self.max_len - len(seq))
            self.padded_sequences.append(padded)
            self.masks.append(mask)
        
        self.padded_sequences = np.array(self.padded_sequences)
        self.masks = np.array(self.masks)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            self.graphs[idx],
            self.padded_sequences[idx],
            self.masks[idx],
            self.targets[idx]
        )


def collate_fn(batch):
    """
    Custom collate function for batching session graphs.
    
    Args:
        batch: List of (graph, sequence, mask, target) tuples
    Returns:
        Tuple of (batched_graph, mask_tensor, targets)
    """
    graphs, sequences, masks, targets = zip(*batch)
    
    # Create PyG batch
    batched_graph = Batch.from_data_list(list(graphs))
    
    # Store sequences in the batch for proper ordering during forward pass
    batched_graph.sequence = list(sequences)
    
    # Convert masks and targets to tensors
    mask_tensor = torch.tensor(np.array(masks), dtype=torch.long)
    
    return batched_graph, mask_tensor, np.array(targets)


def create_dataloader(data: Tuple, batch_size: int, shuffle: bool = False, 
                      num_workers: int = 0) -> DataLoader:
    """
    Create a DataLoader for session data.
    
    Args:
        data: Tuple of (sequences, targets)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
    Returns:
        PyTorch DataLoader
    """
    dataset = SessionDataset(data, shuffle=shuffle)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


# Legacy compatibility class
class Data_Legacy:
    """
    Legacy Data class for backwards compatibility.
    Converts to modern format when iterated.
    """
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        self.raw_inputs = inputs
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph
        
        # Compute masks and padding
        us_lens = [len(upois) for upois in inputs]
        len_max = max(us_lens)
        self.inputs = np.array([upois + [0] * (len_max - len(upois)) for upois in inputs])
        self.mask = np.array([[1] * le + [0] * (len_max - le) for le in us_lens])
        self.len_max = len_max
    
    def generate_batch(self, batch_size):
        """Generate batch indices (legacy interface)."""
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
            self.raw_inputs = [self.raw_inputs[i] for i in shuffled_arg]
        
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices
    
    def get_slice(self, indices):
        """Get a batch slice (legacy interface with modern graph format)."""
        inputs = self.inputs[indices]
        mask = self.mask[indices]
        targets = self.targets[indices]
        raw_inputs = [self.raw_inputs[i] for i in indices]
        
        # Convert to PyG format
        graphs = [sequence_to_graph(seq) for seq in raw_inputs]
        batched_graph = Batch.from_data_list(graphs)
        batched_graph.sequence = [list(seq) for seq in inputs]
        
        mask_tensor = torch.tensor(mask, dtype=torch.long)
        
        return batched_graph, mask_tensor, targets
