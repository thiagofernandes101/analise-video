#!/usr/bin/env python3
"""Inspect ST-GCN checkpoint architecture"""
import torch
import sys

try:
    checkpoint = torch.load('st_gcn_weights.pt', map_location='cpu')
    
    print("=== Checkpoint Keys ===")
    if isinstance(checkpoint, dict):
        # Check if it's a full checkpoint or just state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Group by layer
    layers = {}
    for key in state_dict.keys():
        if 'st_gcn_networks' in key:
            layer_num = key.split('.')[1]
            if layer_num not in layers:
                layers[layer_num] = []
            layers[layer_num].append(key)
    
    print(f"\nFound {len(layers)} ST-GCN layers")
    for layer_num in sorted(layers.keys(), key=int):
        print(f"\nLayer {layer_num}:")
        for key in sorted(layers[layer_num])[:5]:  # Show first 5 keys
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else '?'
            print(f"  {key}: {shape}")
    
    # Check key shapes
    print("\n=== Key Architecture Info ===")
    if 'A' in state_dict:
        print(f"A (adjacency): {state_dict['A'].shape}")
    if 'data_bn.weight' in state_dict:
        print(f"data_bn.weight: {state_dict['data_bn.weight'].shape}")
    if 'fcn.weight' in state_dict:
        print(f"fcn.weight: {state_dict['fcn.weight'].shape}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
