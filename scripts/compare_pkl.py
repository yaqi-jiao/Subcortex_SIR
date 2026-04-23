#!/usr/bin/env python3

import pickle
import numpy as np
import sys


def print_pkl_info(filepath):
    print(f"\n{'='*60}")
    print(f"File: {filepath}")
    print('='*60)
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return
    
    print(f"Keys: {list(data.keys())}\n")
    
    for key, value in data.items():
        print(f"Key: '{key}'")
        # Print type
        print(f"  Type: {type(value)}")
        # If numpy array, print shape, dtype, and a sample
        if isinstance(value, np.ndarray):
            print(f"  Shape: {value.shape}")
            print(f"  Dtype: {value.dtype}")
            # Print first few elements if size is manageable
            if value.size > 0:
                if value.ndim == 1:
                    sample = value[:min(5, len(value))]
                elif value.ndim == 2:
                    sample = value[:min(2, value.shape[0]), :min(5, value.shape[1])]
                else:
                    sample = value.flat[:min(5, value.size)]
                print(f"  Sample (first few): {sample}")
        elif isinstance(value, (list, tuple)):
            print(f"  Length: {len(value)}")
            if len(value) > 0:
                print(f"  First element type: {type(value[0])}")
                print(f"  Sample (first few): {value[:min(5, len(value))]}")
        elif isinstance(value, dict):
            print(f"  Number of keys: {len(value)}")
            print(f"  Keys: {list(value.keys())[:10]}")  # limit
        else:
            # For scalar or other types, print the value (truncated if long)
            str_val = str(value)
            if len(str_val) > 200:
                str_val = str_val[:200] + "..."
            print(f"  Value: {str_val}")
        print()  # blank line

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_first.pkl> <path_to_second.pkl>")
        sys.exit(1)
    
    path1, path2 = sys.argv[1], sys.argv[2]
    print_pkl_info(path1)
    print_pkl_info(path2)