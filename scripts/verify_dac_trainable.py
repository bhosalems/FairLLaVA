#!/usr/bin/env python3
"""
Quick script to verify that DAC parameters are properly trainable during debiasing.
Usage:
    python scripts/verify_dac_trainable.py --config configs/fairness_finetune_mimic_cxr.json
"""

import torch
import json
import argparse
from pathlib import Path

def check_trainable_params(model):
    """Check which parameters are trainable"""
    trainable = []
    frozen = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append((name, param.numel()))
        else:
            frozen.append((name, param.numel()))
    
    return trainable, frozen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to fairness config")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)
    
    print("=" * 80)
    print(f"Config: {args.config}")
    print("=" * 80)
    print(f"fairness_stage: {config.get('fairness_stage')}")
    print(f"dac_frozen: {config.get('dac_frozen')}")
    print(f"dac_query_mode: {config.get('dac_query_mode')}")
    print(f"dac_loss_mode: {config.get('dac_loss_mode')}")
    print()
    
    # Determine expected behavior
    stage = config.get('fairness_stage')
    dac_frozen = config.get('dac_frozen', True)
    query_mode = config.get('dac_query_mode', 'learned')
    
    print("=" * 80)
    print("EXPECTED BEHAVIOR:")
    print("=" * 80)
    
    if stage == "pretrain_dac":
        print("✅ Stage: pretrain_dac")
        print("✅ DAC parameters should be trainable")
        print("✅ Backbone/LoRA should be frozen")
        print()
        
    elif stage in ["debias", "debias_joint"]:
        print(f"✅ Stage: {stage}")
        print(f"✅ LoRA adapters should be trainable")
        
        if dac_frozen:
            print("❌ DAC parameters should be FROZEN")
            if query_mode == "learned":
                print("⚠️  WARNING: Using 'learned' queries with frozen DAC!")
                print("⚠️  Queries won't adapt to changing representations.")
                print("⚠️  Recommendation: Set dac_frozen=false OR use dac_query_mode='conditioned'")
        else:
            print("✅ DAC parameters should be TRAINABLE (added via modules_to_save)")
            if query_mode == "learned":
                print("✅ Using 'learned' queries with trainable DAC - queries will adapt")
            else:
                print("ℹ️  Using 'conditioned' queries - queries adapt even if DAC frozen")
        print()
    
    print("=" * 80)
    print("TO VERIFY DURING TRAINING:")
    print("=" * 80)
    print("1. Check training logs for:")
    print("   - 'modules_to_save (DAC): ...' <- Should appear if dac_frozen=False")
    print("   - 'DAC modules are trainable...' <- Should appear if dac_frozen=False")
    print()
    print("2. After 1st training step, check WandB/logs for:")
    print("   - Number of trainable parameters")
    print("   - Should include DAC params if dac_frozen=False")
    print()
    print("3. Verify gradient flow:")
    print("   - Run with --debug flag and check for gradient norms")
    print("   - DAC parameters should have non-zero gradients if trainable")
    print()
    
    # Additional recommendations
    print("=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    if stage in ["debias", "debias_joint"]:
        if query_mode == "learned" and dac_frozen:
            print("⚠️  PROBLEM DETECTED:")
            print("   You're using learned (static) queries but DAC is frozen.")
            print("   The queries won't adapt to changing representations from LoRA.")
            print()
            print("   FIX: Choose ONE of these options:")
            print("   Option 1: Set 'dac_frozen': false  (finetune DAC + queries)")
            print("   Option 2: Set 'dac_query_mode': 'conditioned'  (adaptive queries)")
            print()
        elif query_mode == "learned" and not dac_frozen:
            print("✅ GOOD CONFIGURATION:")
            print("   Learned queries + trainable DAC = queries will adapt via DAC finetuning")
            print()
        elif query_mode == "conditioned":
            print("✅ GOOD CONFIGURATION:")
            print("   Conditioned queries adapt to representations regardless of DAC frozen state")
            print()

if __name__ == "__main__":
    main()
