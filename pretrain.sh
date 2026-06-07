#!/bin/bash

# 预定义的增强策略池
AUG_METHODS=("dropN" "permE" "maskN")
LEARNING_RATES=(0.01 0.005)

# 遍历所有增强组合和学习率
for lr in "${LEARNING_RATES[@]}"; do
    for aug1 in "${AUG_METHODS[@]}"; do
        for aug2 in "${AUG_METHODS[@]}"; do
            
            echo "=========================================================="
            echo "Running GraphCL Pretrain: aug1=$aug1, aug2=$aug2, lr=$lr"
            echo "=========================================================="
            
            python MyPretrain.py \
                --task 'GraphCL' \
                --dataset_name 'Cora_ml' \
                --preprocess_method 'none' \
                --gnn_type 'GCN' \
                --hid_dim 64 \
                --num_layer 2 \
                --epochs 200 \
                --seed 56 \
                --device 1 \
                --aug1 "$aug1" \
                --aug2 "$aug2" \
                --lr "$lr"
        done
    done
done

echo "All grid search pretraining tasks finished!"