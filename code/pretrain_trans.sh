#!/bin/bash

# This script is used to pretrain a En-Zh text translation model. We use the DoTA dataset for convinience.
base_dir=/Users/vaibhavpatidar/dimtda_project_m2

trans_model_dir=$base_dir/pretrain_output

en_tokenizer_dir=$base_dir/code/utils/en_tokenizer
zh_tokenizer_dir=$base_dir/code/utils/zh_tokenizer

en_mmd_dir=$base_dir/data/DoTA_dataset/en_mmd
zh_mmd_dir=$base_dir/data/DoTA_dataset/zh_mmd

split_json_file_path=$base_dir/data/DoTA_dataset/generated_split_200_50_50.json

# macOS MPS settings
export PYTORCH_ENABLE_MPS_FALLBACK=1

python code/codes/pretrain_trans.py \
    --en_tokenizer_dir $en_tokenizer_dir \
    --zh_tokenizer_dir $zh_tokenizer_dir \
    --en_mmd_dir $en_mmd_dir \
    --zh_mmd_dir $zh_mmd_dir \
    --split_json_file_path $split_json_file_path \
    --output_dir $trans_model_dir \
    --batch_size_per_gpu 1