#!/bin/bash

# This script is used to construct DIMTDA model and finetune it on the DoTA dataset.
base_dir=/Users/vaibhavpatidar/dimtda_project_m2

trans_model_dir=$base_dir/pretrain_output_small/checkpoint-50
dit_model_dir=$base_dir/pretrained_models/dit-base
nougat_model_dir=$base_dir/pretrained_models/nougat-small
dimtda_model_dir=$base_dir/output_small

image_processor_dir=$base_dir/code/utils/image_processor
zh_tokenizer_dir=$base_dir/code/utils/zh_tokenizer
qformer_config_dir=$base_dir/code/utils/blip2-opt-2.7b

image_dir=$base_dir/data/DoTA_dataset/imgs
zh_mmd_dir=$base_dir/data/DoTA_dataset/zh_mmd
split_json_file_path=$base_dir/data/DoTA_dataset/generated_split_200_50_50.json

# macOS MPS settings
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

python code/codes/finetune_dimtda.py \
    --trans_model_dir $trans_model_dir \
    --dit_model_dir $dit_model_dir \
    --nougat_model_dir $nougat_model_dir \
    --image_processor_dir $image_processor_dir \
    --zh_tokenizer_dir $zh_tokenizer_dir \
    --image_dir $image_dir \
    --zh_mmd_dir $zh_mmd_dir \
    --split_json_file_path $split_json_file_path \
    --output_dir $dimtda_model_dir \
    --qformer_config_dir $qformer_config_dir \
    --batch_size_per_gpu 1
