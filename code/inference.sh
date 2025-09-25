#!/bin/bash

# This script is used to do inference and generate the Chinese texts for the input images.
base_dir=/Users/vaibhavpatidar/dimtda_project_m2

trans_model_dir=$base_dir/pretrain_output_small/checkpoint-50
dit_model_dir=$base_dir/pretrained_models/dit-base
nougat_model_dir=$base_dir/pretrained_models/nougat-small
dimtda_model_dir=$base_dir/output_small/checkpoint-1000

image_processor_dir=$base_dir/code/utils/image_processor
zh_tokenizer_dir=$base_dir/code/utils/zh_tokenizer
qformer_config_dir=$base_dir/utils/blip2-opt-2.7b

image_dir=$base_dir/data/DoTA_dataset/imgs
split_json_file_path=$base_dir/data/DoTA_dataset/generated_split_200_50_50.json

result_dir=$base_dir/results

# macOS MPS settings
export PYTORCH_ENABLE_MPS_FALLBACK=1

python code/codes/inference.py \
    --trans_model_dir $trans_model_dir \
    --dit_model_dir $dit_model_dir \
    --nougat_model_dir $nougat_model_dir \
    --dimtda_model_dir $dimtda_model_dir \
    --image_processor_dir $image_processor_dir \
    --zh_tokenizer_dir $zh_tokenizer_dir \
    --image_dir $image_dir \
    --split_json_file_path $split_json_file_path \
    --result_dir $result_dir \
    --qformer_config_dir $qformer_config_dir
