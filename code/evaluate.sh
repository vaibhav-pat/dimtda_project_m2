#!/bin/bash

# This script is used to calcaute BLEU, BLEU-PT and STEDS.
base_dir=/Users/vaibhavpatidar/dimtda_project_m2

split_json_file_path=$base_dir/data/DoTA_dataset/generated_split_200_50_50.json
zh_mmd_dir=$base_dir/data/DoTA_dataset/zh_mmd
result_dir=$base_dir/results

python code/codes/run_evaluation.pyevaluate.shbash \
    --split_json_file_path $split_json_file_path \
    --result_dir $result_dir \
    --zh_mmd_dir $zh_mmd_dir
