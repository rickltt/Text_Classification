#!/bin/bash
MODEL='rnn'
PLMS='../../bert/bert-base-chinese'
DATA_DIR='../data/THUCNews'
python3 run.py \
  --model_name $MODEL \
  --dataset_dir $DATA_DIR \
  --model_name_or_path $PLMS \
  --cache_dir cache_dir \
  --overwrite_cache \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy steps \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --weight_decay 0.01 \
  --output_dir output \
  --overwrite_output_dir \
  --pad_to_max_length \
  --logging_steps 1000 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --seed 42 
