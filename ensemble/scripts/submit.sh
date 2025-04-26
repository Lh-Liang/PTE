#!/bin/bash
currentPath="$( cd "$( dirname "$0"  )" && pwd  )"
cd ..
pwdPath="$(pwd)"

MODEL_PATH=../bert-base-uncased
TRAIN_DATA_PATH=../data/agnews/333/train.csv
EVAL_DATA_PATH=../data/agnews/333/val.csv

python -m src.train \
  --seed 666 \
  --epochs 2000 \
  --batch_size 8 \
  --max_seq_length 512 \
  --learning_rate 5e-5 \
  --log_freq 100 \
  --eval_freq 500 \
  --weight_lr 9e-5 \
  --classes_num 4 \
  --task_name 'AgNews' \
  --model_path $MODEL_PATH \
  --train_data_path $TRAIN_DATA_PATH \
  --eval_data_path $EVAL_DATA_PATH
