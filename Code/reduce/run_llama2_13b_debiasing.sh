#!/bin/bash

OUTPUT=$1
ZERO_STAGE=$2
MODEL_PATH=$3

if [ "$OUTPUT" == "" ]; then
    OUTPUT=outputs/ft_models/llama2_13b_debiasing
fi

if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi

if [ "$MODEL_PATH" == "" ]; then
    MODEL_PATH=models/Llama-2-13b-chat-hf
fi

mkdir -p "$OUTPUT"

CUDA_VISIBLE_DEVICES=1 deepspeed main_lora.py \
   --data_path llama2/debiasing \
   --data_split 2,4,4 \
   --model_name_or_path "$MODEL_PATH" \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0.1 \
   --num_train_epochs 2 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage "$ZERO_STAGE" \
   --lora_dim 128 \
   --lora_module_name layers. \
   --deepspeed \
   --model_type llama2 \
   --output_dir "$OUTPUT" \
   &> "$OUTPUT/training.log"