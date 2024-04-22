#! /bin/bash

# Change for multinode config
MP_SIZE=1

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_zero-offload_10B_config.json"
gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --batch-size 20 \
       --num-layers 1 \
       --train-iters 100 \
       --lazy-loader \
       --distributed-backend nccl \
       --lr 0.00015 \
       --no-load-optim \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --fp16 \
       --finetune \
       --log-interval 1 \
       --max_len 512 \
       --summary_len 150 \
       --train-data-path albert_news_summary/albert_train.csv \
       --val-data-path albert_news_summary/albert_val.csv \
       --test-data-path albert_news_summary/albert_test.csv \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} pretrain_albert.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
