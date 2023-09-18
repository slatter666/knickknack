CUDA_VISIBLE_DEVICES=0,5
NUM_TRAINERS=1

torchrun --nnodes=1 --nproc_per_node=$NUM_TRAINERS \
    train.py --output_dir caption_ckpt --bf16 True \
    --num_train_epochs 5 --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 --evaluation_strategy "no" \
    --remove_unused_columns False --save_strategy "epoch" \
    --learning_rate 2e-5 --weight_decay 0.0 \
    --logging_steps 100 --tf32 True --seed 999 --data_seed 999