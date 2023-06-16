export CUDA_VISIBLE_DEVICES=1,5
export MASTER_ADDR=localhost
export NUM_TRAINERS=2

torchrun --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:0 \
    --nnodes=1 --nproc_per_node=$NUM_TRAINERS \
    run.py --batch 150 --epochs 30 \
    --embed-size 512 --ffn-hid-size 2048 --nhead 8 --norm-first True \
    --encoder-layer 6 --decoder-layer 6 --dropout 0.1 \
    --max-len 128 --lr 2e-4 --warmup 0 \
    --num-gpus $NUM_TRAINERS --mode train

run.py --embed-size 512 --ffn-hid-size 2048 --nhead 8 --norm-first True \
    --encoder-layer 6 --decoder-layer 6 --dropout 0.1 \
    --max-len 128 --lr 2e-4 --warmup 0 \
    --num-gpus $NUM_TRAINERS --mode train