export CUDA_VISIBLE_DEVICES=1,5
export MASTER_ADDR=localhost
export NUM_TRAINERS=2

#torchrun --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:0 \
#    --nnodes=1 --nproc_per_node=$NUM_TRAINERS \
#    run.py --batch 32 --num-workers 8 --epochs 30 \
#    --embed-size 768 --ffn-hid-size 3072 --nhead 12 --norm-first True \
#    --num-layer 12 --dropout 0.1 --activation gelu \
#    --max-len 512 --lr 2e-4 --warmup 0 \
#    --num-gpus $NUM_TRAINERS --mode train

torchrun --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:0 \
    --nnodes=1 --nproc_per_node=$NUM_TRAINERS \
    run.py --batch 64 --num-workers 8 --epochs 30 \
    --embed-size 512 --ffn-hid-size 2048 --nhead 8 --norm-first True \
    --num-layer 8 --dropout 0.1 --activation gelu \
    --max-len 512 --lr 1e-4 --warmup 0 \
    --num-gpus $NUM_TRAINERS --mode train