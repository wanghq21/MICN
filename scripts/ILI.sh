pred_l=(24 36 48 60)

for i in ${pred_l[@]}
do
    python -u run.py \
        --model micn \
        --mode regre \
        --data ILI \
        --features M \
        --freq d \
        --conv_kernel 18 12 \
        --d_layers 1 \
        --d_model 64 \
        --seq_len 36 \
        --label_len 36 \
        --pred_len $i
done