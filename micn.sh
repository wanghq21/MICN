
# --features M  : multivariate
# --features S  : univariate

# --mode regre  : MICN-regre
# --mode mean  : MICN-mean


pred_l=(96 192 336 720)

for i in ${pred_l[@]}
do
    python -u run.py \
        --model micn \
        --mode regre \
        --data ETTm2 \
        --features M \
        --freq t \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --d_model 512 \
        --seq_len 96 \
        --label_len 96 \
        --pred_len $i
done

for i in ${pred_l[@]}
do
    python -u run.py \
        --model micn \
        --mode regre \
        --data ECL \
        --features M \
        --freq h \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --d_model 512 \
        --seq_len 96 \
        --label_len 96 \
        --pred_len $i
done

for i in ${pred_l[@]}
do
    python -u run.py \
        --model micn \
        --mode regre \
        --data Exchange \
        --features M \
        --freq d \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --d_model 512 \
        --seq_len 96 \
        --label_len 96 \
        --pred_len $i
done

for i in ${pred_l[@]}
do
    python -u run.py \
        --model micn \
        --mode regre \
        --data Traffic \
        --features M \
        --freq h \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --d_model 512 \
        --seq_len 96 \
        --label_len 96 \
        --pred_len $i
done

for i in ${pred_l[@]}
do
    python -u run.py \
        --model micn \
        --mode regre \
        --data WTH \
        --features M \
        --freq t \
        --conv_kernel 12 16 \
        --d_layers 1 \
        --d_model 512 \
        --seq_len 96 \
        --label_len 96 \
        --pred_len $i
done


pred_l=(24 36 48 60)

for i in ${pred_l[@]}
do
    python -u run.py \
        --model micn \
        --mode mean \
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

