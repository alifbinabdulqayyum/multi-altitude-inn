# Train GEI-LIIF Models

for file_prefix in "ua" "va";
do
    for height_0 in 10 60;
    do
        for height_1 in 160 200;
        do
            python train.py \
                --data-dir "/media/alif/Alif/Research(Yoon)/MMSR-VAE/data" \
                --use-gpu \
                --use-global-encoder \
                --batch-size 8 \
                --train-sr 5.0 \
                --file-prefix $file_prefix \
                --height-0 $height_0 \
                --height-1 $height_1 \
                --train-frac 0.8 \
                --h-LR 120 \
                --w-LR 160 \
                --latent-dim 1 \
                --sigma 30.0 \
                --m 50 \
                --n-feature-encoder-blocks 2 \
                --encoder-scale 2 \
                --residual-scale 1.0 \
                --input-dim 1 \
                --n-encoder-features 64 \
                --encoder-kernel-size 3 \
                --n-encoder-resblocks 16 \
                --encoder-non-lin 'relu' \
                --decoder-non-lin 'relu' \
                --num-epochs 500 \
                --encoder-model 'edsr' \
                --encoder-lr 1e-4 \
                --encoder-wd 1e-6 \
                --encoder-bias \
                --encoder-upsampling \
                --decoder-lr 1e-4 \
                --decoder-wd 1e-6 \
                --learning-rate-decay 0.999 \
                --start-lr-decay 0 \
                --lr-decay-interval 1 \
                --lambda-s 1.0 \
                --lambda-c 1.0 \
                --lambda-l 0.1 \
                --lambda-kl 1.0 \
                --model-save-dir "./saved-models-${file_prefix}-${height_0}-${height_1}-GEI-LIIF" \
                --save-interval 100 \
                --query-points 1024
        done
    done
done

# Train GPEI-LIIF Models

for file_prefix in "ua" "va";
do
    for height_0 in 10 60;
    do
        for height_1 in 160 200;
        do
            python train.py \
                --data-dir "/media/alif/Alif/Research(Yoon)/MMSR-VAE/data" \
                --use-gpu \
                --use-global-encoder \
                --use-pos-encoder \
                --batch-size 8 \
                --train-sr 5.0 \
                --file-prefix $file_prefix \
                --height-0 $height_0 \
                --height-1 $height_1 \
                --train-frac 0.8 \
                --h-LR 120 \
                --w-LR 160 \
                --latent-dim 1 \
                --sigma 30.0 \
                --m 50 \
                --n-feature-encoder-blocks 2 \
                --encoder-scale 2 \
                --residual-scale 1.0 \
                --input-dim 1 \
                --n-encoder-features 64 \
                --encoder-kernel-size 3 \
                --n-encoder-resblocks 16 \
                --encoder-non-lin 'relu' \
                --decoder-non-lin 'relu' \
                --num-epochs 500 \
                --encoder-model 'edsr' \
                --encoder-lr 1e-4 \
                --encoder-wd 1e-6 \
                --encoder-bias \
                --encoder-upsampling \
                --decoder-lr 1e-4 \
                --decoder-wd 1e-6 \
                --learning-rate-decay 0.999 \
                --start-lr-decay 0 \
                --lr-decay-interval 1 \
                --lambda-s 1.0 \
                --lambda-c 1.0 \
                --lambda-l 0.1 \
                --lambda-kl 1.0 \
                --model-save-dir "./saved-models-${file_prefix}-${height_0}-${height_1}-GPEI-LIIF" \
                --save-interval 100 \
                --query-points 1024
        done
    done
done