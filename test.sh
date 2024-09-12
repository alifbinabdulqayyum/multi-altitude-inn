result_save_dir="saved-results"

# Testing GEI-LIIF Models

for file_prefix in "ua" "va";
do
    for height_0 in 10 60;
    do
        for height_1 in 160 200;
        do
            for sr_scale in 1.0 2.0 3.0 4.0 5.0 6.0 7.0;
            do
                python test.py \
                    --data-dir "./data" \
                    --use-gpu \
                    --use-global-encoder \
                    --saved-model-epoch 500 \
                    --file-prefix $file_prefix \
                    --height-0 $height_0 \
                    --height-1 $height_1 \
                    --train-frac 0.8 \
                    --h-LR 120 \
                    --w-LR 160 \
                    --sr-scale $sr_scale\
                    --sigma 30.0 \
                    --m 50 \
                    --model-save-dir "./saved-models-${file_prefix}-${height_0}-${height_1}-GEI-LIIF" \ \
                    --result-save-dir "${result_save_dir}/${file_prefix}"
            done            
        done
    done
done

# Testing GPEI-LIIF Models

for file_prefix in "ua" "va";
do
    for height_0 in 10 60;
    do
        for height_1 in 160 200;
        do
            for sr_scale in 1.0 2.0 3.0 4.0 5.0 6.0 7.0;
            do
                python test.py \
                    --data-dir "./data" \
                    --use-gpu \
                    --use-global-encoder \
                    --use-pos-encoder \
                    --saved-model-epoch 500 \
                    --file-prefix $file_prefix \
                    --height-0 $height_0 \
                    --height-1 $height_1 \
                    --train-frac 0.8 \
                    --h-LR 120 \
                    --w-LR 160 \
                    --sr-scale $sr_scale\
                    --sigma 30.0 \
                    --m 50 \
                    --model-save-dir "./saved-models-${file_prefix}-${height_0}-${height_1}-GPEI-LIIF" \ \
                    --result-save-dir "${result_save_dir}/${file_prefix}"
            done            
        done
    done
done

# Testing PEI-LIIF Models

for file_prefix in "ua" "va";
do
    for height_0 in 10 60;
    do
        for height_1 in 160 200;
        do
            for sr_scale in 1.0 2.0 3.0 4.0 5.0 6.0 7.0;
            do
                python test.py \
                    --data-dir "./data" \
                    --use-gpu \
                    --use-pos-encoder \
                    --saved-model-epoch 500 \
                    --file-prefix $file_prefix \
                    --height-0 $height_0 \
                    --height-1 $height_1 \
                    --train-frac 0.8 \
                    --h-LR 120 \
                    --w-LR 160 \
                    --sr-scale $sr_scale\
                    --sigma 30.0 \
                    --m 50 \
                    --model-save-dir "./saved-models-${file_prefix}-${height_0}-${height_1}-PEI-LIIF" \ \
                    --result-save-dir "${result_save_dir}/${file_prefix}"
            done            
        done
    done
done

# Testing LIIF Models

for file_prefix in "ua" "va";
do
    for height_0 in 10 60;
    do
        for height_1 in 160 200;
        do
            for sr_scale in 1.0 2.0 3.0 4.0 5.0 6.0 7.0;
            do
                python test.py \
                    --data-dir "./data" \
                    --use-gpu \
                    --saved-model-epoch 500 \
                    --file-prefix $file_prefix \
                    --height-0 $height_0 \
                    --height-1 $height_1 \
                    --train-frac 0.8 \
                    --h-LR 120 \
                    --w-LR 160 \
                    --sr-scale $sr_scale\
                    --sigma 30.0 \
                    --m 50 \
                    --model-save-dir "./saved-models-${file_prefix}-${height_0}-${height_1}-LIIF" \ \
                    --result-save-dir "${result_save_dir}/${file_prefix}"
            done            
        done
    done
done
