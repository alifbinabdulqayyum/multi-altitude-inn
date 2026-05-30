#!/bin/sh
#SBATCH --job-name=Test
#SBATCH -o log-test-10-160.out          
#SBATCH -e log-test-10-160-error.out          
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=alifbinabdulqayyum@tamu.edu  
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:tesla:1
#SBATCH --partition=gpu-research
#SBATCH --qos=olympus-research-gpu
#SBATCH --time=96:00:00

# enter your commands below
# set woring directory if different than current directory
# cd /working/directory
echo "Beginning"
cd /mnt/shared-scratch/Yoon_B/alifbinabdulqayyum/multi-altitude-inn

# Start anaconda shell (if needed)
source /mnt/shared-scratch/Yoon_B/alifbinabdulqayyum/anaconda3/bin/activate

# Activate conda environment
conda activate mmsr

result_save_dir="saved-results"

# Testing GEI-LIIF Models

for file_prefix in "ua";
do
    for height_0 in 10;
    do
        for height_1 in 160;
        do
            for sr_scale in 6.0 5.75 5.5 5.25 5.0 4.75 4.5 4.25 4.0 3.75 3.5 3.25 3.0 2.75 2.5 2.25 2.0 1.75 1.5 1.25;
            do
                for region in 'A' 'B' 'C' 'D';
                do
                    python test.py \
                        --data-dir "/mnt/shared-scratch/Yoon_B/alifbinabdulqayyum/MMSR-NEW/data" \
                        --use-gpu \
                        --use-global-encoder \
                        --saved-model-epoch 500 \
                        --file-prefix $file_prefix \
                        --height-0 $height_0 \
                        --height-1 $height_1 \
                        --h-LR 120 \
                        --w-LR 160 \
                        --sr-scale $sr_scale\
                        --sigma 30.0 \
                        --m 50 \
                        --model-save-dir "./saved-models-${file_prefix}-${height_0}-${height_1}-GEI-LIIF" \
                        --result-save-dir "${result_save_dir}/${file_prefix}" \
                        --test-region $region
                done
            done            
        done
    done
done