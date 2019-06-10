python3 srez_main.py --dataset_input /home/enhaog/GANCS/srez/dataset_MRI/phantom --dataset_output  /home/enhaog/GANCS/srez/dataset_MRI/phantom  --batch_size 8 --run train --gene_mse_factor 0.1 --summary_period 125 --sample_size 256 --train_time 10                    

python3 srez_main.py --dataset_input /home/enhaog/GANCS/srez/dataset_MRI/phantom --dataset_output  /home/enhaog/GANCS/srez/dataset_MRI/phantom  --batch_size 8 --run train --gene_mse_factor 0.01 --gene_l2_factor 0.1 --summary_period 125 --sample_size 256 --train_time 100

# Added by BC (paths need to be replaced)

RED=4 # 4-fold reduction factor
NUM=0 # Trial or cross-validation fold number

python srez_main.py --run train --dataset <dataset-path> --sampling_pattern <sampling-pattern-path> --cv_groups 5 --cv_index 0 --sample_train 6400 --sample_test 1600 --number_of_copies 5 --batch_size 2 --num_epoch 20 --gpu_memory_fraction 0.9 --gene_mse_factor 0.95 --learning_beta1 0.9 --learning_rate_start 1e-5 --learning_rate_half_life 10000 --sample_size 256 --sample_size_y 256 --checkpoint_period -1 --train_dir train_${RED}_${NUM} --checkpoint_dir checkpoint_${RED}_${NUM}

python srez_main.py --run demo --dataset <dataset-path> --sampling_pattern <sampling-pattern-path> --sample_test 160 --sample_size 256 --number_of_copies 5 --batch_size 2 --gpu_memory_fraction 0.9 --train_dir test_${RED}_${NUM} --checkpoint_dir checkpoint_${RED}_${NUM} --summary_period 1

