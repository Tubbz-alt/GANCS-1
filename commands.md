# Sample commands by Benjamin Cottier

## Initial

### Train
```
python srez_main.py --dataset /home/Student/s4360417/honours/datasets/oasis3/exp2_jpg/train --dataset_train /home/Student/s4360417/honours/datasets/oasis3/exp2_jpg/train --dataset_test /home/Student/s4360417/honours/datasets/oasis3/exp2_jpg/valid --batch_size 8 --run train --gene_mse_factor 0.1 --summary_period 1000000 --sample_size 256 --train_time 60 --subsample_train -1 --subsample_test -1
```

## Replication

### Train
```
python srez_main.py --run train --dataset /home/Student/s4360417/honours/datasets/oasis3/exp2_jpg/train --dataset_train /home/Student/s4360417/honours/datasets/oasis3/exp2_jpg/train --dataset_test /home/Student/s4360417/honours/datasets/oasis3/exp2_jpg/valid --sample_train 960 --sample_test 240  --subsample_train -1 --subsample_test -1 --batch_size 4 --gene_mse_factor 0.975 --learning_beta 0.9 --learning_rate_start 0.000001 --learning_rate_half_life 10000 --sample_size 256 --sample_size_y 256 --summary_period 1000000 --train_time 60
```

exp3
```
python srez_main.py --run train --dataset /home/Student/s4360417/honours/datasets/oasis3/exp3_png/train --dataset_train /home/Student/s4360417/honours/datasets/oasis3/exp3_png/train --dataset_test /home/Student/s4360417/honours/datasets/oasis3/exp3_png/valid --batch_size 4 --num_epoch 20 --gene_mse_factor 0.975 --learning_beta 0.9 --learning_rate_start 0.000001 --learning_rate_half_life 10000 --sample_size 256 --sample_size_y 256 --summary_period 1600 --checkpoint_period -1
```

```
python srez_main.py --run train --dataset /home/Student/s4360417/honours/datasets/oasis3/exp3_png/train --dataset_train /home/Student/s4360417/honours/datasets/oasis3/exp3_png/train --dataset_test /home/Student/s4360417/honours/datasets/oasis3/exp3_png/valid --batch_size 4 --num_epoch 1 --gene_mse_factor 0.975 --learning_beta1 0.9 --learning_rate_start 0.000001 --learning_rate_half_life 10000 --sample_size 256 --sample_size_y 256 --summary_period 1600 --checkpoint_period -1
```

```
python srez_main.py --run train --dataset /home/Student/s4360417/honours/datasets/oasis3/exp3_png_2/ground_truth/train --dataset_train /home/Student/s4360417/honours/datasets/oasis3/exp3_png_2/ground_truth/train --dataset_test /home/Student/s4360417/honours/datasets/oasis3/exp3_png_2/ground_truth/train --sampling_pattern  /home/Student/s4360417/honours/datasets/oasis3/exp3_png_2/artefact_fcs_2/mask.mat --sample_train 6400 --sample_test 1600 --batch_size 4 --num_epoch 20 --gene_mse_factor 0.975 --learning_beta1 0.9 --learning_rate_start 0.000001 --learning_rate_half_life 10000 --sample_size 256 --sample_size_y 256 --summary_period 1600 --checkpoint_period -1
```

```
python srez_main.py --run train --dataset /home/Student/s4360417/honours/datasets/oasis3/exp3_png_2/ground_truth/train --dataset_train /home/Student/s4360417/honours/datasets/oasis3/exp3_png_2/ground_truth/train --dataset_test /home/Student/s4360417/honours/datasets/oasis3/exp3_png_2/ground_truth/train --sampling_pattern  /home/Student/s4360417/honours/datasets/oasis3/exp3_png_2/artefact_fcs_1/mask.mat --sample_train 6400 --sample_test 1600 --batch_size 4 --num_epoch 20 --gene_mse_factor 0.975 --learning_beta1 0.9 --learning_rate_start 1e-6 --learning_rate_half_life 10000 --sample_size 256 --sample_size_y 256 --summary_period 1600 --checkpoint_period -1 --train_dir train_r4 --checkpoint_dir checkpoint_r4
```

exp4: recurrent 10 copies, 1 residual block*, cross-validation

```
#!/bin/bash
#SBATCH --job-name=rgancs_train
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bcottier2@gmail.com
#SBATCH --time=2-0
#SBATCH --output=slurm-%A-%a.out
#SBATCH --array=0-9
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

conda activate tf-biomed
cd ~/honours/GANCS
python srez_main.py --run train --dataset /home/Student/s4360417/honours/datasets/oasis3/exp3_png_3/cv --sampling_pattern /home/Student/s4360417/honours/datasets/oasis3/exp3_png_3/artefact_fcs_0/mask.mat --cv_groups 10 --cv_index $SLURM_ARRAY_TASK_ID --sample_train 11520 --sample_test 1280 --batch_size 2 --num_epoch 20 --gene_mse_factor 0.9 --learning_beta1 0.9 --learning_rate_start 1e-5 --learning_rate_half_life 10000 --sample_size 256 --sample_size_y 256 --summary_period 5760 --checkpoint_period -1 --train_dir train$SLURM_ARRAY_TASK_ID --checkpoint_dir checkpoint$SLURM_ARRAY_TASK_ID
```

Initial RGANCS with L1 only
```
python srez_main.py --run train --dataset /home/Student/s4360417/honours/datasets/oasis3/fractal_cs/cv --sampling_pattern /home/Student/s4360417/honours/datasets/oasis3/fractal_cs/artefact_fcs_4/mask.mat --cv_groups 5 --cv_index 0 --sample_train 6400 --sample_test 1600 --batch_size 2 --num_epoch 20 --gene_mse_factor 1.0 --learning_beta1 0.9 --learning_rate_start 1e-5 --learning_rate_half_life 10000 --sample_size 256 --sample_size_y 256 --checkpoint_period -1 --train_dir train0 --checkpoint_dir checkpoint0
```

### Test

```
python srez_main.py --run demo --dataset /home/Student/s4360417/honours/datasets/oasis3/exp2_jpg/train --dataset_train /home/Student/s4360417/honours/datasets/oasis3/exp2_jpg/train --dataset_test /home/Student/s4360417/honours/datasets/oasis3/exp2_jpg/valid --sample_train 960 --sample_test 240  --subsample_train -1 --subsample_test -1 --batch_size 4 --gene_mse_factor 0.975 --learning_beta 0.9 --learning_rate_start 0.000001 --learning_rate_half_life 10000 --sample_size 256 --sample_size_y 256 --summary_period 1000000 --train_time 60
```

```
python srez_main.py --run demo --dataset /home/Student/s4360417/honours/datasets/oasis3/exp3_png_2/ground_truth/train --dataset_train /home/Student/s4360417/honours/datasets/oasis3/exp3_png_2/ground_truth/train --dataset_test /home/Student/s4360417/honours/datasets/oasis3/exp3_png_2/ground_truth/test --sampling_pattern /home/Student/s4360417/honours/datasets/oasis3/exp3_png_2/artefact_fcs_2/mask.mat --sample_train 6400 --sample_test 32 --batch_size 4 --num_epoch 20 --gene_mse_factor 0.975 --learning_beta1 0.9 --learning_rate_start 0.000001 --learning_rate_half_life 10000 --sample_size 256 --sample_size_y 256 --summary_period 1600 --checkpoint_period -1
```

```
python srez_main.py --run demo --dataset /home/Student/s4360417/honours/datasets/oasis3/exp3_png_2/ground_truth/train --dataset_train /home/Student/s4360417/honours/datasets/oasis3/exp3_png_2/ground_truth/train --dataset_test /home/Student/s4360417/honours/datasets/oasis3/exp3_png_2/ground_truth/test --sampling_pattern /home/Student/s4360417/honours/datasets/oasis3/exp3_png_2/artefact_fcs_1/mask.mat --sample_train 6400 --sample_test 1280 --batch_size 2 --num_epoch 20 --gene_mse_factor 0.9 --learning_beta1 0.9 --learning_rate_start 1e-5 --learning_rate_half_life 10000 --sample_size 256 --sample_size_y 256 --summary_period 3200 --checkpoint_period -1 --subsample_test -1 --train_dir test --checkpoint_dir checkpoint
```

RGANCS with CV

```
python srez_main.py --run demo --dataset /home/Student/s4360417/honours/datasets/oasis3/fractal_cs/cv/0 --sampling_pattern /home/Student/s4360417/honours/datasets/oasis3/fractal_cs/artefact_fcs_4/mask.mat --sample_test 1600 --subsample_test 1600 --permutation_test True --sample_size 256 --sample_size_y 256 --batch_size 2 --train_dir test0 --checkpoint_dir checkpoint0 --summary_period 100
```
