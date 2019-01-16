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

### Test

```
python srez_main.py --run demo --dataset /home/Student/s4360417/honours/datasets/oasis3/exp2_jpg/train --dataset_train /home/Student/s4360417/honours/datasets/oasis3/exp2_jpg/train --dataset_test /home/Student/s4360417/honours/datasets/oasis3/exp2_jpg/valid --sample_train 960 --sample_test 240  --subsample_train -1 --subsample_test -1 --batch_size 4 --gene_mse_factor 0.975 --learning_beta 0.9 --learning_rate_start 0.000001 --learning_rate_half_life 10000 --sample_size 256 --sample_size_y 256 --summary_period 1000000 --train_time 60
```
