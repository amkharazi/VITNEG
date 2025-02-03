# How to Run ? 


To run a test, simple open the directory inside your console or terminal and run

```bash
python  TEST_basic.py    --TEST_ID TEST_ID001    --dataset cifar10   --batch_size 32 --n_epoch 10    --image_size 32 --train_size 40000  --patch_size 4  --num_classes 10    --dim 64    --depth 6   --heads 8   --mlp_dim 128 
```

> Make sure you change the TEST_ID as it will overwrite your previous results.

> You also need the datasets, so please use :
```bash
python download_dataset.py
```
