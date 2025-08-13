# ID001 - ViT Base - Vanilla
python train_classifiers.py --dataset cifar100 --data-root ./datasets --epochs 5 --batch-size 128 --softmax_type vanilla --num-classes 100 --save-dir ./results/cifar100/dummy_vit_base_cifar100 --model vit --image-size 224 --vit-patch-size 4 --vit-embed-dim 768 --vit-num-layers 12 --vit-num-heads 12 --vit-mlp-dim 3072
python test_classifiers.py --dataset cifar100 --data-root ./datasets --batch-size 128 --softmax_type vanilla --num-classes 100 --ckpt ./results/cifar100/dummy_vit_base_cifar100/model_stats/best.pth --out-txt ./results/cifar100/dummy_vit_base_cifar100/test_results.txt --model vit --image-size 224 --vit-patch-size 4 --vit-embed-dim 768 --vit-num-layers 12 --vit-num-heads 12 --vit-mlp-dim 3072

# ID002 - ViT Tiny - Vanilla
python train_classifiers.py --dataset cifar100 --data-root ./datasets --epochs 5 --batch-size 256 --softmax_type vanilla --num-classes 100 --save-dir ./results/cifar100/dummy_vit_tiny_cifar100 --model vit --image-size 224 --vit-patch-size 4 --vit-embed-dim 64 --vit-num-layers 6 --vit-num-heads 8 --vit-mlp-dim 128
python test_classifiers.py --dataset cifar100 --data-root ./datasets --batch-size 256 --softmax_type vanilla --num-classes 100 --ckpt ./results/cifar100/dummy_vit_tiny_cifar100/model_stats/best.pth --out-txt ./results/cifar100/dummy_vit_tiny_cifar100/test_results.txt --model vit --image-size 224 --vit-patch-size 4 --vit-embed-dim 64 --vit-num-layers 6 --vit-num-heads 8 --vit-mlp-dim 128

# ID003 - ViT Small - Vanilla
python train_classifiers.py --dataset cifar100 --data-root ./datasets --epochs 5 --batch-size 256 --softmax_type vanilla --num-classes 100 --save-dir ./results/cifar100/dummy_vit_small_cifar100 --model vit --image-size 224 --vit-patch-size 4 --vit-embed-dim 192 --vit-num-layers 9 --vit-num-heads 12 --vit-mlp-dim 384
python test_classifiers.py --dataset cifar100 --data-root ./datasets --batch-size 256 --softmax_type vanilla --num-classes 100 --ckpt ./results/cifar100/dummy_vit_small_cifar100/model_stats/best.pth --out-txt ./results/cifar100/dummy_vit_small_cifar100/test_results.txt --model vit --image-size 224 --vit-patch-size 4 --vit-embed-dim 192 --vit-num-layers 9 --vit-num-heads 12 --vit-mlp-dim 384

# ID004 - Swin Tiny - Vanilla
python train_classifiers.py --dataset cifar100 --data-root ./datasets --epochs 5 --batch-size 128 --softmax_type vanilla --num-classes 100 --save-dir ./results/cifar100/dummy_swin_tiny_cifar100 --model swin --image-size 224 --swin-embed-dim 96 --swin-depths 2,2,6,2 --swin-num-heads 3,6,12,24 --swin-window-size 7
python test_classifiers.py --dataset cifar100 --data-root ./datasets --batch-size 128 --softmax_type vanilla --num-classes 100 --ckpt ./results/cifar100/dummy_swin_tiny_cifar100/model_stats/best.pth --out-txt ./results/cifar100/dummy_swin_tiny_cifar100/test_results.txt --model swin --image-size 224 --swin-embed-dim 96 --swin-depths 2,2,6,2 --swin-num-heads 3,6,12,24 --swin-window-size 7

# ID005 - CvT Tiny - Vanilla
python train_classifiers.py --dataset cifar100 --data-root ./datasets --epochs 5 --batch-size 128 --softmax_type vanilla --num-classes 100 --save-dir ./results/cifar100/dummy_cvt_tiny_cifar100 --model cvt --image-size 224 --cvt-embed-dims 64,192,384 --cvt-depths 1,2,6 --cvt-num-heads 1,3,6 --cvt-kv-strides 1,2,2
python test_classifiers.py --dataset cifar100 --data-root ./datasets --batch-size 128 --softmax_type vanilla --num-classes 100 --ckpt ./results/cifar100/dummy_cvt_tiny_cifar100/model_stats/best.pth --out-txt ./results/cifar100/dummy_cvt_tiny_cifar100/test_results.txt --model cvt --image-size 224 --cvt-embed-dims 64,192,384 --cvt-depths 1,2,6 --cvt-num-heads 1,3,6 --cvt-kv-strides 1,2,2

# ID006 - PvT Tiny - Vanilla
python train_classifiers.py --dataset cifar100 --data-root ./datasets --epochs 5 --batch-size 128 --softmax_type vanilla --num-classes 100 --save-dir ./results/cifar100/dummy_pvt_tiny_cifar100 --model pvt --image-size 224 --pvt-embed-dims 64,128,320,512 --pvt-depths 2,2,2,2 --pvt-num-heads 1,2,5,8 --pvt-sr-ratios 8,4,2,1
python test_classifiers.py --dataset cifar100 --data-root ./datasets --batch-size 128 --softmax_type vanilla --num-classes 100 --ckpt ./results/cifar100/dummy_pvt_tiny_cifar100/model_stats/best.pth --out-txt ./results/cifar100/dummy_pvt_tiny_cifar100/test_results.txt --model pvt --image-size 224 --pvt-embed-dims 64,128,320,512 --pvt-depths 2,2,2,2 --pvt-num-heads 1,2,5,8 --pvt-sr-ratios 8,4,2,1

# ID007 - ViT Base - Neg
python train_classifiers.py --dataset cifar100 --data-root ./datasets --epochs 5 --batch-size 128 --softmax_type neg --num-classes 100 --save-dir ./results/cifar100/dummy_vit_base_neg_cifar100 --model vit --image-size 224 --vit-patch-size 4 --vit-embed-dim 768 --vit-num-layers 12 --vit-num-heads 12 --vit-mlp-dim 3072
python test_classifiers.py --dataset cifar100 --data-root ./datasets --batch-size 128 --softmax_type neg --num-classes 100 --ckpt ./results/cifar100/dummy_vit_base_neg_cifar100/model_stats/best.pth --out-txt ./results/cifar100/dummy_vit_base_neg_cifar100/test_results.txt --model vit --image-size 224 --vit-patch-size 4 --vit-embed-dim 768 --vit-num-layers 12 --vit-num-heads 12 --vit-mlp-dim 3072

# ID008 - ViT Tiny - Neg
python train_classifiers.py --dataset cifar100 --data-root ./datasets --epochs 5 --batch-size 256 --softmax_type neg --num-classes 100 --save-dir ./results/cifar100/dummy_vit_tiny_neg_cifar100 --model vit --image-size 224 --vit-patch-size 4 --vit-embed-dim 64 --vit-num-layers 6 --vit-num-heads 8 --vit-mlp-dim 128
python test_classifiers.py --dataset cifar100 --data-root ./datasets --batch-size 256 --softmax_type neg --num-classes 100 --ckpt ./results/cifar100/dummy_vit_tiny_neg_cifar100/model_stats/best.pth --out-txt ./results/cifar100/dummy_vit_tiny_neg_cifar100/test_results.txt --model vit --image-size 224 --vit-patch-size 4 --vit-embed-dim 64 --vit-num-layers 6 --vit-num-heads 8 --vit-mlp-dim 128

# ID009 - ViT Small - Neg
python train_classifiers.py --dataset cifar100 --data-root ./datasets --epochs 5 --batch-size 256 --softmax_type neg --num-classes 100 --save-dir ./results/cifar100/dummy_vit_small_neg_cifar100 --model vit --image-size 224 --vit-patch-size 4 --vit-embed-dim 192 --vit-num-layers 9 --vit-num-heads 12 --vit-mlp-dim 384
python test_classifiers.py --dataset cifar100 --data-root ./datasets --batch-size 256 --softmax_type neg --num-classes 100 --ckpt ./results/cifar100/dummy_vit_small_neg_cifar100/model_stats/best.pth --out-txt ./results/cifar100/dummy_vit_small_neg_cifar100/test_results.txt --model vit --image-size 224 --vit-patch-size 4 --vit-embed-dim 192 --vit-num-layers 9 --vit-num-heads 12 --vit-mlp-dim 384

# ID010 - Swin Tiny - Neg
python train_classifiers.py --dataset cifar100 --data-root ./datasets --epochs 5 --batch-size 128 --softmax_type neg --num-classes 100 --save-dir ./results/cifar100/dummy_swin_tiny_neg_cifar100 --model swin --image-size 224 --swin-embed-dim 96 --swin-depths 2,2,6,2 --swin-num-heads 3,6,12,24 --swin-window-size 7
python test_classifiers.py --dataset cifar100 --data-root ./datasets --batch-size 128 --softmax_type neg --num-classes 100 --ckpt ./results/cifar100/dummy_swin_tiny_neg_cifar100/model_stats/best.pth --out-txt ./results/cifar100/dummy_swin_tiny_neg_cifar100/test_results.txt --model swin --image-size 224 --swin-embed-dim 96 --swin-depths 2,2,6,2 --swin-num-heads 3,6,12,24 --swin-window-size 7

# ID011 - CvT Tiny - Neg
python train_classifiers.py --dataset cifar100 --data-root ./datasets --epochs 5 --batch-size 128 --softmax_type neg --num-classes 100 --save-dir ./results/cifar100/dummy_cvt_tiny_neg_cifar100 --model cvt --image-size 224 --cvt-embed-dims 64,192,384 --cvt-depths 1,2,6 --cvt-num-heads 1,3,6 --cvt-kv-strides 1,2,2
python test_classifiers.py --dataset cifar100 --data-root ./datasets --batch-size 128 --softmax_type neg --num-classes 100 --ckpt ./results/cifar100/dummy_cvt_tiny_neg_cifar100/model_stats/best.pth --out-txt ./results/cifar100/dummy_cvt_tiny_neg_cifar100/test_results.txt --model cvt --image-size 224 --cvt-embed-dims 64,192,384 --cvt-depths 1,2,6 --cvt-num-heads 1,3,6 --cvt-kv-strides 1,2,2

# ID012 - PvT Tiny - Neg
python train_classifiers.py --dataset cifar100 --data-root ./datasets --epochs 5 --batch-size 128 --softmax_type neg --num-classes 100 --save-dir ./results/cifar100/dummy_pvt_tiny_neg_cifar100 --model pvt --image-size 224 --pvt-embed-dims 64,128,320,512 --pvt-depths 2,2,2,2 --pvt-num-heads 1,2,5,8 --pvt-sr-ratios 8,4,2,1
python test_classifiers.py --dataset cifar100 --data-root ./datasets --batch-size 128 --softmax_type neg --num-classes 100 --ckpt ./results/cifar100/dummy_pvt_tiny_neg_cifar100/model_stats/best.pth --out-txt ./results/cifar100/dummy_pvt_tiny_neg_cifar100/test_results.txt --model pvt --image-size 224 --pvt-embed-dims 64,128,320,512 --pvt-depths 2,2,2,2 --pvt-num-heads 1,2,5,8 --pvt-sr-ratios 8,4,2,1
