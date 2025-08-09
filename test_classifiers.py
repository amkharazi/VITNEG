import sys
sys.path.append('.')

import os, argparse, torch
import torch.nn as nn
import torchvision.transforms as transforms

# models
from models.vit import vit
from models.swin import swin
from models.cvt import cvt
from models.pvt import pvt

# your loaders
from utils.cifar10_loaders import get_cifar10_dataloaders
from utils.cifar100_loaders import get_cifar100_dataloaders
from utils.mnist_loaders import get_mnist_dataloaders
from utils.tinyimagenet_loaders import get_tinyimagenet_dataloaders
from utils.fashionmnist_loaders import get_fashionmnist_dataloaders
from utils.flowers102_loaders import get_flowers102_dataloaders
from utils.oxford_pets_loaders import get_oxford_pets_dataloaders
from utils.stl10_classification_loaders import get_stl10_classification_dataloaders

def topk_accuracy(logits, targets, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk); B = targets.size(0)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        return {k: correct[:k].reshape(-1).float().sum().item() / B for k in topk}

def parse_int_tuple(s: str):
    return tuple(int(x) for x in s.split(",")) if isinstance(s, str) else tuple(s)

def build_model(args):
    im = args.image_size
    if args.model == "vit":
        patch = args.vit_patch_size if args.vit_patch_size is not None else (4 if im <= 64 else 16)
        return vit(
            input_size=(3, im, im),
            patch_size=patch,
            num_classes=args.num_classes,
            embed_dim=args.vit_embed_dim,
            num_heads=args.vit_num_heads,
            num_layers=args.vit_num_layers,
            mlp_dim=args.vit_mlp_dim,
            softmax_type=args.softmax_type
        )
    elif args.model == "swin":
        depths = parse_int_tuple(args.swin_depths)
        heads  = parse_int_tuple(args.swin_num_heads)
        return swin(
            img_size=im,
            patch_size=4,
            num_classes=args.num_classes,
            embed_dim=args.swin_embed_dim,
            depths=depths,
            num_heads=heads,
            window_size=args.swin_window_size,
            mlp_ratio=4.0,
            softmax_type=args.softmax_type
        )
    elif args.model == "cvt":
        embed_dims = parse_int_tuple(args.cvt_embed_dims)
        depths     = parse_int_tuple(args.cvt_depths)
        heads      = parse_int_tuple(args.cvt_num_heads)
        kv_strides = parse_int_tuple(args.cvt_kv_strides)
        return cvt(
            img_size=im,
            in_chans=3,
            num_classes=args.num_classes,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=heads,
            kv_strides=kv_strides,
            softmax_type=args.softmax_type
        )
    elif args.model == "pvt":
        embed_dims = parse_int_tuple(args.pvt_embed_dims)
        depths     = parse_int_tuple(args.pvt_depths)
        heads      = parse_int_tuple(args.pvt_num_heads)
        sr         = parse_int_tuple(args.pvt_sr_ratios)
        return pvt(
            img_size=im,
            in_chans=3,
            num_classes=args.num_classes,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=heads,
            sr_ratios=sr,
            softmax_type=args.softmax_type
        )
    else:
        raise ValueError("model must be one of: vit, swin, cvt, pvt")

def get_test_loader(args):
    ds = args.dataset.lower()
    image_size = args.image_size
    batch_size = args.batch_size
    train_size = args.train_size
    repeat_count = args.repeat_count

    if ds == 'cifar10':
        transform_train = transforms.Compose([
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            RandomErasing(p=0.25)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        _, test_loader = get_cifar10_dataloaders(args.data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count)

    elif ds == 'cifar100':
        transform_train = transforms.Compose([
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            RandomErasing(p=0.25)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        _, test_loader = get_cifar100_dataloaders(args.data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count)

    elif ds == 'mnist':
        transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            RandomErasing(p=0.25)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        _, test_loader = get_mnist_dataloaders(args.data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count)

    elif ds == 'tinyimagenet':
        transform_train = transforms.Compose([
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            RandomErasing(p=0.25)
        ])
        transform_val = transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        _, _, test_loader = get_tinyimagenet_dataloaders(args.data_root, transform_train, transform_val, transform_test, batch_size, image_size, train_size, repeat_count)

    elif ds == 'fashionmnist':
        transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            RandomErasing(p=0.25)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        _, test_loader = get_fashionmnist_dataloaders(args.data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count)

    elif ds == 'flowers102':
        transform_train = transforms.Compose([
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            RandomErasing(p=0.25)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        _, test_loader = get_flowers102_dataloaders(args.data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count)

    elif ds == 'oxford_pets':
        transform_train = transforms.Compose([
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            RandomErasing(p=0.25)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        _, test_loader = get_oxford_pets_dataloaders(args.data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count)

    elif ds == 'stl10':
        transform_train = transforms.Compose([
            RandAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            RandomErasing(p=0.25)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        _, test_loader = get_stl10_classification_dataloaders(args.data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count)

    else:
        raise ValueError("unknown dataset")

    return test_loader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10","cifar100","mnist","tinyimagenet","fashionmnist","flowers102","oxford_pets","stl10"])
    ap.add_argument("--data-root", type=str, default="./datasets")
    ap.add_argument("--image-size", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out-txt", type=str, default="./results/test_results.txt")
    ap.add_argument("--num-classes", type=int, required=True)

    ap.add_argument("--model", type=str, default="vit", choices=["vit","swin","cvt","pvt"])
    ap.add_argument("--softmax_type", type=str, default="vanilla", choices=["vanilla","neg"])
    # ViT
    ap.add_argument("--vit-embed-dim", type=int, default=384)
    ap.add_argument("--vit-num-layers", type=int, default=8)
    ap.add_argument("--vit-num-heads", type=int, default=6)
    ap.add_argument("--vit-mlp-dim", type=int, default=1536)
    ap.add_argument("--vit-patch-size", type=int, default=None)
    # Swin
    ap.add_argument("--swin-embed-dim", type=int, default=96)
    ap.add_argument("--swin-depths", type=str, default="2,2,6,2")
    ap.add_argument("--swin-num-heads", type=str, default="3,6,12,24")
    ap.add_argument("--swin-window-size", type=int, default=7)
    # CvT
    ap.add_argument("--cvt-embed-dims", type=str, default="64,192,384")
    ap.add_argument("--cvt-depths", type=str, default="1,2,6")
    ap.add_argument("--cvt-num-heads", type=str, default="1,3,6")
    ap.add_argument("--cvt-kv-strides", type=str, default="1,2,2")
    # PVT
    ap.add_argument("--pvt-embed-dims", type=str, default="64,128,320,512")
    ap.add_argument("--pvt-depths", type=str, default="2,2,2,2")
    ap.add_argument("--pvt-num-heads", type=str, default="1,2,5,8")
    ap.add_argument("--pvt-sr-ratios", type=str, default="8,4,2,1")

    # keep compatibility with loader signatures
    ap.add_argument("--train-size", type=str, default="default")
    ap.add_argument("--repeat-count", type=int, default=5)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    test_loader = get_test_loader(args)

    tot1 = tot5 = n = 0
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device); labels = labels.to(device)
            logits = model(images)
            accs = topk_accuracy(logits, labels, topk=(1,5))
            bs = images.size(0)
            tot1 += accs[1] * bs; tot5 += accs[5] * bs; n += bs
            loss_sum += criterion(logits, labels).item() * bs

    top1 = tot1 / max(1, n)
    top5 = tot5 / max(1, n)
    avg_loss = loss_sum / max(1, n)

    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    with open(args.out_txt, "a") as f:
        f.write(f"ckpt={args.ckpt}, dataset={args.dataset}, model={args.model}, softmax_type={args.softmax_type}, image_size={args.image_size}, top1={top1:.6f}, top5={top5:.6f}, loss={avg_loss:.6f}\n")

if __name__ == "__main__":
    main()
