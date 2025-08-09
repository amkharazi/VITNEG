import sys
sys.path.append('.')

import os, math, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import RandAugment, RandomErasing

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

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def topk_accuracy(logits, targets, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk); B = targets.size(0)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        return {k: correct[:k].reshape(-1).float().sum().item() / B for k in topk}

def mixup_data(x, y, alpha=0.8):
    if alpha <= 0: return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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

def get_train_loader(args):
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
        loader, _ = get_cifar10_dataloaders(args.data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count)

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
        loader, _ = get_cifar100_dataloaders(args.data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count)

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
        loader, _ = get_mnist_dataloaders(args.data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count)

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
        loader, _, _ = get_tinyimagenet_dataloaders(args.data_root, transform_train, transform_val, transform_test, batch_size, image_size, train_size, repeat_count)

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
        loader, _ = get_fashionmnist_dataloaders(args.data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count)

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
        loader, _ = get_flowers102_dataloaders(args.data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count)

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
            transformsNormalize := transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        # fix typo: implement without walrus (some envs older)
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        loader, _ = get_oxford_pets_dataloaders(args.data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count)

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
        loader, _ = get_stl10_classification_dataloaders(args.data_root, transform_train, transform_test, batch_size, image_size, train_size, repeat_count)

    else:
        raise ValueError("unknown dataset")

    return loader

def main():
    ap = argparse.ArgumentParser()
    # data / training
    ap.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10","cifar100","mnist","tinyimagenet","fashionmnist","flowers102","oxford_pets","stl10"])
    ap.add_argument("--data-root", type=str, default="./datasets")
    ap.add_argument("--image-size", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--mixup", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", type=str, default="./results/run1")
    ap.add_argument("--train-size", type=str, default="default")
    ap.add_argument("--repeat-count", type=int, default=5)
    ap.add_argument("--num-classes", type=int, required=True)

    # model selector + common
    ap.add_argument("--model", type=str, default="vit", choices=["vit","swin","cvt","pvt"])
    ap.add_argument("--softmax_type", type=str, default="vanilla", choices=["vanilla","neg"])
    # ViT hyperparams
    ap.add_argument("--vit-embed-dim", type=int, default=384)
    ap.add_argument("--vit-num-layers", type=int, default=8)
    ap.add_argument("--vit-num-heads", type=int, default=6)
    ap.add_argument("--vit-mlp-dim", type=int, default=1536)
    ap.add_argument("--vit-patch-size", type=int, default=None)
    # Swin hyperparams
    ap.add_argument("--swin-embed-dim", type=int, default=96)
    ap.add_argument("--swin-depths", type=str, default="2,2,6,2")
    ap.add_argument("--swin-num-heads", type=str, default="3,6,12,24")
    ap.add_argument("--swin-window-size", type=int, default=7)
    # CvT hyperparams
    ap.add_argument("--cvt-embed-dims", type=str, default="64,192,384")
    ap.add_argument("--cvt-depths", type=str, default="1,2,6")
    ap.add_argument("--cvt-num-heads", type=str, default="1,3,6")
    ap.add_argument("--cvt-kv-strides", type=str, default="1,2,2")
    # PVT hyperparams
    ap.add_argument("--pvt-embed-dims", type=str, default="64,128,320,512")
    ap.add_argument("--pvt-depths", type=str, default="2,2,2,2")
    ap.add_argument("--pvt-num-heads", type=str, default="1,2,5,8")
    ap.add_argument("--pvt-sr-ratios", type=str, default="8,4,2,1")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = get_train_loader(args)
    model = build_model(args).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(max(1, args.warmup_epochs))
        e = epoch - args.warmup_epochs
        T = max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * e / T))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    os.makedirs(os.path.join(args.save_dir, "model_stats"), exist_ok=True)
    log_path = os.path.join(args.save_dir, "train_log.txt")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    best_top1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_top1, running_cnt, running_loss = 0.0, 0, 0.0
        for images, labels in train_loader:
            images = images.to(device); labels = labels.to(device)
            if args.mixup and args.mixup > 0:
                images, y_a, y_b, lam = mixup_data(images, labels, alpha=args.mixup)
            optim.zero_grad(set_to_none=True)
            logits = model(images)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam) if args.mixup and args.mixup > 0 else criterion(logits, labels)
            loss.backward()
            optim.step()

            bs = images.size(0)
            acc = topk_accuracy(logits, labels, topk=(1,))[1]
            running_top1 += acc * bs
            running_cnt  += bs
            running_loss += loss.item() * bs

        scheduler.step()
        ep_top1 = running_top1 / max(1, running_cnt)
        ep_loss = running_loss / max(1, running_cnt)

        with open(log_path, "a") as f:
            f.write(f"{epoch},{ep_top1:.6f},{ep_loss:.6f}\n")

        if ep_top1 > best_top1:
            best_top1 = ep_top1
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_stats", "best.pth"))
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_stats", f"epoch_{epoch}.pth"))

if __name__ == "__main__":
    main()
