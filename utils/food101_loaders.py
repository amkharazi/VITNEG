import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.datasets import Food101

def get_food101_dataloaders(
    data_dir: str = '../datasets',
    transform_train=None,
    transform_test=None,
    batch_size: int = 64,
    image_size: int = 192,
    train_size: str = 'default',
    repeat_count: int = 3
):
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])

    train_dataset = Food101(root=data_dir, split='train', transform=transform_train, download=False)
    test_dataset = Food101(root=data_dir, split='test', transform=transform_test, download=False)

    if train_size != 'default':
        total_train = len(train_dataset)
        keep = int(train_size)
        keep = max(0, min(keep, total_train))
        leftover = total_train - keep
        train_dataset, temp_extra = random_split(train_dataset, [keep, leftover])
        if leftover > 0:
            test_dataset = ConcatDataset([temp_extra, test_dataset])

    repeated_train_dataset = ConcatDataset([train_dataset] * int(max(1, repeat_count)))

    train_loader = DataLoader(repeated_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
