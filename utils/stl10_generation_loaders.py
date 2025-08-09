import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import STL10

def get_stl10_unlabeled_dataloaders(
    data_dir='../datasets',
    transform=None,
    batch_size=64,
    image_size=192,
    train_size='default'
):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))
        ])

    full_dataset = STL10(root=data_dir, split='unlabeled', transform=transform, download=False)

    if train_size != 'default':
        train_size = int(train_size)
        val_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, val_size])
    else:
        train_dataset = full_dataset
        test_dataset = []

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None

    return train_loader, test_loader
