import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.datasets import Flowers102

def get_flowers102_dataloaders(
    data_dir='../datasets',
    transform_train=None,
    transform_test=None,
    batch_size=64,
    image_size=192,
    train_size='default',
    repeat_count=3
):
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.Resize((image_size, image_size)), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))
        ])
    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))
        ])

    train_split = Flowers102(root=data_dir, split='train', transform=transform_train, download=False)
    val_split = Flowers102(root=data_dir, split='val', transform=transform_train, download=False)
    train_dataset = ConcatDataset([train_split, val_split])

    test_dataset = Flowers102(root=data_dir, split='test', transform=transform_test, download=False)

    if train_size != 'default':
        total_train = len(train_dataset)
        temp_test_size = total_train - int(train_size)
        train_dataset, dataset_temp_test = random_split(train_dataset, [int(train_size), temp_test_size])
        test_dataset = ConcatDataset([dataset_temp_test, test_dataset])

    repeated_train_dataset = ConcatDataset([train_dataset] * repeat_count)

    train_loader = DataLoader(repeated_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Example usage:
# train_loader, test_loader = get_flowers102_dataloaders(data_dir='./dataset', image_size=192)
