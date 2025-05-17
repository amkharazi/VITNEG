import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import OxfordIIITPet

def get_oxford_pets_dataloaders(
    data_dir='../datasets',
    transform_train=None,
    transform_test=None,
    batch_size=64,
    image_size=192,
    train_size='default'
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

    basic_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


    full_dataset = OxfordIIITPet(
        root=data_dir,
        transform=basic_transform,
        download=False
    )

    total_size = len(full_dataset)

    if train_size != 'default':
        train_size = int(train_size)
        temp_test_size = total_size - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, temp_test_size])
    else:
        train_size_default = int(0.8 * total_size)
        test_size_default = total_size - train_size_default
        train_dataset, test_dataset = random_split(full_dataset, [train_size_default, test_size_default])

    train_dataset.dataset.transform = transform_train
    test_dataset.dataset.transform = transform_test

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
