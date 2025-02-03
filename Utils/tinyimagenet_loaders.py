import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
import torchvision.datasets as datasets


def label_names_all(root_dir = '../datasets'):
    path = os.path.join(root_dir, 'tiny-imagenet-200/words.txt')
    labels = {}
    with open(path, 'r') as file:
        for line in file:
            key, value = line.strip().split('\t')
            labels[key] = value
    return labels

def label_names_train(root_dir = '../datasets'):
    path = os.path.join(root_dir, 'tiny-imagenet-200/train')
    labels = {}
    indices = {}
    idx = 0
    description = label_names_all(root_dir = root_dir)
    for item in sorted(os.listdir(path)):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            labels[item] = [idx, description[item]]
            indices[idx] = [item, description[item]]
            idx += 1
    return labels, indices

class test_dataset(Dataset):
    def __init__(self, root_dir = '../datasets', transform=None):
        self.root_dir = os.path.join(root_dir, 'tiny-imagenet-200/test/images') 
        self.transform = transform
        self.image_files = sorted(os.listdir(self.root_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        return image
    


class val_dataset(Dataset):
    def __init__(self, root_dir='../datasets', transform=None):
        self.root_dir = os.path.join(root_dir, 'tiny-imagenet-200/val/images')
        self.labels_name, self.indices = label_names_train(root_dir = root_dir)
        self.transform = transform
        self.labels_file = os.path.join(root_dir, 'tiny-imagenet-200/val/val_annotations.txt')
        self.labels = self.load_labels(self.labels_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name, label = self.labels[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def load_labels(self, labels_file):
        labels = []
        with open(labels_file, 'r') as file:
            for line in file:
                line_parts = line.strip().split()
                img_name = line_parts[0]
                label = line_parts[1]
                label = self.labels_name[label][0]
                labels.append((img_name, label))
        return labels


def get_tinyimagenet_dataloaders(data_dir='../datasets', transform_train=None, transform_val=None, transform_test=None, batch_size=64, image_size=192, train_size= 'default'):
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.Resize((image_size, image_size)), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    if transform_val is None:
        transform_val = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    dataset_train = datasets.ImageFolder(root=os.path.join(data_dir,'tiny-imagenet-200/train'), transform=transform_train)
    dataset_val = val_dataset(root_dir=data_dir, transform=transform_val)
    dataset_test = test_dataset(root_dir=data_dir, transform=transform_test)
    
    if train_size!='default':
        total_train = len(dataset_train)
        temp_val_size = total_train- int(train_size)
        dataset_train, dataset_temp_val = random_split(dataset_train, [int(train_size), temp_val_size])
        dataset_val = ConcatDataset([dataset_temp_val, dataset_val])

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader   