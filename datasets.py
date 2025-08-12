"""
Dataset Downloader Utility

This script downloads and stores datasets in the specified directory (`./datasets` by default).
It supports downloading the following datasets:

- CIFAR-10
- CIFAR-100
- MNIST
- FashionMNIST
- STL10
- Oxford-IIIT Pet
- Flowers102
- Food101
- TinyImageNet

Usage:
    python download_datasets.py --dataset <dataset_name> --save_dir <directory>

If no dataset is specified, all datasets will be downloaded.

Example:
    python download_datasets.py --dataset cifar10
    python download_datasets.py --save_dir ./datasets
"""

import os
import argparse
import torchvision.datasets as datasets
import wget
import zipfile

def download_cifar10(save_dir: str) -> None:
    """Download CIFAR-10 dataset."""
    print('Downloading CIFAR-10 dataset...')
    os.makedirs(save_dir, exist_ok=True)
    datasets.CIFAR10(root=save_dir, download=True)
    print('CIFAR-10 dataset downloaded successfully.')

def download_cifar100(save_dir: str) -> None:
    """Download CIFAR-100 dataset."""
    print('Downloading CIFAR-100 dataset...')
    os.makedirs(save_dir, exist_ok=True)
    datasets.CIFAR100(root=save_dir, download=True)
    print('CIFAR-100 dataset downloaded successfully.')

def download_mnist(save_dir: str) -> None:
    """Download MNIST dataset."""
    print('Downloading MNIST dataset...')
    os.makedirs(save_dir, exist_ok=True)
    datasets.MNIST(root=save_dir, download=True)
    print('MNIST dataset downloaded successfully.')

def download_fashion_mnist(save_dir: str) -> None:
    """Download FashionMNIST dataset."""
    print('Downloading FashionMNIST dataset...')
    os.makedirs(save_dir, exist_ok=True)
    datasets.FashionMNIST(root=save_dir, download=True)
    print('FashionMNIST dataset downloaded successfully.')

def download_stl10(save_dir: str) -> None:
    """Download STL10 dataset (train, test, and unlabeled splits)."""
    print('Downloading STL10 dataset...')
    os.makedirs(save_dir, exist_ok=True)
    datasets.STL10(root=save_dir, split='train', download=True)
    datasets.STL10(root=save_dir, split='test', download=True)
    datasets.STL10(root=save_dir, split='unlabeled', download=True)
    print('STL10 dataset downloaded successfully.')

def download_oxford_pets(save_dir: str) -> None:
    """Download Oxford-IIIT Pet dataset."""
    print('Downloading Oxford-IIIT Pet dataset...')
    os.makedirs(save_dir, exist_ok=True)
    datasets.OxfordIIITPet(root=save_dir, download=True)
    print('Oxford-IIIT Pet dataset downloaded successfully.')

def download_flowers102(save_dir: str) -> None:
    """Download Flowers102 dataset."""
    print('Downloading Flowers102 dataset...')
    os.makedirs(save_dir, exist_ok=True)
    datasets.Flowers102(root=save_dir, download=True)
    print('Flowers102 dataset downloaded successfully.')

def download_food101(save_dir: str) -> None:
    """Download Food101 dataset."""
    print('Downloading Food101 dataset...')
    os.makedirs(save_dir, exist_ok=True)
    datasets.Food101(root=save_dir, download=True)
    print('Food101 dataset downloaded successfully.')

def download_tiny_imagenet(save_dir: str) -> None:
    """
    Download TinyImageNet dataset from the official Stanford CS231n link
    and extract it to the specified directory.
    """
    print('Downloading TinyImageNet dataset...')
    os.makedirs(save_dir, exist_ok=True)
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    download_path = os.path.join(save_dir, 'tiny-imagenet-200.zip')

    if not os.path.exists(download_path):
        try:
            wget.download(url, out=download_path)
        except Exception as e:
            print(f'Error downloading the file: {e}')
            return

    if not os.path.exists(os.path.join(save_dir, 'tiny-imagenet-200')):
        try:
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(save_dir)
        except Exception as e:
            print(f'Error extracting the zip file: {e}')
            return

    print('TinyImageNet dataset downloaded successfully.')

def main(dataset: str = None, save_dir: str = None) -> None:
    """
    Download the specified dataset or all datasets if none is specified.

    Parameters
    ----------
    dataset : str, optional
        Name of the dataset to download.
        If None, all supported datasets are downloaded.
    save_dir : str, optional
        Directory where the datasets will be stored.
        Defaults to './datasets'.
    """
    if save_dir is None:
        save_dir = './datasets'
    if dataset is None:
        download_cifar10(save_dir)
        download_cifar100(save_dir)
        download_mnist(save_dir)
        download_fashion_mnist(save_dir)
        download_stl10(save_dir)
        download_oxford_pets(save_dir)
        download_flowers102(save_dir)
        download_food101(save_dir)
        download_tiny_imagenet(save_dir)
    elif dataset == 'cifar10':
        download_cifar10(save_dir)
    elif dataset == 'cifar100':
        download_cifar100(save_dir)
    elif dataset == 'mnist':
        download_mnist(save_dir)
    elif dataset == 'fashionmnist':
        download_fashion_mnist(save_dir)
    elif dataset == 'stl10':
        download_stl10(save_dir)
    elif dataset == 'oxford_pets':
        download_oxford_pets(save_dir)
    elif dataset == 'flowers102':
        download_flowers102(save_dir)
    elif dataset == 'food101':
        download_food101(save_dir)
    elif dataset == 'tiny_imagenet':
        download_tiny_imagenet(save_dir)
    else:
        print('Invalid dataset choice. Please choose from cifar10, cifar100, mnist, fashionmnist, '
              'stl10, oxford_pets, flowers102, food101, or tiny_imagenet.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download datasets.')
    parser.add_argument(
        '--dataset',
        type=str,
        nargs='?',
        choices=[
            'cifar10', 'mnist', 'cifar100', 'tiny_imagenet',
            'fashionmnist', 'stl10', 'oxford_pets', 'flowers102', 'food101'
        ],
        default=None,
        help='Choose which dataset to download. If None, all will be downloaded.'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save the datasets. Defaults to ./datasets.'
    )
    args = parser.parse_args()
    main(args.dataset, args.save_dir)
