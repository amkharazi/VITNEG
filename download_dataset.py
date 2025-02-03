# Download Following Datasets and store them in ./Datasets
# 1. Cifar10
# 2. Mnist
# 3. TinyImageNet   
# 4. Cifar100


import os
import argparse
import torchvision.datasets as datasets
import wget
import zipfile

def download_cifar10(save_dir):
    print('Downloading CIFAR-10 dataset...')
    os.makedirs(save_dir, exist_ok=True)
    datasets.CIFAR10(root=save_dir, download=True)
    print('CIFAR-10 dataset downloaded successfully.')

def download_cifar100(save_dir):
    print('Downloading CIFAR-100 dataset...')
    os.makedirs(save_dir, exist_ok=True)
    datasets.CIFAR100(root=save_dir, download=True)
    print('CIFAR-100 dataset downloaded successfully.')

def download_mnist(save_dir):
    print('Downloading MNIST dataset...')
    os.makedirs(save_dir, exist_ok=True)
    datasets.MNIST(root=save_dir, download=True)
    print('MNIST dataset downloaded successfully.')

def download_tiny_imagenet(save_dir):
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
    
    # os.remove(download_path)
    
    print('TinyImageNet dataset downloaded successfully.')

def main(dataset=None, save_dir=None):
    if save_dir is None:
        save_dir = './datasets'
    if dataset is None:
        download_cifar10(save_dir)
        download_cifar100(save_dir)
        download_mnist(save_dir)
        download_tiny_imagenet(save_dir)
    elif dataset == 'cifar10':
        download_cifar10(save_dir)
    elif dataset == 'cifar100':
        download_cifar100(save_dir)
    elif dataset == 'mnist':
        download_mnist(save_dir)
    elif dataset == 'tiny_imagenet':
        download_tiny_imagenet(save_dir)
    else:
        print('Invalid dataset choice. Please choose from cifar10, cifar100, mnist, or tiny_imagenet.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download datasets.')
    parser.add_argument('--dataset', type=str, nargs='?', choices=['cifar10', 'mnist', 'cifar100', 'tiny_imagenet'], 
                        default=None,
                        help='Choose which dataset to download (cifar10, cifar100, mnist, or tiny_imagenet). If None, then all datasets will be downloaded')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Specify the directory to save the datasets. If None, then ./datasets is set as the path')
    args = parser.parse_args()
    main(args.dataset, args.save_dir)