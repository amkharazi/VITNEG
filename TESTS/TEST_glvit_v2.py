# Add all .py files to path
import sys
sys.path.append('..')

# Import Libraries

from Models.glvit_v2 import VisionTransformer
from Utils.cifar10_loaders import get_cifar10_dataloaders
from Utils.cifar100_loaders import get_cifar100_dataloaders
from Utils.mnist_loaders import get_mnist_dataloaders
from Utils.tinyimagenet_loaders import get_tinyimagenet_dataloaders


from Utils.accuracy_measures import topk_accuracy
from Utils.num_parameters import count_parameters
import torchvision.transforms as transforms
from torch import nn
from torch import optim
import time
import torch
import os

import argparse

import numpy as np
import random

def set_seed(seed: int = 42):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False


def main(dataset = 'cifar10', 
        TEST_ID = 'Test_ID001',
         batch_size = 32,
         n_epoch = 200,
         image_size = 32,
         train_size = 'default',
         patch_size = 4,
         num_classes = 10,
         dim = 64,
         depth = 6,
         heads = 8,
         mlp_dim = 128,
         SEED = None):
    
    # Setup the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f'Device is set to : {device}')

    if SEED is None:
        print(f'No seed is set!')
    else:
        set_seed(seed=SEED)
        
    # Set up the vit model
    model = VisionTransformer(img_size=image_size,
                               patch_size=patch_size,
                                 in_channels=3,
                                   num_classes=num_classes,
                                     dim=dim,
                                       depth=depth,
                                         heads=heads,
                                           mlp_dim=mlp_dim,
                                             dropout=0.1).to(device)
    
    if dataset=='cifar10':
        cifar10_transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((image_size, image_size)), 
                transforms.RandomCrop(image_size, padding=5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        cifar10_transform_test = transforms.Compose([
                transforms.Resize((image_size, image_size)), 
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        train_loader, _ = get_cifar10_dataloaders(data_dir = '../datasets',
                                                    transform_train=cifar10_transform_train,
                                                    transform_test=cifar10_transform_test,
                                                    batch_size=batch_size,
                                                    image_size=image_size,
                                                    train_size=train_size)
    if dataset=='cifar100':
        cifar100_transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((image_size, image_size)), 
                transforms.RandomCrop(image_size, padding=5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        cifar100_transform_test = transforms.Compose([
                transforms.Resize((image_size, image_size)), 
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        train_loader, _ = get_cifar100_dataloaders(data_dir = '../datasets',
                                                    transform_train=cifar100_transform_train,
                                                    transform_test=cifar100_transform_test,
                                                    batch_size=batch_size,
                                                    image_size=image_size,
                                                    train_size=train_size)
    if dataset=='mnist':
        mnist_transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=2), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        mnist_transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.Grayscale(num_output_channels=3), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_loader, _ = get_mnist_dataloaders(data_dir = '../datasets',
                                                    transform_train=mnist_transform_train,
                                                    transform_test=mnist_transform_test,
                                                    batch_size=batch_size,
                                                    image_size=image_size,
                                                    train_size=train_size)
    if dataset=='tinyimagenet':
        tiny_transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((image_size, image_size)), 
            transforms.RandomCrop(image_size, padding=5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        tiny_transform_val = transforms.Compose([
                transforms.Resize((image_size, image_size)), 
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        tiny_transform_test = transforms.Compose([
                transforms.Resize((image_size, image_size)), 
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        
        train_loader, _,_ = get_tinyimagenet_dataloaders(data_dir = '../datasets',
                                                        transform_train=tiny_transform_train,
                                                        transform_val=tiny_transform_val,
                                                        transform_test=tiny_transform_test,
                                                        batch_size=batch_size,
                                                        image_size=image_size)
    
    num_parameters = count_parameters(model)
    print(f'This Model has {num_parameters} parameters')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    
    # Define train and test functions (use examples)
    def train_epoch(loader, epoch):
        model.train()
    
        start_time = time.time()
        running_loss = 0.0
        correct = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0} # set the initial correct count for top1-to-top5 accuracy

        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
        
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            accuracies = topk_accuracy(outputs, targets, topk=(1, 2, 3, 4, 5))
            for k in accuracies:
                correct[k] += accuracies[k]['correct']
            # print(f'batch{i} done!')

        elapsed_time = time.time() - start_time
        top1_acc, top2_acc, top3_acc, top4_acc, top5_acc = [(correct[k]/len(loader.dataset)) for k in correct]
        avg_loss = running_loss / len(loader.dataset)
    
        report_train = f'Train epoch {epoch}: top1%={top1_acc}, top2%={top2_acc}, top3%={top3_acc}, top4%={top4_acc}, top5%={top5_acc}, loss={avg_loss}, time={elapsed_time}s'
        print(report_train)

        return report_train

    
    # Set up the directories to save the results
    result_dir = os.path.join('../results', TEST_ID)
    result_subdir = os.path.join(result_dir, 'accuracy_stats')
    model_subdir = os.path.join(result_dir, 'model_stats')

    os.makedirs(result_subdir, exist_ok=True)
    os.makedirs(model_subdir, exist_ok=True)
    
    with open(os.path.join(result_dir, 'model_stats', 'model_info.txt'), 'a') as f:
        f.write(f'total number of parameters:\n{num_parameters}\n dataset is {dataset}\n seed is ${SEED}')

    # Train from Scratch - Just Train
    print(f'Training for {len(range(n_epoch))} epochs\n')
    for epoch in range(0+1,n_epoch+1):
        report_train = train_epoch(train_loader, epoch)
    
        report = report_train + '\n'
        if epoch % 5 == 0:
            model_path = os.path.join(result_dir, 'model_stats', f'Model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), model_path)
        with open(os.path.join(result_dir, 'accuracy_stats', 'report_train.txt'), 'a') as f:
            f.write(report)     

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Basic Experiment Settings - Train")
    
    # Add arguments to the parser
    parser.add_argument('--TEST_ID', type=str, help='Experiment test ID')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Experiment test ID')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--image_size', type=int, default=32, help='Image size (must be square / only width or height)')
    parser.add_argument('--train_size', type=str, default='default', help='Size of the training set')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size for the model')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--dim', type=int, default=64, help='Dimensionality of model features')
    parser.add_argument('--depth', type=int, default=6, help='Depth of the model')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--mlp_dim', type=int, default=128, help='MLP hidden layer dimension')
    parser.add_argument('--seed', type=int, default=None, help='The randomness seed')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function with the parsed arguments
    main(args.dataset, args.TEST_ID, args.batch_size, args.n_epoch, args.image_size, args.train_size,
         args.patch_size, args.num_classes, args.dim, args.depth, args.heads, args.mlp_dim,args.seed)


    


