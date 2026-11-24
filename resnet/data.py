import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

cifar_train = datasets.CIFAR10(root='../data', train=True, download=True) #Dataset download
cifar_scale = cifar_train.data/255 #Scale to 0-1
mean = cifar_scale.mean(axis=(0,1,2)) #Mean per color
std = cifar_scale.std(axis=(0,1,2)) #Standard deviation per color

train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), #Padd and crop 32x32
                               transforms.RandomHorizontalFlip(), #Flip horizontally
                               transforms.ToTensor(), #Convert to Tensor from PIL
                               transforms.Normalize(mean, std)]) #Norm

test_tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean, std)])

train_aug = datasets.CIFAR10(root='../data', train=True, download=True, transform=train_tf) #Training copy with augmentation
train_plain = datasets.CIFAR10(root='../data', train=True, download=True, transform=test_tf) #Validation copy without augmentation
test_set = datasets.CIFAR10(root='../data', train=False, download=True, transform=test_tf) #10,000 test images, no augmentation

rng = np.random.default_rng(0) #Random number generator with seed 0
idx = rng.permutation(50000) #0-49,999 shuffled numbers
train_idx = idx[:45000].tolist() #First 45000 for training
val_idx = idx[45000:].tolist() #Last 5000 for validation
train_set = Subset(train_aug, train_idx)
val_set = Subset(train_plain, val_idx)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0) #Training batch loader
val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0) #Val batch loader
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0) #Test batch loader
