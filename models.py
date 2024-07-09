#AlexNet consists of 5 convolution layers, 3 max-pooling layers, 2 Normalized layers, 2 fully connected layers and 1 SoftMax layer. 
import torch
import torch.nn as nn
__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    def __init__(self, no_of_classes):
        super(AlexNet, self).__init__()
        
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=1),  # N x 96 x 55 x 55,
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2), # N x 96 x 27 x 27,
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5),  # N x 256 x 23 x 23
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),  # N x 256 x 11 x 11
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=348, kernel_size=3),  # N x 348 x 9 x 9
            nn.ReLU(),
            nn.Conv2d(in_channels=348, out_channels=348, kernel_size=3), # N x 348 x 7 x 7
            nn.ReLU(),
            nn.Conv2d(in_channels=348, out_channels=256, kernel_size=3),  # N x 348 x 5 x 5
            nn.MaxPool2d(kernel_size=3, stride=2), # N x 256 x 2 x 2
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), #N x 1024
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256*2*2, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Softmax()
        )
        self.init_parameter()
        
    def init_parameter(self):
        for layer in self.convs:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.1),
                nn.init.constant_(layer.bias, 0)
            
        nn.init.constant_(self.convs[4].bias, 1)
        nn.init.constant_(self.convs[10].bias, 1)
        nn.init.constant_(self.convs[12].bias, 1)
        nn.init.constant_(self.classifier[2].bias, 1)
        nn.init.constant_(self.classifier[5].bias, 1)
        nn.init.constant_(self.classifier[7].bias, 1)
        
    
    def forward(self, x):
        x = self.convs(x)
        x = self.classifier(x)
        return x


import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from models import AlexNet
__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}
GPUS = [0]
EPOCHS = 90
NO_CLASSES = 1000
TRAIN_DIR = 'imagenet-mini/train'
VAL_DIR = 'imagenet-mini/val'
IMG_DIM = 227
BATCH_SIZE = 128
L_RATE = 0.01
W_DECAY = 0.0005
MOMENTUM = 0.9
CHECKPOINT_DIR = 'checkpoints/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = torch.initial_seed()
# create model
model = AlexNet(NO_CLASSES).to(device)

# train with multi GPU,,,,, if available
model = torch.nn.parallel.DataParallel(model, device_ids=GPUS)
print(model)

# image augmentation and transformation
data_transform = transforms.Compose([
    transforms.CenterCrop(IMG_DIM),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.CenterCrop(IMG_DIM),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# prepare the dataset
train_dataset = datasets.ImageFolder(TRAIN_DIR, data_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, val_transform)

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    # num_workers=8
)
val_loader = DataLoader(
    val_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    # num_workers=8
)

# optimizer
optim = torch.optim.SGD(
    model.module.parameters(),
    lr=L_RATE,
    momentum=MOMENTUM,
    weight_decay=W_DECAY
)
# loss function
criterion = nn.CrossEntropyLoss()

# decay the learning rate
optim = torch.optim.SGD(model.parameters(), lr=L_RATE, momentum=MOMENTUM, weight_decay=W_DECAY)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.1)
total_steps = 1
# training
for epoch in range(EPOCHS):
    for step, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        # refreshing gradients
        optim.zero_grad()
        # forward_pass
        pred = model(X)
        # taking loss
        loss = criterion(pred, y).to(device)
        # backward pass
        loss.backward()
        # taking step
        optim.step()
        
        if total_steps % 10 == 0:
            print(f'step: {total_steps} | Loss: {loss}')
        total_steps += 1
    
    # saving checkpoints
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model_checkpoint{epoch+1}.pkl')
    state = {
        'epoch': epoch,
        'total_steps': total_steps,
        'optimizer': optim.state_dict(),
        'model': model.module.state_dict(),
        'seed': seed
    }
    torch.save(state, checkpoint_path)