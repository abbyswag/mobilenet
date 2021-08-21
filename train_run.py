import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import StepLR


from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from models.mobilenetv1 import MobileNetv1
from models.mobilenetv2 import MobileNetV2
from models.mobilenetv3 import MobileNetV3
from models.weights_init import weights_init
from metrics import compute_accuracy

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--version', choices=['v1', 'v2', 'v3'], default='v3')
parser.add_argument('-b', '--batch_size', default=64,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate',
                    default=0.01, type=float, help='initial learning rate')
parser.add_argument('-epoch', '--max_epoch', default=100,
                    type=int, help='max epoch for training')
parser.add_argument('--save_folder', default='img/')
parser.add_argument('--save_img', default=True, help='save test images')
parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
parser.add_argument('--weight_decay', default=5e-4,
                     type=float, help='Weight decay for SGD') #4e-5
parser.add_argument('--momentum', default=0.999, type=float, help='momentum')
parser.add_argument('--load', default=False, help='load model')
parser.add_argument('--mobilenet', help='pretrained model')
args = parser.parse_args()

device = torch.device("cuda")
print(device)
sys.stdout.flush()
test_save_dir = args.save_folder
if not os.path.exists(test_save_dir):
    os.makedirs(test_save_dir)
    
    
##LOAD DATA
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_val = transforms.Compose([
        transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers}

train_dataset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, **params)
val_dataset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_val)
val_loader = torch.utils.data.DataLoader(val_dataset, **params)


cfgv2= [
            [1, 16, 1, 1], # t, c, n, s
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

cfgv3 =  mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]

if args.version == 'v1':
    model =  MobileNetv1().to(device)
elif args.version == 'v2':  
    model = MobileNetV2(cfgv2).to(device)
elif args.version == 'v3':  
    model =  MobileNetV3(cfgv3).to(device)
model.apply(weights_init)

if args.load:
    # load network
    model_path = args.mobilenet
    print('Loading resume network', model_path)
    model.load_state_dict(torch.load(model_path))
    
criterion = nn.CrossEntropyLoss().to(device)
opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# opt = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(opt, step_size=10, gamma=0.5)

def train(model, train_loader, val_loader, criterion, opt, n_epochs, scheduler):
    val = []
    device = torch.device("cuda")
    print("Starting Training Loop...")
    sys.stdout.flush()
    for epoch in range(n_epochs):
        model.train()
        n_iters = 0
        train_loss = []
        acc_train = []
        
        for batch in train_loader:
            model.zero_grad()
            image, labels = batch
            image, labels = image.to(device), labels.to(device)
            outputs = model(image)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            acc = compute_accuracy(outputs,labels)
            train_loss.append(loss.item())
            acc_train.append(acc.item())
            n_iters += 1
        loss_train= np.mean(train_loss)
        train_acc  = round(np.mean(acc_train),3)
        
        model.eval()
        n_iters = 0
        loss_val = []
        acc_val = []
        for batch in val_loader:
            image, label = batch
            image, label = image.to(device), label.to(device)
            pred = model(image)
            val_loss = criterion(pred, label)

            loss_val.append(val_loss.item())
            acc = compute_accuracy(pred, label)
            acc_val.append(acc.item())
            
            n_iters += 1
        val_acc= round(np.mean(acc_val),3)  
        val_loss = np.mean(loss_val)
        if scheduler is not None:
             scheduler.step()
        if epoch > 0:
            if val_acc > val[epoch-1]:
                torch.save(model.state_dict(), 'mobilenet' + args.version + '.pth')
        val.append(val_acc)
        print("Epoch {} | Training loss {}  | Testing loss  {} | Training Accuracy {}  | Testing Accuracy  {}".format(epoch, loss_train, val_loss,train_acc, val_acc))


def eval(model, val_loader, save_img, folder):
    val_accuracy_batch = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    n = 0
    for X_batch, y_batch in tqdm(val_loader):
            X_batch_gpu, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch_gpu)

            accuracy = compute_accuracy(logits, y_batch, device=device)
            val_accuracy_batch.append(accuracy.item())
            if save_img:
                labels = train_loader.dataset.class_to_idx
                classes = list(labels.keys())
                for i in range(len(X_batch_gpu)):
                    pred_num = torch.argmax(logits, dim=1)
                    pred_label = classes[list(labels.values()).index(pred_num[i])]
                    true_label = classes[list(labels.values()).index(y_batch[i])] 
                    img = X_batch_gpu[i] / 2 + 0.5     # unnormalize
                    npimg = img.to('cpu').numpy().transpose(1,2,0)
                    
                    fig = plt.figure(figsize=(1,1))
                    fig.figimage(npimg, xo = 0, yo = 0, origin='upper',resize=True , norm=True )
                    if true_label == pred_label:
                        fig.suptitle( pred_label, color="green", fontsize="x-small")
                    else:
                        fig.suptitle( pred_label, color="red", fontsize="x-small")
               
                   
#                     plt.imsave(os.path.join('./{}/'.format(folder) + 'img{}_{}_{}.png'.format(n,true_label,pred_label)),npimg) 
                    plt.savefig(os.path.join('./{}/'.format(folder) + 'img{}_{}_{}.png'.format(n,true_label,pred_label)))
                    plt.close(fig)
                    n+=1
    val_accuracy_overall = np.mean(val_accuracy_batch) * 100     
    return val_accuracy_overall

if __name__ == '__main__':
    if args.mode == 'train':
        train(model, train_loader, val_loader, criterion, opt, args.max_epoch, scheduler)
    elif args.mode == 'test':
        val_accuracy = eval(model, val_loader, args.save_img, test_save_dir)
        print("Validation accuracy: %.2f%%" % val_accuracy)
        
