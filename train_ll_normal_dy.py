from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss_ll_normal import FocalLoss
from retinanet import RetinaNet
from datagen import ListDataset

from torch.autograd import Variable
import copy
from tqdm import tqdm

#==============================================================================
# parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
# parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# args = parser.parse_args()
#==============================================================================

lr=1e-4
resume=False
fix='pre_train'

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

trainset = ListDataset(list_file='/group/proteindl/ps793/Dota/train_retina_dota_nd.txt' , train=True, transform=transform, input_size=1024)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)

testset = ListDataset(list_file='/group/proteindl/ps793/Dota/val_retina_dota_nd.txt' , train=False, transform=transform, input_size=1024)
testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, num_workers=8, collate_fn=testset.collate_fn)

# Model
net = RetinaNet(num_classes=15)
#net=torch.load('./model/dota_15c_9ma.pth')
#net.load_state_dict(torch.load('./model/dota_15c_9ma.pth'))
net=torch.load('./checkpoint/adam_e4_iou45_pre_fpn50_full_b2_dota_50_nd_9ma_s20_105_5block_epoch1_25.pkl')
net=net.module
net.fpn.eval()
#net=net.module
if resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()

if fix=='head':
    for param in net.module.fpn.parameters():
        param.requires_grad = False
if fix=='pre_train':
    for param in net.module.fpn.conv1.parameters():
        param.requires_grad = False
    for param in net.module.fpn.bn1.parameters():
        param.requires_grad = False
    for param in net.module.fpn.layer1.parameters():
        param.requires_grad = False
    for param in net.module.fpn.layer2.parameters():
        param.requires_grad = False
    for param in net.module.fpn.layer3.parameters():
        param.requires_grad = False
    for param in net.module.fpn.layer4.parameters():
        param.requires_grad = False
if fix=='full':
    for param in net.parameters():
        param.requires_grad = True
        
        
from itertools import ifilter
op_parameters = ifilter(lambda p: p.requires_grad, net.parameters())
  
'''    
count=0
for param in net.parameters():
    count+=1
    print(param.requires_grad)
    print(count) 
'''
criterion = FocalLoss(num_classes=15)
optimizer = optim.Adam(op_parameters, lr=lr, betas=(0.9, 0.99))
#later add scheduler into optimizer
from torch.optim import lr_scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#optimizer = optim.SGD(op_parameters, lr=lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    scheduler.step()
    net.train()
    net.module.freeze_bn()
    '''
    for param in fpn.parameters():
        param.requires_grad = False
    '''
    train_loss = 0
    count=0
    t = tqdm(trainloader)
    for batch_idx, (inputs, loc_targets, cls_targets, flip) in enumerate(t):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()
        if num_pos==0:
            count+=1
            #print(batch_idx)
            continue
        
        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        #loss,loc_loss,cls_loss,iw = criterion(inputs, flip, loc_preds, loc_targets, cls_preds, cls_targets)
        loss,loc_loss,cls_loss,iw= criterion(inputs, net.module.fpn,loc_preds, loc_targets, cls_preds, cls_targets)
        
	loss.backward()
        optimizer.step()
	#print(iw)
        train_loss += loss.data[0]
        #print('train_loss: %.3f | avg_loss: %.3f' % (loss.data[0], train_loss/(batch_idx-count+1)))
    	#t.set_description('loc_loss: %.3f|cls_loss: %.3f|iw: %.2f | train_loss: %.3f | avg_loss: %.3f, %d' % (loc_loss, cls_loss, iw, loss.data[0], train_loss/(batch_idx+1),flip[0]))
        t.set_description('loc_loss: %.3f|cls_loss: %.3f|train_loss: %.3f | avg_loss: %.3f, %d' % (loc_loss, cls_loss, loss.data[0], train_loss/(batch_idx+1),flip[0]))
        
    return train_loss/ len(trainloader)
# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    count=0
#t.set_description('loc_loss: %.3f|cls_loss: %.3f|iw: %.2f | train_loss: %.3f | avg_loss: %.3f, %d' % (loc_loss, cls_loss, iw, loss.data[0], train_loss/(batch_idx+1),flip[0]))
    t = tqdm(testloader)    
    for batch_idx, (inputs, loc_targets, cls_targets, flip) in enumerate(t):
        #need to check 10_G272_15Nov2016_0019.JPG, batch_idx=189
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()
        if num_pos==0:
            count+=1
            #print(batch_idx)
            continue
        
        loc_preds, cls_preds = net(inputs)
        #loss,loc_loss,cls_loss,iw = criterion(inputs, flip, loc_preds, loc_targets, cls_preds, cls_targets)
        loss,loc_loss,cls_loss,iw= criterion(inputs, net.module.fpn,loc_preds, loc_targets, cls_preds, cls_targets)
        
	test_loss += loss.data[0]
        #print('test_loss: %.3f | avg_loss: %.3f' % (loss.data[0], test_loss/(batch_idx-count+1)))
        t.set_description('loc_loss: %.3f|cls_loss: %.3f| test_loss: %.3f | avg_loss: %.3f, %d' % (loc_loss, cls_loss, loss.data[0], test_loss/(batch_idx+1),flip[0]))
	#print('test_loss: %.3f | avg_loss: %.3f' % (loss.data[0], test_loss/(batch_idx-count+1)))
    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        best_model_wts = copy.deepcopy(net.state_dict())
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        net.load_state_dict(best_model_wts)
        torch.save(net, './checkpoint/flipflop_adam_e4_fpn50_pretrain_b2_50_nd_9ma_s10_105_5gbs_ll_normal_dy.pkl' )
        best_loss = test_loss
    
    return test_loss

import csv
train_loss_list=[]
test_loss_list=[]
for epoch in range(start_epoch, start_epoch+20):
    loss=train(epoch)
    train_loss_list.append(loss)
    loss=test(epoch)
    test_loss_list.append(loss)
    with open('/group/proteindl/ps793/Dota/retina/flipflop_train_loss_e4_pretrain_adam_nd_9ma_b2_fpn50_s10_105_5gbs_ll_normal_dy.csv', 'wb') as myfile:
    	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    	wr.writerow(train_loss_list)

    with open('/group/proteindl/ps793/Dota/retina/flipflop_test_loss_e4_pretrain_adam_nd_9ma_b2_fpn50_s10_105_5gbs_ll_normal_dy.csv', 'wb') as myfile:
    	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    	wr.writerow(test_loss_list)
    if epoch % 5 ==0:
       torch.save(net, './checkpoint/flipflop_adam_e5_fpn50_pretrain_b2_dota_50_nd_9ma_s10_105_5gbs_ll_epoch_normal_dy_%d.pkl' % epoch )

torch.save(net, './checkpoint/flipflop_adam_e5_fpn50_pretrain_b2_dota_50_nd_9ma_s10_105_5gbs_ll_final_normal_dy.pkl' )   

