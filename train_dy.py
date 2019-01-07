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

from loss_dy import FocalLoss
from retinanet import RetinaNet
from datagen import ListDataset

from torch.autograd import Variable
import copy
from tqdm import tqdm
import numpy as np
#==============================================================================
# parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
# parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# args = parser.parse_args()
#==============================================================================

lr=1e-4
resume=False
fix='full'

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

trainvalloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=8, collate_fn=trainset.collate_fn)


testset = ListDataset(list_file='/group/proteindl/ps793/Dota/val_retina_dota_nd.txt' , train=False, transform=transform, input_size=1024)
testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, num_workers=8, collate_fn=testset.collate_fn)

# Model
net = RetinaNet(num_classes=15)
net.load_state_dict(torch.load('./model/dota_15c_9ma.pth'))
if resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()

if fix=='head':
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
else:
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
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
#optimizer = optim.SGD(op_parameters, lr=lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    ###preparation for normalization
    print('\ntraining prepare')
    pre_net=nn.Sequential(*list(net.module.fpn.children())[:6])
    for param in pre_net.parameters():
        param.requires_grad = False
    f1=nn.Sequential(*list(pre_net.children())[:3])
    f2=nn.Sequential(*list(pre_net.children())[3:4])
    f3=nn.Sequential(*list(pre_net.children())[4:5])
    f4=nn.Sequential(*list(pre_net.children())[5:6])
    saliency_map=[f1,f2,f3,f4]
    t = tqdm(trainvalloader)
    feature_layer=[]
    for batch_idx, (inputs, loc_targets, cls_targets,flip) in enumerate(t):
        #if batch_idx>10:
	#	break 
        #need to check 10_G272_15Nov2016_0019.JPG, batch_idx=5
        inputs = Variable(inputs.cuda(), volatile=True)
        p1=f1(inputs)#5
        p2=f2(p1.cuda())#6
        p3=f3(p2.cuda())#7
        p4=f4(p3.cuda())#8
        
        result=[p1,p2,p3,p4]   
        #print(result.size())
        x=result
        features_list=[]
        for p in x:
            r = p.size(3)       
            p = F.avg_pool2d(p, r)
            p = p.view(p.size(0), -1)
            p = F.relu(p)
            value=torch.mean(p,1).cpu().data.numpy()[0]
            features_list.append(value)
        feature_layer.append(features_list)
    saliency=np.array(feature_layer)
    salien_min=np.min(saliency,axis=0)
    salien_max=np.max(saliency,axis=0)
    salien_min=torch.from_numpy(salien_min.copy())
    salien_max=torch.from_numpy(salien_max.copy())
    
    del f1,f2,f3,f4,p
    
    
    idx=2###saliency idx change here for ind
    f=saliency_map[:idx] # f1
    del saliency_map
    
    print('\ntraining')
    scheduler.step()
    net.train()
    net.module.freeze_bn()
    train_loss = 0
    count=0
    t = tqdm(trainloader)
    for batch_idx, (inputs, loc_targets, cls_targets, flip) in enumerate(t):
        #if batch_idx>10:
        #        break
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
	##saliency weight
	x=inputs
	for i in range(len(f)):
	    x=f[i](x.cuda())#5
        r = x.size(3)
        z = F.avg_pool2d(x, r)
        z = z.view(z.size(0), -1)
        z = F.relu(z)
        iw = torch.mean(z,1)
        del x,z
        #w3=(feature_list[3]-0.311)*(1-0.5)/(0.465-0.311)+0.5
        #w2=(feature_list[2]-0.017)*(1-0.5)/(0.042-0.017)+0.5
        #w1=(feature_list[1]-0.058)*(1-0.5)/(0.113-0.058)+0.5
        iw=(iw-salien_min[idx])*(1-0.5)/(salien_max[idx]-salien_min[idx])+0.5

        loss,loc_loss,cls_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets,iw)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        #print('train_loss: %.3f | avg_loss: %.3f' % (loss.data[0], train_loss/(batch_idx-count+1)))
    	t.set_description('loc_loss: %.3f|cls_loss: %.3f|iw: %.2f | train_loss: %.3f | avg_loss: %.3f, %d' % (loc_loss, cls_loss, iw.data[0], loss.data[0], train_loss/(batch_idx+1),flip[0]))
        
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
        
	#if batch_idx>10:
        #  	 break#need to check 10_G272_15Nov2016_0019.JPG, batch_idx=189
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
	iw=torch.from_numpy(np.array([1]*inputs.size()[0])).cuda()
        
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()
        if num_pos==0:
            count+=1
            #print(batch_idx)
            continue
        
        loc_preds, cls_preds = net(inputs)
        loss,loc_loss,cls_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets,iw)
        test_loss += loss.data[0]
        #print('test_loss: %.3f | avg_loss: %.3f' % (loss.data[0], test_loss/(batch_idx-count+1)))
        t.set_description('loc_loss: %.3f|cls_loss: %.3f|iw: %.2f | test_loss: %.3f | avg_loss: %.3f, %d' % (loc_loss, cls_loss, iw[0], loss.data[0], test_loss/(batch_idx+1),flip[0]))
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
        torch.save(net, './checkpoint/adam_e4_iou45_pre_fpn50_full_b2_50_nd_9ma_s20_105_7block_dy.pkl' )
        best_loss = test_loss
    
    return test_loss
import csv
train_loss_list=[]
test_loss_list=[]
for epoch in range(start_epoch, start_epoch+50):
    loss=train(epoch)
    train_loss_list.append(loss)
    loss=test(epoch)
    test_loss_list.append(loss)
    if epoch % 5 ==0:
       torch.save(net, './checkpoint/adam_e4_iou45_pre_fpn50_full_b2_50_nd_9ma_s20_105_7block_dy_epoch_%d.pkl' % epoch )
    with open('/group/proteindl/ps793/Dota/retina/train_loss_e4_full_adam_nd_9ma_iou45_b2_fpn50_s20_105_7block_dy.csv', 'wb') as myfile:
    	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    	wr.writerow(train_loss_list)


    with open('/group/proteindl/ps793/Dota/retina/test_loss_e4_full_adam_nd_9ma_iou45_b2_fpn50_s20_105_7block_dy.csv', 'wb') as myfile:
    	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    	wr.writerow(test_loss_list)

torch.save(net, './checkpoint/adam_e4_iou45_pre_fpn50_full_b2_50_nd_9ma_s20_105_7block_dy_final.pkl' )   
