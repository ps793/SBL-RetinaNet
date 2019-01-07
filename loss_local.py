from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools import one_hot_embedding
from torch.autograd import Variable
from torchvision import  models
import numpy as np

model_res=models.resnet50(pretrained=True).cuda()

class FocalLoss(nn.Module):
    def __init__(self, num_classes=20):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
	self.features= nn.Sequential(*list(model_res.children())[0:6])
    def focal_loss(self, x, y):
        '''Focal loss.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)  # [N,21]
        t = t[:,1:]  # exclude background
        t = Variable(t).cuda()  # [N,20]

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    def focal_loss_alt(self, x, y,iw):
        '''Focal loss alternative.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
	n=iw.size()[0]

        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)
        t = t[:,1:]
        t = Variable(t).cuda()

        xt = x*(2*t-1)  # xt = x if t > 0 else -x
        pt = (2*xt+1).sigmoid()
        pt= pt.clamp(min=1e-15)
        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2

	m=loss.size()[0]
        i=0
        w_loss=0
        while i<n:
            w_loss+=iw[i]*loss[i*m/n:(i+1)*m/n].sum()
            i+=1
        return w_loss

    def forward(self,x,net, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()
	# ==============================================================
        # weight
        # ==============================================================
       	''' 
	#####flip info
	n=flip.size()[0]
        iw=[]
        i=0
        while i<n:
            if flip[i]==1:
                y = x[i].data.cpu().numpy()
                y = np.flip(y,2)#left to right
                y = torch.from_numpy(y.copy())
                y = y.unsqueeze(0)
                y = Variable(y)
                z = self.features(y.cuda())
                r = z.size(3)       
                z = F.avg_pool2d(z, r)
                z = z.view(z.size(0), -1)
                z = F.relu(z)
                iw.append(torch.mean(z,1) )
            else:
                #train min 0.311, max 0.465  
                img = x[i].unsqueeze(0)
                z = self.features(img.cuda())
                r = z.size(3)       
                z = F.avg_pool2d(z, r)
                z = z.view(z.size(0), -1)
                z = F.relu(z)
                iw.append(torch.mean(z,1) )
                
            i+=1  
        iw=torch.cat(iw, dim=0) 
	
	#########no flip info
	z = self.features(x.cuda())
        r = z.size(3)      
        z = F.avg_pool2d(z, r)
        z = z.view(z.size(0), -1)
        z = F.relu(z)
	iw = torch.mean(z,1)
	
	#iw=(iw-0.311)*(1.0-0.5)/(0.465-0.311)+0.5#median 0.4497
	#iw=(iw-0.017)*(1.0-0.5)/(0.042-0.017)+0.5
	iw=(iw-0.129)*(1.0-0.5)/(0.180-0.129)+0.5#block 5
        '''
	z=self.features(x.cuda())
        g_z=torch.mean(z,1)
        g_z=F.relu(g_z)
        
        n=net(x.cuda())
        l=n[0]#index of feature map  p3->6,p4->7,p5->8 
        l=F.relu(l)
        l_z=torch.mean(l,1)
        #l_z=F.relu(l_z)
        
        fea_dot=g_z*l_z
        
        r=fea_dot.size(2)
        z=F.avg_pool2d(fea_dot,r)
        z=z.view(z.size(0),-1)
        z=F.relu(z)
        iw=z
        iw = (iw-0.0105)*(1-0.5)/(0.036-0.0105)+0.5 #c3
	#print(iw)
        ##0.0011994859, 0.0047983024 c4
        ##0.0075986516, 0.025505692 c5
	################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg],iw) #cls_targets[pos_neg].unsqueeze(1)

        #print('loc_loss: %.3f|cls_loss: %.3f|iw: %.2f' % (loc_loss.data[0]/num_pos, cls_loss.data[0]/num_pos, iw.data[0]), end=' | ')
        loss = (loc_loss+cls_loss)/num_pos
        return loss,loc_loss.data[0]/num_pos, cls_loss.data[0]/num_pos, iw.data[0]
