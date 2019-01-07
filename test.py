import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw


print('Loading model..')
net = RetinaNet(num_classes=1)
net=torch.load('./checkpoint/ckpt50.pkl')
#net.load_state_dict(state['net'])
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

print('Loading image..')
img = Image.open('/home/peng/Desktop/final_papers/crop512_upload/easy/test_512/G260_13OCT2016_0246_47.jpg')
w = h = 600
img = img.resize((w,h))

print('Predicting..')
x = transform(img)
x = x.unsqueeze(0)
x = Variable(x, volatile=True)
loc_preds, cls_preds = net(x)

loc_preds, cls_preds= loc_preds.cpu(), cls_preds.cpu()
print('Decoding..')
encoder = DataEncoder()


#loc_preds, cls_preds, input_size
#CLS=0.3,NMS=0.15
boxes, labels, score = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(),(w,h),0.3,0.15)



# =============================================================================
# ####make prediction for all images
# =============================================================================
boxes=boxes.numpy()
score=score.numpy()
scale=600.0/512
all_blobs=[]
for idx in range(boxes.shape[0]):
    x1,y1,x2,y2=boxes[idx]
    x=((x1+x2)/2)/scale
    y=((y1+y2)/2)/scale
    px=int(x)
    py=int(y)
    all_blobs.append([px,py,score[idx]])

# =============================================================================
# ####mask prediction 
# =============================================================================


import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

import glob
from skimage import io
import skimage.feature
def IoU(box1, box2):

    [xmin1, ymin1, xmax1, ymax1] = box1
    [xmin2, ymin2, xmax2, ymax2] = box2
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    xmin_inter = max(xmin1, xmin2)
    xmax_inter = min(xmax1, xmax2)
    ymin_inter = max(ymin1, ymin2)
    ymax_inter = min(ymax1, ymax2)
    if xmin_inter > xmax_inter or ymin_inter > ymax_inter:
        return 0
    area_inter = (xmax_inter - xmin_inter) * (ymax_inter - ymin_inter)
    
    return float(area_inter) / (area1 + area2 - area_inter)

def evaluation_blob(blobs, pred_blobs, iou_ratio):      
    if len(blobs)==0 and len(pred_blobs)!=0:
        red=0
        fp=len(pred_blobs)
        tp=0
        fn=0
    if len(blobs)!=0 and len(pred_blobs)==0:
        red=0
        fn=len(blobs)
        tp=0
        fp=0
    if len(blobs)==0 and len(pred_blobs)==0:
        red=0
        fp=0
        tp=0
        fn=0  
        
    if len(blobs)!=0 and len(pred_blobs)!=0: 
        fp=0
        tp=0
        fn=0
        red=0
        nneg = lambda x: max(0, x) 
        thres=10
        for truth in blobs:
            ty=int(truth[0])
            tx=int(truth[1])
            box1=[nneg(tx - thres),nneg(ty - thres),nneg(tx + thres),nneg(ty + thres)]
            all_dis=[]
            all_score=[]
            pred_blobs=np.array(pred_blobs)           
            logic1=(pred_blobs[:,0]>=nneg(ty-100)) & (pred_blobs[:,0]<=nneg(ty+100)) 
            logic2=(pred_blobs[:,1]>=nneg(tx-100)) & (pred_blobs[:,1]<=nneg(tx+100)) 
            small_pred=pred_blobs[logic1 & logic2]
            for pred in small_pred:
                py= int(pred[0])
                px= int(pred[1])
                score=pred[2]
                
                box2=[nneg(px - thres),nneg(py - thres),nneg(px + thres),nneg(py + thres)]
                #dis=np.sqrt(np.power((px-tx),2)+np.power((py-ty),2))
                #need to change to IoU
                iou=IoU(box1,box2)
                all_score.append(score)
                all_dis.append(iou)
            #print(all_dis)
            if sum(np.array(all_dis)>iou_ratio) !=0:
                #for two ture with one pred problem
                min_dis=max(all_dis) 
                min_ind=all_dis.index(min_dis)  
                
                small_pred[min_ind]=[-100,-100,-100]
                tp+=1
                #red+=sum(np.array(all_dis)>iou_ratio)-1
            
            if sum(np.array(all_dis)>iou_ratio) ==0:
                fn+=1
            pred_blobs[logic1 & logic2]=small_pred
        fp=len(pred_blobs)-tp
        #print(tp,fn,fp)
    
    return fp,tp,fn,red

#pred_blobs=box
def nms(pred_blobs,thres):    
    all_light=[]
    
    for p1 in pred_blobs:
        y1,x1,score1=p1
        y1=int(y1)
        x1=int(x1)
        all_light.append(score1)
    nneg = lambda x: max(0, x)
    r=10
    for idx1 in range(len(pred_blobs)): 
        y1,x1,score1=pred_blobs[idx1]
        y1=int(y1)
        x1=int(x1)
        score1=all_light[idx1]
        #box1=[nneg(x1 - r),nneg(y1 - r),nneg(x1 + r),nneg(y1 + r)]
        for idx2 in range(len(pred_blobs)):
            if idx1==idx2:
                continue            
            y2,x2,score2=pred_blobs[idx2]
            y2=int(y2)
            x2=int(x2)
            score2=all_light[idx2]
            
            distance=np.sqrt(np.power((x2-x1),2)+np.power((y2-y1),2))
            if distance<=thres*np.sqrt(2):
                if score1>=score2:
                    all_light[idx2]=-1.0
            '''
            box2=[nneg(x2 - r),nneg(y2 - r),nneg(x2 + r),nneg(y2 + r)]
            iou=IoU(box1,box2)
            if iou>=thres:
                if score1>=score2:
                    all_light[idx2]=-1.0
            '''
    ind=np.array(all_light)>0
    output_blobs=pred_blobs[ind]
    return output_blobs
    
    
    
image_root='/home/peng/Desktop/final_papers/crop512_upload/easy/'
input_path=image_root+'val_512/*.jpg'
image_list=sorted(glob.glob(input_path))
test_gt=[]
test_pred=[]
for idx in range(len(image_list)):
    print(idx)
    output=io.imread(image_list[idx])
    img_name=image_list[idx].split('/')[-1]
    #output=io.imread(os.path.join(image_list[idx]))
    target=io.imread(os.path.join(image_root,'val_512_dot',img_name))

    image_3 = cv2.absdiff(target,output)
    mask_1 = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    mask_1[mask_1 < 20] = 0
    mask_1[mask_1 > 0] = 255
    
    mask_2 = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    mask_2[mask_2 < 20] = 0
    mask_2[mask_2 > 0] = 255
    
    image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
    image_5 = cv2.bitwise_or(image_4, image_4, mask=mask_2) 
    
    # convert to grayscale to be accepted by skimage.feature.blob_log
    image_6 = cv2.cvtColor(image_5, cv2.COLOR_RGB2GRAY)
    # detect blobs
    blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=10, num_sigma=1, threshold=0.05)
    #get gt cood
    all_gt=[]
    for i in range(len(blobs)):
        y1,x1,s1=blobs[i]
        y=int(y1)
        x=int(x1)
        true_box=[y,x]
        all_gt.append(true_box)
    test_gt.append(all_gt)
    
    img = Image.open(image_list[idx])
    w = h = 600
    img = img.resize((w,h))
    
    #print('Predicting..')
    x = transform(img)
    x = x.unsqueeze(0)
    x = Variable(x, volatile=True)
    loc_preds, cls_preds = net(x)
    
    #loc_preds, cls_preds= loc_preds.cpu(), cls_preds.cpu()
    #print('Decoding..')
    encoder = DataEncoder()
    
    
    #loc_preds, cls_preds, input_size
    #CLS=0.3,NMS=0.15
    boxes, labels, score = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(),(w,h),0.0001,0.15)
    #draw = ImageDraw.Draw(img)
    '''
    for box in boxes:
        draw.rectangle(list(box), outline='red')
    img.show()
    '''

    all_blobs=[]

    boxes=boxes.cpu().numpy()
    score=score.cpu().numpy()
    scale=600.0/512
    if score !=[]:

        x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3] # different order compared with mask rcnn and unet
        x=((x1+x2)/2)/scale
        y=((y1+y2)/2)/scale
        new_y=y
        new_x=x
        length=len(new_y)
        new_y=new_y.reshape(length,1)
        new_x=new_x.reshape(length,1)
        score=score.reshape(length,1)
        box=np.concatenate((new_y, new_x,score), axis=1)
        if len(score)>1:
           box=nms(box,10)     ####most of time comsuming part!!!!!
        all_blobs.append(box)
    test_pred.append(all_blobs)
    
print('calculating f1 score for %d test' %len(image_list))
thres_list=np.arange(0, 1.0, 0.05)


# on easy one, 0.3 -> 0.8729 adam

for thres in thres_list:
    all_tp=[]
    all_fn=[]
    all_fp=[]
    iou_ratio=0
    all_pred=[]
    #thres=0.95
    all_recall=[]
    all_precision=[]
    all_true=[]
    print thres
    for idx in range(len(image_list)):   
        #get gt cood
        all_gt=test_gt[idx]
    
        
        pred=test_pred[idx]
        all_blobs=[]           
        for i in range(len(pred)):
            for m in range(len(pred[i])):
               y1,x1,score=pred[i][m]
               if score>thres:
                   box=[y1,x1,score]
                   all_blobs.append(box)
                           
    
                    
        pred_len=len(all_blobs)  
        #print(pred_len)
        fp,tp,fn,red=evaluation_blob(all_gt,all_blobs,iou_ratio=iou_ratio)
        assert(tp+fn==len(all_gt))
        assert(tp+fp+red==pred_len)
        all_tp.append(tp)
        all_fn.append(fn)
        all_fp.append(fp)   
        all_pred.append(pred_len)
        if tp+fn==0:
            recall=0
        else:
            recall=float(tp)/(tp+fn)
            
        if tp+fp==0:
            precision=0
        else:
            precision=tp/(tp+fp)
        
        all_recall.append(recall)
        all_precision.append(precision)
        all_true.append(len(all_gt))
        
    #print(all_tp)
    #print(all_fn)
    #print(all_fp)
    
    recall_result=float(np.sum(all_tp))/(np.sum(all_tp)+np.sum(all_fn))
    precision_result=float(np.sum(all_tp))/(np.sum(all_tp)+np.sum(all_fp))
    
    
    f1=(2*precision_result*recall_result)/(recall_result+precision_result)
    
    
    print('full test f1: %.4f' % f1)    

img = img.resize((w,h))

draw = ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(list(box), outline='red')
img.show()
