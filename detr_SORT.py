import argparse
import cv2
import numpy as np
import math
from PIL import Image
import requests
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/piyalong/multi-object-tracker/')

from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks

import ipywidgets as widgets
from IPython.display import display, clear_output

from support_functions import nms_python
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);




# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch. set_grad_enabled(False)

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]



model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).to(device)
model.eval();


# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b


# In[18]:


def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()



def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result



def predict(im):
    img = transform(im).unsqueeze(0).to(device)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    
    pil_img, prob, boxes=im, probas[keep], bboxes_scaled
    # print(prob)
    
    colors = COLORS * 100
    confs=[]
    bboxes=[]
    classes=[]
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):

        cl = p.argmax()
        if cl!=3:continue
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'

        confs.append(p[p.argmax()].detach().cpu().numpy())
        bboxes.append([xmin, ymin, xmax, ymax])
        classes.append(cl.detach().cpu().numpy())
    # print(torch.from_numpy(np.array(confs)),"\n",torch.from_numpy(np.array(bboxes)))
    # print(leb)
    if len(bboxes)!=0:
        tensorboxes = nms_python(np.array(bboxes),np.array(confs),0.5)
    else:tensorboxes=bboxes
    return [confs,tensorboxes,classes]



def get_trace(file_path):

    
    print(file_path)
    video_capture = cv2.VideoCapture(file_path)
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
   
    out = cv2.VideoWriter(file_path[:-4]+'_Tracker.avi', fourcc, fps, (width, height))

    tracker = IOUTracker(10,0.5)

    # tracker = CentroidTracker(max_lost=fps)

    data={}
    frame_number=0
    while True:
        frame_number+=1
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        img_raw = frame.copy()
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_raw = Image.fromarray(img_raw)

        # run inference
        detection_confidences ,bboxes_scaled , detection_class_ids = predict(img_raw)
        if len(bboxes_scaled)==0:continue
        detection_bboxes = [[x1,y1,x2-x1,y2-y1] for [x1,y1,x2,y2] in bboxes_scaled]
        tracks = tracker.update(np.array(detection_bboxes), detection_confidences, detection_class_ids)

        # draw the predictions
        # bboxes_scaled = predict(img_raw)
        # bboxes_scaled=detection_bboxes
        for c,bbox,cl in zip(detection_confidences ,bboxes_scaled , detection_class_ids):
            # print(bbox)
            x1,y1,x2,y2 = bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 3)     
            text = f'{CLASSES[cl]}: {c:0.2f}'
            cv2.putText(frame,text,(int(x1), int(y1+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
                        
        for t in tracks:
            frame_numeber=t[0]
            ID = t[1]
            xmin,ymin,w,h = t[2:6]
            xcentroid, ycentroid = int(xmin + 0.5*w), int(ymin + 0.5*h)
            if ID in data.keys():
                data[ID].append([xcentroid, ycentroid,frame_number])
            else:
                data.update({ID:[]})
                data[ID].append([xcentroid, ycentroid,frame_number])

        updated_image = draw_tracks(frame, tracks)
        # print()
        updated_image =  cv2.resize(updated_image, (width,height),interpolation=cv2.INTER_AREA)
        cv2.imshow("Out",updated_image)
        out.write(updated_image)    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    cv2.destroyAllWindows()
    np.save(file_path[:-4]+'.npy',data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('process video and get the car traces', add_help=False)
    parser.add_argument('--video_path', help='video file path',)
   
    args=parser.parse_args()

    get_trace(args.video_path)

