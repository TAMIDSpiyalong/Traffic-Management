import argparse
import os
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import re
import csv
import utm
import json
import torch
print(torch.cuda.get_device_name(),torch.cuda.get_device_properties(0), torch.__version__)
import warnings
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)
from shapely.geometry import Point as P
from shapely.geometry.polygon import LinearRing, Polygon
from collections import deque
from utils.datasets import *
from utils.utils import *
from deep_sort import preprocessing,nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from intersection import *
from . import Mapping,NSG_threshold_selection,non_max_suppression_
import seaborn as sns; sns.set_theme()


class VideoProcesser:
    def __init__(self):
    # Project parameters
        self.running_with_a_monitor=True
        self.classes_interested = {0:"Pedestrain",2:"Car",5:"Bus", 7:"Truck"}
        self.colors = [ 
        (0,0,255),
        (102,204,0), 
        (255,0,255),
        (0,128,255)
        ]
    # Deep SORT parameters
        self.max_cosine_distance = 0.1
        self.nn_budget = None
        self.nms_max_overlap = 1
        self.model_filename = 'C:/Users/piyalong/Traffic Management/yolov5-master/model_data/veri.pb'
    #     YOLO v5 parameters
        self.weights = "C:/Users/piyalong/Traffic Management/yolov5-master/weights/yolov5x.pt"
        self.img_size=1280
        self.conf_thres=0.2
        self.iou_thres = 0.5
        self.show_study_zone=False
    def tracking (self,video_path):
        classes=[i for i in self.classes_interested.keys()]
        video_capture = cv2.VideoCapture(video_path)
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        buffer = fps
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        half = device.type != 'cpu'  # half precision only supported on CUDA
        model = torch.load(self.weights, map_location=device)['model'].float()  # load to FP32
        if half:
            model.half()  # to FP16
        model.to(device).eval()
        names = model.names if hasattr(model, 'names') else model.modules.names
        img = torch.zeros((1, 3, self.img_size, self.img_size), device=device)  # init img
    ###########################################################################################################
       
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        out = cv2.VideoWriter(video_path[:-4]+'_out.avi', fourcc, fps, (w+h, h))
    ##############################################################################################################################
        trajectory={i:{} for i in self.classes_interested.values()}
        traces_to_save={i:{} for i in self.classes_interested.values()}
        trajectory_wait={i:{} for i in self.classes_interested.values()}
        encoder = gdet.create_box_encoder(self.model_filename,batch_size=8)
        # metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = {class_name:Tracker(nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)) for class_name in self.classes_interested.values()}
        t=[]

        frame_index = 0
        while True:
            print("_{}_".format(frame_index),end="")
            t0 = time.time()
            ret, frame = video_capture.read()  # frame shape 640*480*3
            if ret != True:
                break
            conf_thres = NSG_threshold_selection.determine_threshod(frame,'Two_cluster')
            imgtest = letterbox(frame, new_shape=self.img_size)[0]
            imgtest = imgtest[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            imgtest = np.ascontiguousarray(imgtest)
            imgtest  = torch.from_numpy(imgtest).to(device)
            imgtest = imgtest.half() if half else imgtest.float()  # uint8 to fp16/32
            if imgtest.ndimension() == 3:
                imgtest = imgtest.unsqueeze(0)
            imgtest /= 255.0  # 0 - 255 to 0.0 - 1.0
            #     imgtest.shape
            pred = model(imgtest)[0]
            pred = non_max_suppression_.non_max_suppression_(pred, conf_thres, self.iou_thres, fast=True, classes=classes, agnostic=False)
            results = []
            object_tracked={i:{'dets':[],'conf':[]} for i in self.classes_interested.values()}
###################################################################################################################################
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    # Rescale boxes from img_size to frame size
                    det[:, :4] = scale_coords(imgtest.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in det:
                        if int(cls) not in self.classes_interested:
                            continue
                        if int(cls)==7:
                            cls=2
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, frame, label=label, color=self.colors[int(cls)], line_thickness=3)
                        x1,y1,x2,y2 = [ i.item () for i in xyxy]
                        object_tracked[self.classes_interested[int(cls)]]['dets'].append([x1, y1, x2-x1, y2-y1])
                        results.append([int(cls),x1, y1, x2, y2,float(conf)])
                        object_tracked[self.classes_interested[int(cls)]]['conf'].append(float(conf))
            traces_to_save.update({frame_index : results})

            for class_name, queue in object_tracked.items():
                dets= queue['dets']
                confidence =queue['conf']
                # Process detections
                features = encoder(frame,dets)
                detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in zip(dets, confidence, features)]
                # Call the tracker
                tracker[class_name].predict()
                tracker[class_name].update(detections)
                for track in tracker[class_name].tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue 
                    bbox = track.to_tlbr()
                    x_bc,y_bc = int((bbox[0]+bbox[2])/2),int(bbox[3])
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[3])), (int(bbox[2]), int(bbox[3]+20)), (255,0,0), -1)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
                    cv2.putText(frame, '#'+str(track.track_id),(int(bbox[0]), int(bbox[3]+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
                    obj_id=int(track.track_id)
                    if obj_id not in trajectory[class_name]:
                        trajectory[class_name].update({obj_id:deque(maxlen=buffer)})
                        traces_to_save[class_name].update({obj_id:[]})
                        trajectory_wait[class_name].update({obj_id:0})
                    trajectory_wait[class_name][obj_id]+=1
                    trajectory[class_name][obj_id].appendleft((x_bc,y_bc))
                    traces_to_save[class_name][obj_id].append((x_bc,y_bc,frame_index))
                trajectory_wait[class_name] = {k:v-1 for k,v in trajectory_wait[class_name].items()}
                for k,v in trajectory_wait[class_name].items():
                    if v<0 and k in trajectory[class_name].keys():
                        del trajectory[class_name][k]
######################################################################################################################################################
                for key, pts in trajectory[class_name].items():
                    line_color = (0, 0, 255)
                    for i in np.arange(1, len(pts)):
                        if pts[i - 1] is None or pts[i] is None:
                            continue
                        else: 
                            thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
                            cv2.line(frame, pts[i - 1], pts[i], line_color, thickness)
######################################################################################################################################                    
            t.append(time.time() - t0)
            cv2.putText(frame, 'Frame: '+str(frame_index), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.putText(frame, 'FPS: '+str(round(1/(np.mean(t)),2)), (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            if self.show_study_zone == True:
                for k,v in self.pixelzones.items():
                    v=np.array(v).astype(np.int32)        
                    x,y=zip(*v)
                    center=(max(x)+min(x))//2, (max(y)+min(y))//2
                    cv2.polylines(frame,[v],True,(0,255,255))    
                    cv2.putText(frame, str(k),(center),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_index = frame_index + 1
        out.release()
        cv2.destroyAllWindows()
        np.save(video_path[:-4]+"_traces.npy",traces_to_save)
        
        
    def zoning (self,studyzones):
        self.show_study_zone =True
        self.pixelzones = {}
        with open(studyzones, 'r') as json_file:
            zones = json.load(json_file)
            for each in (zones['shapes']):
                if each['label']=='reference':
                    continue
                else:
                    self.pixelzones.update({each['label']:each['points']})
        
#         return frame
    
    def projection(self,studyzones,reference):
        real_reference = np.load(reference)
#         lat_long_reference = reference
#         real_reference = [utm.from_latlon(i,j)[:2] for i,j in lat_long_reference]
        pixel_reference = []
        pixelzones = {}
        with open(studyzones, 'r') as json_file:
            zones = json.load(json_file)
            for each in (zones['shapes']):
                if each['label']=='reference':
                    pixel_reference=each['points']
                else:
                    pixelzones.update({each['label']:each['points']})
        maps = Mapping(real_reference,pixel_reference, pixelzones)
        fig = maps.show_study_zones()
        boundary=[ ]
        for corner  in maps.checkzones.values():
            x_min = np.min(np.array(corner)[:,0])
            x_max = np.max(np.array(corner)[:,0])
            y_min = np.min(np.array(corner)[:,1])
            y_max = np.max(np.array(corner)[:,1])
            boundary.append([x_min,x_max,y_min,y_max])
        boundary = np.array(boundary)
        margin = 200
        plt.xlim([min(boundary[:,0])-margin, max(boundary[:,1])+margin])
        plt.ylim([min(boundary[:,2])-margin,max(boundary[:,3])+margin])
        with open(video_path[:-4]+"_raw_points.csv", "w",newline='') as a_file:
            writera = csv.writer(a_file)
            rawpoints = maps.project_trajectory(trajectory[class_name])
#                 waypoints = maps.Waypoint(trajectory[class_name])
            for k,v in rawpoints.items():
                if v is not None and len(v)>1:
                    i = [float(x[0]) for x in v]
                    j = [float(x[1]) for x in v]
                    plt.plot(i, j,"r")
                    x1,y1,x2,y2 = v[0][0],v[0][1],v[-1][0],v[-1][1]
                    dist = math.hypot(x2 - x1, y2 - y1)
                    speed = dist/(buffer/fps)*2.23694
                    plt.text(v[0][0],v[0][1],'#'+str(k)+"("+str(round(speed))+" MPH)",fontsize=15)
                    plt.plot(v[0][0],v[0][1],'go',markersize=10)
                    writera.writerow([round(k),class_name,frame_index,v[0][0],v[0][1]])
                            
                fig.canvas.draw()
                the_map = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                the_map  = the_map.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                the_map = cv2.cvtColor(the_map,cv2.COLOR_RGB2BGR)
                the_map_show = cv2.resize(the_map,(h,h))
                plt.close()
                vis = np.concatenate((frame, the_map_show), axis=1)
                out.write(vis)