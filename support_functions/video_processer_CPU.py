import argparse
import os
import cv2
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import re
import csv
import utm
import time
import json
import torch
from collections import deque
from tools import generate_detections as gdet
import sys
sys.path.insert(0, "C:/Users/piyalong/yolov5-master/yolov5-master/")
import datetime
from . import mapping,NSG_threshold_selection,non_max_suppression_
from utils.datasets import *
from utils.utils import *
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression
from deep_sort import preprocessing,nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from shapely.geometry import Point as P
from shapely.geometry.polygon import LinearRing, Polygon
import pandas as pd
from  support_functions.mapping import *
import warnings
warnings.filterwarnings('ignore')


class VideoProcesser:
    def __init__(self):
        self.display=False
        self.classes_interested = {0:"Pedestrain",2:"Car"}
        self.skipframe = 1
    # Deep SORT parameters
        self.max_cosine_distance = 1
        self.nn_budget = 30
        self.nms_max_overlap = 0.5
        self.model_filename = 'C:/Users/piyalong/yolov5-master/yolov5-master/mars-small128.pb'
    #     YOLO v5 parameters
        self.weights = "C:/Users/piyalong/yolov5-master/yolov5-master/yolov5s6.pt"
        self.img_size=1280
        self.conf_thres=0.2
        self.iou_thres = 0.5
        self.traces_to_save = {}
        self.pixelzones={}
        self.pixel_reference=[]
        self.interval='1min'
        
    def read_pixel_file(self,pixelfilename):
        with open(pixelfilename, 'r') as json_file:
            zones = json.load(json_file)
            for each in (zones['shapes']):
                if each['label']=='reference':
                    self.pixel_reference=each['points']
                else:
                    self.pixelzones.update({each['label']:each['points']})
        self.total_leaving = {i:{} for i in self.pixelzones.keys()}
        self.total_entering = {i:{}  for i in self.pixelzones.keys()}        
    def show_study_zone (self,frame):
        for k,v in self.pixelzones.items():
            v=np.array(v).astype(np.int32)        
            x,y=zip(*v)
            center=(max(x)+min(x))//2, (max(y)+min(y))//2
            cv2.polylines(frame,[v],True,(0,255,255),5)    
            cv2.putText(frame, str(k),(center),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 5)
        return frame        
        
    def tracking (self,video_path):
        cmap = plt.get_cmap('tab20b')

        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        classes=[i for i in self.classes_interested.keys()]
        video_capture = cv2.VideoCapture(video_path)
        buffer=fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        device = torch.device("cpu")
        model = DetectMultiBackend(self.weights, device=device, dnn=False)
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        model.to(device).eval()
        names = model.names if hasattr(model, 'names') else model.modules.names
        img = torch.zeros((1, 3, self.img_size, self.img_size), device=device)  # init img
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        w ,h = [1080,1080]
        out = cv2.VideoWriter(video_path[:-4]+'_out.avi', fourcc, fps, (w+h, h))
        trajectory={i:{} for i in self.classes_interested.values()}
        traces_to_save={i:{} for i in self.classes_interested.values()}
        trajectory_wait={i:{} for i in self.classes_interested.values()}
        encoder = gdet.create_box_encoder(self.model_filename)
        tracker = {class_name:Tracker(nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)) for class_name in self.classes_interested.values()}
        t0 = time.time()
        frame_index = 0
        while(video_capture.isOpened()):
            frame_index = frame_index + 1
            if frame_index%self.skipframe!=0 :continue
            print("Frame_{},".format(frame_index),end="")
            ret, frame = video_capture.read()  
            if ret != True:                break
#             conf_thres = NSG_threshold_selection.determine_threshod(frame,'Two_cluster')
            imgtest = letterbox(frame, new_shape=self.img_size)[0]
            imgtest = imgtest[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            imgtest = np.ascontiguousarray(imgtest)
            imgtest  = torch.from_numpy(imgtest).to(device)
            imgtest=imgtest.float()
            if imgtest.ndimension() == 3:
                imgtest = imgtest.unsqueeze(0)
            imgtest /= 255.0
            pred = model(imgtest, augment=augment, visualize=visualize)
            pred = non_max_suppression_.non_max_suppression_(pred, self.conf_thres, self.iou_thres, fast=True, classes=classes, agnostic=False)
            results = []
            object_tracked={i:{'dets':[],'conf':[]} for i in self.classes_interested.values()}
##########################################################
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    # Rescale boxes from img_size to frame size
                    det[:, :4] = scale_coords(imgtest.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in det:
                        if int(cls) not in self.classes_interested:
                            continue
#                         if int(cls)==7: cls=2
                        label = '%s %.2f' % (names[int(cls)], conf)
                        color = colors[int(cls) % len(colors)]
                        color = [i * 255 for i in color]
                        plot_one_box(xyxy, frame, label=label, color=color, line_thickness=3)

                        x1,y1,x2,y2 = [ int(i) for i in xyxy]
                        object_tracked[self.classes_interested[int(cls)]]['dets'].append([x1, y1, x2-x1, y2-y1])
                        results.append([int(cls),x1, y1, x2, y2,float(conf)])
                        object_tracked[self.classes_interested[int(cls)]]['conf'].append(float(conf))
            self.traces_to_save.update({frame_index : results})
            for class_name, queue in object_tracked.items():
                dets= queue['dets']
                confidence =queue['conf']
                features = encoder(frame,dets)
                detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in zip(dets, confidence, features)]
                tracker[class_name].predict()
                tracker[class_name].update(detections)
                for track in tracker[class_name].tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue 
                    bbox = track.to_tlbr()
                    x_bc,y_bc = int((bbox[0]+bbox[2])/2),int(bbox[3])
 
                    obj_id=int(track.track_id)
                    if obj_id not in trajectory[class_name]:
                        trajectory[class_name].update({obj_id:deque(maxlen=buffer)})
                        traces_to_save[class_name].update({obj_id:[]})
                        trajectory_wait[class_name].update({obj_id:0})
                    trajectory_wait[class_name][obj_id]+=1
                    trajectory[class_name][obj_id].appendleft((x_bc,y_bc))
                    if len(trajectory[class_name][obj_id])>2:
                        self.counting(trajectory[class_name][obj_id][1],(x_bc,y_bc))
                    traces_to_save[class_name][obj_id].append((x_bc,y_bc,frame_index))
                trajectory_wait[class_name] = {k:v-1 for k,v in trajectory_wait[class_name].items()}
                for k,v in trajectory_wait[class_name].items():
                    if v<0 and k in trajectory[class_name].keys():
                        del trajectory[class_name][k]
            t1= time.time()
            print('FPS_'+str(frame_index//(t1-t0)),end=";  ")
# Visualization
            if self.display:
                for class_name, queue in object_tracked.items():
                    for track in tracker[class_name].tracks:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue 
                        bbox = track.to_tlbr()
                        x_bc,y_bc = int((bbox[0]+bbox[2])/2),int(bbox[3])
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[3])), (int(bbox[2]), int(bbox[3]+20)), (255,0,0), -1)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
                        cv2.putText(frame, '#'+str(track.track_id),(int(bbox[0]), int(bbox[3]+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
                    for key, pts in trajectory[class_name].items():
                        line_color = (0, 0, 255)
                        for i in np.arange(1, len(pts)):
                            if pts[i - 1] is None or pts[i] is None:
                                continue
                            else: 
                                thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
                                cv2.line(frame, pts[i - 1], pts[i], line_color, thickness)

                t2= time.time()
                if len(self.pixelzones)!=0 and t2-t0>60:
                    
                    frame=self.show_study_zone(frame)
                    df_leave = pd.DataFrame(self.total_leaving)
                    df_enter = pd.DataFrame(self.total_entering)
                    fig=plt.figure(figsize=(5,5),dpi=300)
                    plt.title('Count With '+str(self.interval)+ ' Interval')
                    for column in df_leave:
#                         if 'P3' not in column:continue
                        average = (df_leave[column].resample(self.interval).agg('sum')+df_enter[column].resample(self.interval).agg('sum'))/2
                        average.plot(label=column)
                    plt.legend()
                    fig.canvas.draw()
                    the_map = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    the_map  = the_map.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    the_map = cv2.cvtColor(the_map,cv2.COLOR_RGB2BGR)
                    the_map_show = cv2.resize(the_map,(h,h))
                    plt.close()
                    
                frame=cv2.resize(frame, (w,h)) 
                blank_image = np.zeros((h,h,3), np.uint8)
                cv2.putText(blank_image,'No Data', (0,int(h/2)),cv2.FONT_HERSHEY_SIMPLEX ,2,(255,255,255),7)
                cv2.putText(frame, 'Frame: '+str(frame_index), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 7)
                cv2.putText(frame, 'FPS: '+str(frame_index//(t2-t0)), (0,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 7)
                vis = np.concatenate((frame, blank_image), axis=1)
                if len(self.pixelzones)!=0 and t2-t0>60:

                    vis = np.concatenate((frame, the_map_show), axis=1)
                # save video
                out.write(vis)
                #show video
                to_show = cv2.resize(vis,(int((w+h)/2),int(h/2)))
                cv2.imshow('Frame', to_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        out.release()
        cv2.destroyAllWindows()
        np.save(video_path[:-4]+"_traces.npy",self.traces_to_save)
        

    
    def counting (self,previous_point,current_point):
        traj_line = [previous_point,current_point]
        first_point = P(traj_line[0])
        last_point = P(traj_line[1])
#         print(traj_line)
        current_t = datetime.datetime.now()
        current_t = pd.Timestamp(current_t)
        for zonename,check_zone in self.pixelzones.items():
            zone = Polygon(check_zone)
            for i in range(0,len(check_zone)):
                checkline = [check_zone[i],check_zone[i-1]]

                if check_cross(checkline,traj_line) :
                    if  zone.contains(last_point,):
                        self.total_entering[zonename].update({current_t:1})
                    if  zone.contains(first_point):
                        self.total_leaving[zonename].update({current_t:1})


    def projection(self,reference):
        lat_long_reference = np.load(reference)
        real_reference = [utm.from_latlon(i,j)[:2] for i,j in lat_long_reference]

        maps = Mapping(real_reference,self.pixel_reference, self.pixelzones)
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
            