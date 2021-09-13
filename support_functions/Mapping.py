import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point as P
from shapely.geometry.polygon import LinearRing, Polygon


class Point:
	def __init__(self,x,y):
		self.x = x
		self.y = y

def ccw(A,B,C):
	return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)

def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def check_cross(checkline, traj_line):
    
    previous_x,previous_y = traj_line[1][0],traj_line[1][1]
    next_x,next_y = traj_line[0][0],traj_line[0][1]
    
    return intersect(Point(previous_x,previous_y), Point(next_x,next_y),Point(checkline[0][0],checkline[0][1]),Point(checkline[1][0],checkline[1][1]))


class Mapping:
    def __init__(self,real_reference,pixel_reference, pixelzones):
        self.real_reference =real_reference
        self.pixel_reference=pixel_reference
        self.pixelzones = pixelzones
        self.M = cv2.getPerspectiveTransform(np.float32(self.pixel_reference),np.float32(self.real_reference))
        
        self.checkzones = {}
        for lane_name, pixelzone in self.pixelzones.items():
            checkzone =[]
            for each_point in pixelzone:
                x_bc,y_bc = each_point

                point_1 = np.array([[[x_bc,y_bc]]], dtype='float32')  
                projected_position = cv2.perspectiveTransform(point_1, self.M)
                position_on_map = tuple(projected_position[0][0])
                checkzone.append(position_on_map)
            self.checkzones.update({lane_name:checkzone})
#         print("mapping running!")
        
    def show_study_zones(self):
        
#         fig, ax = plt.subplots(1,1,figsize = (10,10),dpi=100)
        my_dpi=96
        fig = plt.figure(figsize=(1024/my_dpi, 1024/my_dpi), dpi=my_dpi)
        plt.xlabel('X', fontsize=10,)
        plt.ylabel('Y', fontsize=10)
        plt.ticklabel_format(useOffset=False, style='plain')
        
        for lane_name, checkzone in self.checkzones.items():
            poly = Polygon(checkzone)
            x,y = poly.exterior.xy
            ring = LinearRing(checkzone)
            x, y = ring.xy
            plt.plot(x, y)    
            plt.text(checkzone[0][0],checkzone[0][1],lane_name)
        return fig
                
    def project_trajectory(self,trajectories):
        projected_trajectory = {}

        for k,v in trajectories.items():
            mapped= []
            for each_point in v:
                point = np.array([[[each_point[0],each_point[1]]]], dtype='float32')  
                projected_position = cv2.perspectiveTransform(point, self.M)
                position_on_map = tuple(projected_position[0][0])
                last_point = P(position_on_map)
#                 print(last_point)

#                 for check_zone in self.checkzones.values():
#                     polygon = Polygon(check_zone)
#                     if  polygon.contains(last_point):
                mapped.append(position_on_map)
#                     else:continue
            projected_trajectory[k]=mapped
        return projected_trajectory
    def Waypoint (self,trajectories):
        results = {}
        for k,v in trajectories.items():
            mapped= []
            for each_point in v:
                point = np.array([[[each_point[0],each_point[1]]]], dtype='float32')  
                projected_position = cv2.perspectiveTransform(point, self.M)
                position_on_map = tuple(projected_position[0][0])
                last_point = P(position_on_map)
                mapped.append(position_on_map)
            if (len(mapped))<2:continue
            traj_line=[mapped[0],mapped[1]]
            last_point = P(mapped[0])

            for lane_name, check_zone in self.checkzones.items():
                zone = Polygon(check_zone)
                for i in range(1,len(check_zone)):
                    checkline = (check_zone[i],check_zone[i-1])
            #         break
                    if(check_cross(checkline,traj_line)) :
                        sense = ""
                        if  zone.contains(last_point):
                            sense="ingress "
                        else:
                            sense="egress "
                        results.update({k:[traj_line[0][0],traj_line[0][1],traj_line[1][0],traj_line[1][1],checkline[0][0],checkline[0][1],checkline[1][0],checkline[1][1],sense]}) 
        return results