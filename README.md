# Computer Vision and Multi-object Tracking for Traffic Measurement from Campus Monitoring Cameras
Files in the support function folder are the tracking, counting and mapping package. Essentially, this system is an integration of YOLOv5+Deep_SORT+Homography_Transformation for pedestrian and vehicle, and all other classes in COCO. Several packages are required to make the whole code work as follow. The CPU jupyter lab file is the main deployment on regular CPU with faster than real-time speed. 
### Dependencies
- `YOLOv5`
- `Deep-SORT`
- `Pytorch>1.9`
- `Python>3.8`
- `Tensorflow 1.0`
- `Labelme`

## How to use
The class VideoProcesser_CPU has the following attributes. 

- threshold Detection threshold which is luminence dependend (We suggest the two cluster mode).
- img_size The input images will be resized to this dimension.
- display Show and save the resulting videoin the same folder.
- skipframe Skip detecting frames in order to get faster processing speed. Note the more skipped frames, the more like to mis-track objects.
- studyzone Lableme polygon annotation on the image pixel coordinates including the reference points ({reference:[point1,point2,point3,point4]}) and study zones ({name:[COORDINATES]}). 
- reference points list[point1,point2,point3,point4]. 



### LaTeX citation:

@article{Pi2021,
    author  = {Yalong Pi and Nick Duffield and Amir H. Behzadan and Tim Lomax},
    title   = {Computer Vision and Multi-object Tracking for Traffic Measurement from Campus Monitoring Cameras},

Please cite the article if you use the dataset, model or method(s), or find the article useful in your research. Thank you!



