# Computer vision tools to extract game day data from cameras with visulization.
Several packages are required to make the whole code work as follow.The deployment is designed to run on regular CPU with real-time speed. 
### Supporting packages
- `Labelme` This is a tool to design the study zones, i.e., where to count. Save the zone definition in a json file and pass its path to the main.py code. 
- `DETR and SORT` This tool is selected due to the big size of the cars exiting the garage. DETR is suitable for these irregularly large object compared to YOLO seires. 
- `P2PNET and SORT`. There is servere occlusion in the pedestrain videos, therefore we use a point based head detector and point based tracker to count the pedestrains. 

## How to use
The class VideoProcesser_CPU has the following attributes. 

- threshold Detection threshold which is luminence dependend (We suggest the two cluster mode).
- img_size The input images will be resized to this dimension.
- display Show and save the resulting videoin the same folder.
- skipframe Skip detecting frames in order to get faster processing speed. Note the more skipped frames, the more like to mis-track objects.
- studyzone Lableme polygon annotation on the image pixel coordinates including the reference points ({reference:[point1,point2,point3,point4]}) and study zones ({name:[COORDINATES]}). 
- reference points list[point1,point2,point3,point4]. 



### Papers about this work:

@article{pi2022visual,
  title={Visual recognition for urban traffic data retrieval and analysis in major events using convolutional neural networks},
  author={Pi, Yalong and Duffield, Nick and Behzadan, Amir H and Lomax, Tim},
  journal={Computational Urban Science},
  volume={2},
  number={1},
  pages={1--16},
  year={2022},
  publisher={Springer}
}

@article{pi2023lane,
  title={Lane-specific speed analysis in urban work zones with computer vision},
  author={Pi, Yalong and Duffield, Nick and Behzadan, Amir and Lomax, Tim},
  journal={Traffic injury prevention},
  pages={1--9},
  year={2023},
  publisher={Taylor \& Francis}
}

@incollection{pi2021computer,
  title={Computer vision and multi-object tracking for traffic measurement from campus monitoring cameras},
  author={Pi, Yalong and Duffield, Nick and Behzadan, Amir H and Lomax, Tim},
  booktitle={Computing in Civil Engineering 2021},
  pages={950--958}
}


