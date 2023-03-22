# Computer Vision Tools to Extract Traffic Data from Cameras.
Several packages are required to make the whole code work as follow.The deployment is designed to run on regular CPU with real-time speed. 
### Supporting packages
- `Labelme` This is a tool to design the study zones, i.e., where to count. Save the zone definition in a json file and pass its path to the main.py code. 
- `DETR and SORT` This tool is selected due to the big size of the cars exiting the garage. DETR is suitable for these irregularly large object compared to YOLO seires. 
- `P2PNET and SORT`. There is servere occlusion in the pedestrain videos, therefore we use a point based head detector and point based tracker to count the pedestrains. 

## How to use
1. Install lableme `pip install labelme` and load one screen shot of the camera and defined the study zones. 
2. git clone detr, p2pnet, and SORT.
    <br> `git clone https://github.com/facebookresearch/detr.git`    
    <br> `git clone https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet.git`    
    <br> `git clone https://github.com/adipandas/multi-object-tracker`    
3. Add the directories above to the python sys in `mian.py`.
4. pass the video path to `main.py` for now. Later the signal will be live feed. 


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


