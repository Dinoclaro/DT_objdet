# **DT_objdet**

## About the Directory

This directory contains a Duckietown-compliant Docker image of a ROS agent to run a YOLOv5 model and Braitenberg controller on a Duckiebot.

## Prerequisites

The list below states the prerequisites to use this directory.

1. Laptop Setup:
   - Ubuntu 22.04 (Recommended)
   - Docker
   - Duckietown shell

2. Assembled Duckiebot: The Duckiebot should be able to boot up. Follow these setup instructions:
   - [Assembly](https://docs.duckietown.com/daffy/opmanual-duckiebot/assembly/db21m/index.html)
   - [Flashing SD Card](https://docs.duckietown.com/daffy/opmanual-duckiebot/setup/setup_sd_card/index.html)
   - [First Boot](https://docs.duckietown.com/daffy/opmanual-duckiebot/setup/setup_boot/index.html)
   - [Manual Control](https://docs.duckietown.com/daffy/opmanual-duckiebot/operations/make_it_move/index.html)

## Instructions

1. Clone this repository 
```
git clone https://github.com/Dinoclaro/DT_objdet.git
```
Alternatively, the repository can be downloaded as a zip file by clicking on the green "code" drop-down menu. 

2. cd into this repository
```
cd DT_objdet
```
3. Switch on the duckiebot and ensure it is discoverable 
```
dts fleet discover ROBOT_NAME
```
4. Build and run the image. Note this runs the object_detection_node on the user's machine and connects to the duckiebot master node. 
```
dts devel build -f
dts devel run -R ROBOT_NAME
```
5. Open a new terminal and run the gui tools to use ROS tools
```
 dts start_gui_tools ROBOT_NAME
```
6 Open rqt image tools to view the various camera feeds including the YOLOv5 model predictions.
```
rqt_image_view
```

## Issues
Currently, the image only runs when built locally and with the master node running on the duckiebot. The image successfully builds on the duckiebot but fails to run due to ```torchvision``` crashing when using the ```dts devel run -H ROBOT_NAME``` command. 
