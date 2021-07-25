# Adversarial Attack Detector in Self-Driving Vehicle

## Introduction
In this project, we design a system of obeject detection intending for self-driving car usage. The system is able to defense adversarial attack. The system can identify the inputs which are attacked and we use YOLO to verify the results.
[demo video](https://drive.google.com/file/d/1AKKPc-QH2vZLM4rj__xob8UrC6t0MjXO/view)
## System Architecture
![](https://i.imgur.com/Mu9Ff4q.png)
## Overview of Demo
Considering some limitation of the development board, we demonstrate the system in two different scenario. The overview of each scenarios is shown below. For more information, please refer to the demo part of our [demo video](https://drive.google.com/file/d/1AKKPc-QH2vZLM4rj__xob8UrC6t0MjXO/view).
### Scenario 1: Clean data captured by camera
![](https://i.imgur.com/sl9Xs3y.png)
### Scenario 2: Attacked data loaded with model
![](https://i.imgur.com/wlCFlDb.png)

## HW/SW Setup
1. Install [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/cli-installation)
2. Install [ARC GNU ToolChain](https://github.com/foss-for-synopsys-dwc-arc-processors/toolchain/releases)
3. Donwlaod and Setup [SDK](https://github.com/foss-for-synopsys-dwc-arc-processors/arc_contest) 
4. Clone this repository to your local computer
5. Modify the `ROOT_PATH` in scenario_1/Makefile and scenario_2/Makefile
6. Connect the WE-I to the computer by USB cable

## User manual

### To use the system for scenario 1:
1. `$ cd scnario_1`
2. Place tflite file under `checkpopints/` directory. Pretrained model file is available at https://drive.google.com/file/d/1OZqE7vJ8KH-Pt2yfjFHCvaPEqgmCetMN/view?usp=sharing
3. `$ make`
4. `$ make flash`
5. `$ himax-flash-tool -f output_gnu.img`
6. Press the "reset" button on the board
7. `$ python3 arc_detect.py #receive image from the baord and run YOLO inference`
### To use the system for scenario 2:
1. `$ cd scenario_2`
2. Copy one of test_sample from test_samples/ to src/
3. `$ make`
4. `$ make flash`
5. `$ himax-flash-tool -f output_gnu.img`
6. `screen /dev/ttyUSB0 115200  #see the result`
7. Verify the result using YOLO model with the images in test_sample_images/
