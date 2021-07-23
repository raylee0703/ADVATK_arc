# Adversarial Attack Detector in Self-Driving Vehicle

## Introduction
In this project, we design a system of obeject detection intending for self-driving car usage. The system is able to defense adversarial attack. The system can identify the inputs which are attacked and we use YOLO to verify the results.
## System Architecture
### Senario 1: Clean data captured by camera
![](https://i.imgur.com/sl9Xs3y.png)
### Senario 2: Attacked data loaded with model
![](https://i.imgur.com/wlCFlDb.png)

## Model Verification
To get the model verified on PC, please refer to `test/` .

## HW/SW Setup
1. Install [ARC GNU ToolChain](https://github.com/foss-for-synopsys-dwc-arc-processors/toolchain/releases)
2. Donwlaod and Setup [SDK](https://github.com/foss-for-synopsys-dwc-arc-processors/arc_contest) 
3. Clone this repository to your local computer
4. Modify the `ROOT_PATH` in senario_1/Makefile and senario_2/Makefile
5. Connect the WE-I to the computer by USB cable

## User manual
### To train the defense model for ARC dev board: 
```
$ cd model
$ python3 train_autoencoder_for_arc.py
```
### To generate trained model file:
```
$ python3 convert_to_onnx.py
$ python3 onnx_to_tfilte.py
```
### To use the system for senario 1:
1. `$ cd senario_1`
2. `$ make`
3. `$ make flash`
4. `$ himax-flash-tool -f output_gnu.img`
5. press the "reset" button on the board
6. `$ python3 recv_img.py #receive image from the baord and run YOLO inference`
### To use the system for senario 2:
1. `$ cd senario_2`
2. copy one of test_sample from test_samples/ to src/
3. `$ make`
4. `$ make flash`
5. `$ himax-flash-tool -f output_gnu.img`
6. `screen /dev/ttyUSB0 115200  #see the result`
7. verify the result using YOLO model with the images in test_sample_images/
