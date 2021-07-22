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

## Manual
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
```
$ cd senario_1
$ make
$ make flash
load *.img into the board
press the "reset" button 
$ python3 recv_img.py
```
### To use the system for senario 2:
```
$ cd senario_2
copy one of test_sample from test_samples/ to src/
$ make
$ make flash
load *.img into the board
$ screen /dev/ttyUSB0 115200  # get the result
verify the result using YOLO model with the images in test_sample_images/
```
