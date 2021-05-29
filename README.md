# ADVATK_arc

To get the model verified on PC, please refer to `test/` .

## Getting Started With ARC Project

To train the defense model for ARC dev board: 
```
$ cd model
$ python3 train_autoencoder_for_arc.py
```

To generate trained model file:
```
$ python3 convert_to_onnx.py
$ python3 onnx_to_tfilte.py
```
