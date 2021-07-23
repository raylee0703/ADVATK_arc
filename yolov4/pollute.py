from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all
import cv2
from PIL import Image


flags.DEFINE_string('weights', './scripts/yolov4.weights', 'pretrained weights')

flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
    

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    polluteset = Dataset(FLAGS, is_training=False)

    input_layer = tf.keras.layers.Input([cfg.TEST.INPUT_SIZE, cfg.TEST.INPUT_SIZE, 3])
    freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
    EPS = 0.2

    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)

    if FLAGS.tiny:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    else:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)

    if FLAGS.weights == None:
        print("Training from scratch")
    else:
        if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
            utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
        else:
            model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)

    for idx, (image_data, target) in enumerate(polluteset):
        if(idx >= 100):
            break
        image_data = tf.Variable(image_data)
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=False)
            giou_loss = conf_loss = prob_loss = 0
                # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]
            total_loss = giou_loss + conf_loss + prob_loss

            
            gradients = tape.gradient(total_loss, image_data)
            adv_image = image_data + EPS * tf.sign(gradients)
            adv_image = tf.clip_by_value(adv_image, 0, 1).numpy()
            adv_image = np.squeeze(adv_image)
            cln_image = np.squeeze(image_data.numpy())
            
            adv_image = adv_image[88:416-88,48:416-48] #(416*235) >> (320*240)
            
            
            
            
            
            adv_image2save = Image.fromarray((adv_image*255).astype(np.uint8))

            #adv_image2save.resize((320))

            cln_image2save = Image.fromarray((cln_image*255).astype(np.uint8))
            adv_save_path = os.path.join('./adv_results', 'adv_'+str(idx)+'.png')
            cln_save_path = os.path.join('./cln_results', 'cln_'+str(idx)+'.png')
            print("saving data No.", idx)
            adv_image2save.save(adv_save_path)
            cln_image2save.save(cln_save_path)



        





if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass