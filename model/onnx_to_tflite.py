import onnx
import numpy as np

def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 1, 40, 40)
        yield [data.astype(np.float32)]

try:

    onnx_model = onnx.load("magnet.onnx")  # load onnx model
    from onnx_tf.backend import prepare
    tf_exp = prepare(onnx_model)  # prepare tf representation
    tf_exp.export_graph("magnet_saved_model")  # export the model
    print('saved_model export success, saved as magnet_saved_model' )
except Exception as e:
    print('saved_model export failure: %s' % e)

try:
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model('magnet_saved_model')
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = representative_dataset
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    #converter.inference_input_type = tf.int8
    #converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open('magnet.tflite','wb') as g:
        g.write(tflite_model)
    print('tflite export success, saved as magnet.tflite' )
except Exception as e:
    print('tflite export failure: %s' % e)
