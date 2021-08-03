/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

#include "main_functions.h"
#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "model.h"
#include "test_samples.h"
#include "hx_drv_tflm.h"
//#include "synopsys_wei_delay.h"

#include "stdio.h"
#include "string.h"
// Globals, used for compatibility with Arduino-style sketches.

uint8_t string_buf[100] = "test\n";
uint8_t image_buf[320*240+1];

hx_drv_sensor_image_config_t pimg_config;

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 200 * 1024;
//alignas(16) static uint8_t tensor_arena[kTensorArenaSize] = {0};
alignas(16) static uint8_t tensor_arena[kTensorArenaSize] = {0};
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(magnet_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    hx_drv_uart_print("Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<15> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddMul();
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddPad();
  micro_op_resolver.AddTranspose();
  micro_op_resolver.AddLogistic();
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddResizeNearestNeighbor();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddSplit();
  micro_op_resolver.AddDequantize();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    hx_drv_uart_print("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
  output = interpreter->output(0);
  // Obtain quantization parameters for result dequantization
}

// The name of this function is important for Arduino compatibility.
void loop()
{

  uint8_t key_data = '\0';
  hx_drv_uart_initial(UART_BR_57600);

  //sensor start capture and start streaming
  if(hx_drv_sensor_initial(&pimg_config) == HX_DRV_LIB_PASS);
  uint8_t * img_ptr;
  int count=0;
  while (1)
  {
    hx_drv_uart_getchar(&key_data);
    if(key_data == 'A')
    {
      if(hx_drv_sensor_capture(&pimg_config) == HX_DRV_LIB_PASS)
      {

        hx_drv_uart_print("S");

        img_ptr = (uint8_t *) pimg_config.raw_address;

        for(uint32_t height_cnt = 0; height_cnt < pimg_config.img_height; height_cnt+=2)
        {
          for(uint32_t width_cnt = 0; width_cnt < pimg_config.img_width; width_cnt+=2)
          {
            hx_drv_uart_print("%c", *img_ptr);
            image_buf[count] = *img_ptr;
            img_ptr = img_ptr + 2;
            count++;
          }
        }
        image_buf[320*240] = '\0';
      }
      break;
    }
    key_data = '\0';
  }

  img_ptr = (uint8_t *) pimg_config.raw_address;
  float total=0.0;
  float block_average = 0.0;
  int count_clean=0;

  int k, l, m, n, block_no=0;
  for(k=0;k<6;k++){
    for(l=0;l<8;l++){
      for(m=0;m<40;m++){
        for(n=0;n<40;n++){
          test_samples[block_no].image[n+40*m] = img_ptr[n+320*m+40*l+12800*k];
        }
      }
      block_no++;
    }
  }


  for (int j = 0; j < kNumSamples; j++)
  {
    uint8_t temp;
    // Write image to input data
    float image_input[kImageSize];
    for (int i = 0; i < kImageSize; i++) {
      temp = test_samples[j].image[i];
      *((float*)input->data.data + i*sizeof(float)) = image_input[i] = ((float)temp/255.0);
    }

    // Run the model on this input and make sure it succeeds.
    if (kTfLiteOk != interpreter->Invoke()) {
      hx_drv_uart_print("Invoke failed.");
    }
    float* results_ptr = (float*)output->data.data;
    float sum = 0.0;
    float part;
    for(int i=0;i<kImageSize;i++){
      part = results_ptr[i] - image_input[i];
      part = part * part ;
      sum += part;
      total += part;
    }
    block_average += sum/1600;
    if(sum/1600 < 0.08)
    	count_clean += 1;
    //hx_drv_uart_print("Block sum = %f", sum);
    hx_drv_uart_print("Block number: %2d", j);

  }

  //hx_drv_uart_print("TOTAL = %d", (int)(total*100));
  //hx_drv_uart_print("Block Average = %d", (int)((block_average/192)*100000));
  hx_drv_uart_print("Count clean = %2d", count_clean);
  while(1);
}
