# Prepare Data
*From the root directory*
*Specify the data directory which has following structure:*
```
+ path
    + annotations
        - 1.XML
        - 2.XML
        - ...
    +images
        - 1.JPG
        - 2.JPG
        - ...
    - label_map.pbtxt
```
```
python utils/preprocess.py --path YOUR_DATA_DIR --class_all [--arguments 1,2][--argument_all][--classes 3,4][--previous PREVIOUS_DATA_DIR]
python utils/create_tf_record --input=YOUR_DATA_DIR/TIME_STAMP_output [--output=data/face][--set=train,valid,trainvalid,dev,test]
```


# Download pretrained model
*From the root directory*
```
chmod +x ./models/modelzoo/download_ssdlite_mobilenet_v2_coco.sh
./models/modelzoo/download_ssdlite_mobilenet_v2_coco.sh
```



# Train
*From the root directory*
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
export CUDA_VISIBLE_DEVICES=1
python object_detection/train.py \
    --pipeline_config_path=models/face/pipeline.config \
    --train_dir=models/face/train
```



# Valid
*From the root directory*
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
export CUDA_VISIBLE_DEVICES=0
python object_detection/eval.py \
    --pipeline_config_path=models/face/pipeline.config \
    --checkpoint_dir=models/face/train \
    --eval_dir=models/face/eval
```



# Tensorboard
*From the root directory*
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
export CUDA_VISIBLE_DEVICES=0
tensorboard --logdir=models/face
```



# Export model.pb
*From the root directory*
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
export CUDA_VISIBLE_DEVICES=0
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path models/face/pipeline.config \
    --trained_checkpoint_prefix models/face/train_debug/model.ckpt-0 \
    --output_directory models/face/inference
```


# Quantize model
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
   --in_graph=YOUR_MODEL_EXPORT_DIR/inference_quantize_full/frozen_inference_graph.pb \
   --out_graph=YOUR_MODEL_EXPORT_DIR/inference_quantize_full/tfmobile_model.pb \
   --inputs=image_tensor \
   --outputs=detection_boxes,detection_scores,detection_classes,num_detections \
   --transforms='remove_attribute(attribute_name=_class) quantize_weights quantize_nodes sort_by_execution_order'

bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=YOUR_MODEL_EXPORT_DIR/inference_half/frozen_inference_graph.pb \
  --output_file=YOUR_MODEL_EXPORT_DIR/inference_half/tflite_model.tflite \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=FLOAT \
  --input_shapes="1,300,300,3" \
  --input_arrays=image_tensor \
  --output_arrays="box_encodings,class_predictions_with_background" \
  --std_values=127.5 --mean_values=127.5 \

bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=YOUR_MODEL_EXPORT_DIR/inference_quantize_half/frozen_inference_graph.pb \
  --output_file=YOUR_MODEL_EXPORT_DIR/inference_quantize_half/tflite_model_quantize.tflite \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=QUANTIZED_UINT8 \
  --input_shapes="1,300,300,3" \
  --input_arrays=image_tensor \
  --output_arrays="box_encodings,class_predictions_with_background" \
  --std_values=127.5 --mean_values=127.5



