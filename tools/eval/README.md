# TensorRT model Evaluation
## mAP 측정 방법
1. 모델 변환
2. ```convert_gt2eval.py``` : ground truth label(*.txt) -> evaluation form 변환
3. ```get_result_evel.py``` : tensorrt model로 image inference 결과 추출
4. mAP 추출
```shell
# 1. 모델 변환
cd /workspace
python3 export.py --weights ${WEIGHT_DIR}/best.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.45 --conf-thres 0.1 --img-size 640 640
git clone https://github.com/JinhaSong/TensorRT-For-YOLO-Series.git
cd TensorRT-For-YOLO-Series
python export.py -o ${WEIGHT_DIR}/best.onnx -e ${WEIGHT_DIR}/best.trt -p fp16

# 2. ground truth evaluation form 변환
python3 tools/eval/convert_gt2eval.py --dataset=obstacle --source=${TEST_LABEL_DIR} --target=tools/evel/input/ground-truth

# 3. TensorRT 모델로 image inference 결과 추출
python3 tools/eval/get_result_eval.py --dataset=obstacle --model=${TRT_MODEL_PATH} --image_dir=${TEST_IMAGE_DIR} --target=tools/evel/input/detection-results

# 4. mAP 추출
python3 tools/eval/evaluation.py --iou=0.5 --gt=/workspace/tools/eval/input/ground-truth --det=/workspace/tools/eval/input/detection-results
```