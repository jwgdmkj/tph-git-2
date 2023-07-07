import os
import cv2
import sys
import argparse
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import yaml

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

class TRTyolov7:
    def __init__(self, model_path, class_names):
        self.imgsz = (640, 640)
        self.mean = None
        self.std = None
        self.class_names = class_names
        self.n_classes = len(self.class_names)

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger, '')  # initialize TensorRT plugins
        with open(model_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    @staticmethod
    def postprocess(predictions, ratio):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        return dets

    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        return data

    def inference_image(self, image, confidence_threshold=0.1):
        img, ratio = preproc(image, self.imgsz, self.mean, self.std)
        data = self.infer(img)
        num, final_boxes, final_scores, final_cls_inds = data
        final_boxes = np.reshape(final_boxes / ratio, (-1, 4))
        dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1),
                               np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        results = []
        if dets is not None:
            boxes, scores, cls_idxs = dets[:, :4], dets[:, 4], dets[:, 5]
            for i in range(len(boxes)):
                cls_idx = int(cls_idxs[i])
                cls = self.class_names[cls_idx]
                score = float(scores[i])
                box = boxes[i]
                left = int(box[0])
                top = int(box[1])
                right = int(box[2])
                bottom = int(box[3])
                results.append(f"{cls} {score} {left} {top} {right} {bottom}")
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='get_result_evel.py')
    parser.add_argument('--dataset', type=str, default='obstacle', help='dataset name')
    parser.add_argument('--model', type=str, default='/data/model/yolov7-tiny-final/yolov7-tiny/weights/best_289.trt', help='model path')
    parser.add_argument('--image_dir', type=str, default='dataset/test/images/', help='image directory path')
    parser.add_argument('--target', type=str, default='tools/eval/input/detection-results', help='target directory path')

    opt = parser.parse_args()
    dataset = opt.dataset
    model = opt.model
    image_dir = opt.image_dir
    target = opt.target

    if not os.path.exists(image_dir) or len(os.listdir(image_dir)) == 0:
        print("images are not existed")
        exit()

    if not os.path.exists(target):
        os.makedirs(target)

    dataset_path = os.path.join("/workspace/data", f"{dataset}.yaml")
    dataset_file = open(dataset_path, "r")
    dataset_info = yaml.safe_load(dataset_file)
    class_names = dataset_info["names"]
    dataset_file.close()

    image_paths = [os.path.join(image_dir, image_name) for image_name in sorted(os.listdir(image_dir))]
    target_paths = [os.path.join(target, image_name.replace(".jpg", ".txt")) for image_name in sorted(os.listdir(image_dir))]

    model = TRTyolov7(model_path=model, class_names=class_names)

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        results = model.inference_image(image)
        result_file = open(target_paths[i], "w")

        for result in results:
            result_file.write(result + "\n")
        result_file.close()
        if i % 20 == 0:
            print(f"\r{i:05d}/{len(image_paths):05d} - {target_paths[i]}", end="")
    print()