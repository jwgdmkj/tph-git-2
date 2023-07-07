import os
import sys
import cv2
import yaml
import argparse
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='convert_gt2eval.py')
    parser.add_argument('--dataset', type=str, default='obstacle', help='dataset name')
    parser.add_argument('--source', type=str, default='dataset/test/labels/', help='ground truth label directory path')
    parser.add_argument('--target', type=str, default='tools/eval/input/ground-truth', help='target directory path')

    opt = parser.parse_args()
    dataset = opt.dataset
    source = opt.source
    target = opt.target
    image_dir = source.replace("labels", "images")

    dataset_path = os.path.join("/workspace/data", f"{dataset}.yaml")
    dataset_file =  open(dataset_path, "r")
    dataset_info = yaml.safe_load(dataset_file)
    classes = dataset_info["names"]
    dataset_file.close()

    if not os.path.exists(source) or len(os.listdir(source)) == 0:
        print("Source labels are not existed")
        exit()

    if not os.path.exists(target):
        os.makedirs(target)

    image_paths = [os.path.join(image_dir, image_name) for image_name in sorted(os.listdir(image_dir))]
    label_paths = [os.path.join(source, label) for label in sorted(os.listdir(source))]
    target_paths = [os.path.join(target, label) for label in sorted(os.listdir(source))]

    for i in range(len(image_paths)):
        image_path = image_paths[i]
        label_path = label_paths[i]
        target_path = target_paths[i]
        image = cv2.imread(image_path)

        label_file = open(label_path, 'r')
        target_file = open(target_path, 'w')
        for line in label_file:
            cls_idx, s_center_x, s_center_y, s_w, s_h = line.split(" ")
            cls = classes[int(cls_idx)]
            w = int(image.shape[1] * float(s_w))
            h = int(image.shape[0] * float(s_h))
            left = int(image.shape[1] * float(s_center_x) - (w/2))
            top = int(image.shape[0] * float(s_center_y) - (h/2))
            right = left + w
            bottom = top + h
            target_file.write(f"{cls} {left} {top} {right} {bottom}\n")
        label_file.close()
        target_file.close()
        if i % 20 == 0:
            print(f"\r{i:05d}/{len(label_paths):05d} - {target_path}", end="")
    print()