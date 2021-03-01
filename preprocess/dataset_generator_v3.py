#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 1.3.21
"""
import os
import sys
import collections
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import *
from myutils.cv_utils import *
from root_dir import DATA_DIR, ROOT_DIR


class DatasetGeneratorV3(object):
    """
    数据集生成
    """
    def __init__(self):
        self.file_path = os.path.join(DATA_DIR, 'page_dataset_raw', '7_train_ori_gaoyan.txt')
        # self.out_dir = os.path.join(DATA_DIR, 'ps_datasets_v3')
        # mkdir_if_not_exist(self.out_dir)
        # self.imgs_dir = os.path.join(self.out_dir, 'images')
        # self.lbls_dir = os.path.join(self.out_dir, 'labels')
        # mkdir_if_not_exist(self.imgs_dir)
        # mkdir_if_not_exist(self.lbls_dir)
        # self.train_imgs_dir = os.path.join(self.imgs_dir, 'train')
        # self.val_imgs_dir = os.path.join(self.imgs_dir, 'val')
        # mkdir_if_not_exist(self.train_imgs_dir)
        # mkdir_if_not_exist(self.val_imgs_dir)
        # self.train_lbls_dir = os.path.join(self.lbls_dir, 'train')
        # self.val_lbls_dir = os.path.join(self.lbls_dir, 'val')
        # mkdir_if_not_exist(self.train_lbls_dir)
        # mkdir_if_not_exist(self.val_lbls_dir)

        self.out_dir = os.path.join(ROOT_DIR, '..', 'datasets', 'segmention_ds_v3_4')
        self.imgs_dir = os.path.join(self.out_dir, 'images')
        self.lbls_dir = os.path.join(self.out_dir, 'labels')
        self.train_imgs_dir = os.path.join(self.imgs_dir, 'train')
        self.val_imgs_dir = os.path.join(self.imgs_dir, 'val')
        self.train_lbls_dir = os.path.join(self.lbls_dir, 'train')
        self.val_lbls_dir = os.path.join(self.lbls_dir, 'val')


    @staticmethod
    def convert(iw, ih, box):
        """
        将标注的xml文件标注转换为darknet形的坐标
        """
        iw = float(iw)
        ih = float(ih)
        dw = 1. / iw
        dh = 1. / ih
        x = (box[0] + box[2]) / 2.0 - 1
        y = (box[1] + box[3]) / 2.0 - 1
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    @staticmethod
    def filer_iou(bbox_list):
        """
        过滤iou, 避免bbox重叠
        """
        n_b = len(bbox_list)
        flags = [True] * n_b
        for i in range(0, n_b):
            box_i = bbox_list[i]
            if not flags[i]:
                continue
            for j in range(i+1, n_b):
                box_j = bbox_list[j]
                v_iou = min_iou(box_i, box_j)
                if v_iou > 0.5:
                    flags[j] = False
        new_bbox_list = []
        for i in range(0, n_b):
            if flags[i]:
                new_bbox_list.append(bbox_list[i])

        # print('[Info] 旧: {}, 新: {}'.format(len(bbox_list), len(new_bbox_list)))
        return new_bbox_list


    @staticmethod
    def process_line(idx, data_line, img_dir, lbl_dir):
        print('-' * 50)
        img_format = "http://sm-transfer.oss-cn-hangzhou.aliyuncs.com/yjb219735/ori_imgs/{}"
        # print('[Info] idx: {}, data_line: {}'.format(idx, data_line))
        data_dict = json.loads(data_line)
        img_name = data_dict['url']

        name_x = img_name.split('.')[0]
        img_path = os.path.join(img_dir, '{}.jpg'.format(name_x))
        lbl_path = os.path.join(lbl_dir, '{}.txt'.format(name_x))

        img_url = img_format.format(img_name)
        # print('[Info] img_url: {}'.format(img_url))
        is_ok, img_bgr = download_url_img(img_url)
        # print('[Info] img_bgr: {}'.format(img_bgr.shape))

        cv2.imwrite(img_path, img_bgr)  # 写入图像

        ih, iw, _ = img_bgr.shape  # 高和宽
        # show_img_bgr(img_bgr)
        coord_boxes = data_dict['coord']
        label_list = data_dict['label']
        # print('[Info] boxes: {}, labels: {}'.format(len(coord_boxes), len(label_list)))

        label_boxes_dict = collections.defaultdict(list)
        for box, label in zip(coord_boxes, label_list):
            box = [int(x) for x in box]
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            label_boxes_dict[label].append(box)

        img_boxes = []
        # print('[Info] label_boxes_dict: {}'.format(label_boxes_dict))
        for label in label_boxes_dict.keys():
            # print('[Info] label: {}'.format(label))
            if label == 5:
                bbox_list = label_boxes_dict[label]
                bbox_list = DatasetGeneratorV3.filer_iou(bbox_list)
                # draw_box_list(img_bgr, bbox_list, is_new=True, is_show=True)

                for bbox in bbox_list:
                    bbox_yolo = DatasetGeneratorV3.convert(iw, ih, bbox)
                    bbox_yolo = [str(round(i, 6)) for i in bbox_yolo]
                    img_boxes.append(" ".join(["0", *bbox_yolo]))

        create_file(lbl_path)
        write_list_to_file(lbl_path, img_boxes)
        print('[Info] idx: {} 处理完成: {}'.format(idx, img_path))

    def process(self):
        print('[Info] 开始生成!')
        print('[Info] file_path: {}'.format(self.file_path))

        data_lines = read_file(self.file_path)
        data_lines = data_lines[:20]
        random.seed(47)
        random.shuffle(data_lines)
        print('[Info] 文件数: {}'.format(len(data_lines)))

        n_split = len(data_lines) // 10
        train_lines = data_lines[:n_split*9]
        val_lines = data_lines[n_split*9:]
        print('[Info] 训练: {}, 测试: {}'.format(len(train_lines), len(val_lines)))

        pool = Pool(processes=20)
        for idx, data_line in enumerate(train_lines):
            # DatasetGeneratorV3.process_line(idx, data_line, self.train_imgs_dir, self.train_lbls_dir)
            pool.apply_async(DatasetGeneratorV3.process_line,
                             (idx, data_line, self.train_imgs_dir, self.train_lbls_dir))

        for idx, data_line in enumerate(val_lines):
            # DatasetGeneratorV3.process_line(idx, data_line, self.val_imgs_dir, self.val_lbls_dir)
            pool.apply_async(DatasetGeneratorV3.process_line,
                             (idx, data_line, self.val_imgs_dir, self.val_lbls_dir))
        pool.close()
        pool.join()

        print('[Info] 处理完成: {}'.format(self.out_dir))


def main():
    dg3 = DatasetGeneratorV3()
    dg3.process()


if __name__ == '__main__':
    main()
