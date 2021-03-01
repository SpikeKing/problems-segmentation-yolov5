#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 22.12.20
"""

import os
import sys
import cv2

from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import *
from myutils.cv_utils import *
from root_dir import DATA_DIR


class DatasetGenerator(object):
    """
    生产数据集
    """
    def __init__(self):
        pass

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
    def check_darknet_data(img_bgr, data_line):
        ih, iw, _ = img_bgr.shape
        items = data_line.split(" ")
        items = [float(i) for i in items]
        x, y, w, h = items[1:]
        x, y, w, h = x*iw, y*ih, w*iw, h*ih
        x_min, x_max = x - w // 2, x + w // 2
        y_min, y_max = y - h // 2, y + h // 2
        img_out = draw_box(img_bgr, [x_min, y_min, x_max, y_max], is_show=False)
        return img_out

    @staticmethod
    def get_boxes_from_items(label_str):
        """
        解析标注框
        """
        # print('[Info] label_str: {}'.format(label_str))
        label_list = json.loads(label_str)
        coord_list = label_list[0]
        box_list =[]
        for coord in coord_list:
            box = coord["coord"]
            box_list.append(box)
            # print('[Info] box: {}'.format(box))
        return box_list

    @staticmethod
    def process_line(data_line):
        """
        处理文件行
        """
        items = data_line.split(';')
        label_str = items[3]
        url = items[5]
        url = url.split("?")[0]
        img_name = url.split("/")[-1]
        try:
            p_box_list = DatasetGenerator.get_boxes_from_items(label_str)
        except Exception as e:
            print('[Info] label error: {}, label_str: {}'.format(url, label_str))
            p_box_list = []

        # print('[Info] url: {}'.format(url))
        # print('[Info] img_name: {}'.format(img_name))
        # print('[Info] p_box_list: {}'.format(p_box_list))
        return img_name, url, p_box_list

    @staticmethod
    def generate_item(idx, url, p_box_list, out_img_path, out_txt_path):
        if not p_box_list:
            return
        is_ok, img_bgr = download_url_img(url)
        cv2.imwrite(out_img_path, img_bgr)

        create_file(out_txt_path)

        ih, iw, _ = img_bgr.shape
        for p_box in p_box_list:
            x, y, w, h = DatasetGenerator.convert(iw, ih, p_box)
            label = 0
            box_str_list = [str(round(i, 6)) for i in [x, y, w, h]]
            data_line = " ".join([str(label), *box_str_list])
            # print('[Info] box: {}'.format(data_line))
            # DataPreprocess.check_darknet_data(img_bgr, data_line)  # 测试图像标注
            write_line(out_txt_path, data_line)

        print('[Info] idx: {}'.format(idx))

    def generate(self):
        """
        生成数据集
        """
        random.seed(47)

        data_dir = os.path.join(DATA_DIR, 'raw_data')
        print('[Info] 数据文件夹: {}'.format(data_dir))

        # 创建数据文件夹
        out_dataset_dir = os.path.join(DATA_DIR, 'ps_datasets')

        out_images_dir = os.path.join(out_dataset_dir, 'images')
        out_images_train_dir = os.path.join(out_images_dir, 'train')
        out_images_val_dir = os.path.join(out_images_dir, 'val')

        out_labels_dir = os.path.join(out_dataset_dir, 'labels')
        out_labels_train_dir = os.path.join(out_labels_dir, 'train')
        out_labels_val_dir = os.path.join(out_labels_dir, 'val')

        mkdir_if_not_exist(out_dataset_dir)
        mkdir_if_not_exist(out_images_dir)
        mkdir_if_not_exist(out_images_train_dir)
        mkdir_if_not_exist(out_images_val_dir)
        mkdir_if_not_exist(out_labels_dir)
        mkdir_if_not_exist(out_labels_train_dir)
        mkdir_if_not_exist(out_labels_val_dir)
        print('[Info] 文件夹新建完成!')

        paths_list, names_list = traverse_dir_files(data_dir)
        print('[Info] 文件列表: {}'.format(paths_list))

        data_lines = []
        for path, name in zip(paths_list, names_list):
            sub_data_lines = read_file(path)
            data_lines += sub_data_lines[1:]  # 去除第1行label

        print('[Info] 文件行数: {}'.format(len(data_lines)))

        img_dict = {}
        for data_line in data_lines:
            img_name, url, p_box_list = self.process_line(data_line)
            if p_box_list and img_name not in img_dict:
                img_dict[img_name] = [url, p_box_list]

        print('[Info] 有效文件数: {}'.format(len(img_dict.keys())))

        img_keys = list(img_dict.keys())
        random.shuffle(img_keys)  # shuffle

        n_slice = len(img_keys) // 10

        pool = Pool(processes=80)
        for idx, img_name in enumerate(img_keys):
            # 下载图像
            file_name = img_name.split(".")[0]
            if idx < n_slice * 9:
                out_img_path = os.path.join(out_images_train_dir, file_name+".jpg")
                out_txt_path = os.path.join(out_labels_train_dir, file_name+".txt")
            else:
                out_img_path = os.path.join(out_images_val_dir, file_name+".jpg")
                out_txt_path = os.path.join(out_labels_val_dir, file_name+".txt")

            url, p_box_list = img_dict[img_name]

            # 测试
            DatasetGenerator.generate_item(idx, url, p_box_list, out_img_path, out_txt_path)
            if idx == 20:
                print('[Info] idx: {}'.format(idx))
                break

            # 多进程
            # pool.apply_async(DatasetGenerator.generate_item, (idx, url, p_box_list, out_img_path, out_txt_path))

        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(out_dataset_dir))

    def check_dataset(self):
        dataset_dir = os.path.join(DATA_DIR, 'ps_datasets_v3')
        images_dir = os.path.join(dataset_dir, 'images', 'train')
        labels_dir = os.path.join(dataset_dir, 'labels', 'train')
        out_dir = os.path.join(DATA_DIR, 'ps_checked_{}'.format(get_current_time_str()))
        mkdir_if_not_exist(out_dir)

        paths_list, names_list = traverse_dir_files(images_dir)
        paths_list, names_list = shuffle_two_list(paths_list, names_list )
        paths_list, names_list = paths_list[:20], names_list[:20]
        print('[Info] 检查样本数: {}'.format(len(paths_list)))

        for path, name in zip(paths_list, names_list):
            label_name = name.split('.')[0] + ".txt"
            img_bgr = cv2.imread(path)
            label_path = os.path.join(labels_dir, label_name)
            data_lines = read_file(label_path)
            for idx, data_line in enumerate(data_lines):
                img_bgr = self.check_darknet_data(img_bgr, data_line)

            out_path = os.path.join(out_dir, name)
            cv2.imwrite(out_path, img_bgr)

        print('[Info] 检查完成! {}'.format(out_dir))

    @staticmethod
    def split_prelabeled_data():
        data_file = os.path.join(DATA_DIR, 'prelabeled_2020_12_23.txt')
        data_lines = read_file(data_file)
        print('[Info] 文件数: {}'.format(len(data_lines)))

        img_name_dict = dict()
        for data_line in data_lines:
            img_name = data_line.split("/")[-1]
            img_name_dict[img_name] = data_line

        data_lines = list(img_name_dict.values())
        print('[Info] 清洗文件数: {}'.format(len(data_lines)))

        out_dir = os.path.join(DATA_DIR, 'prelabeled_2020_12_23_split')
        mkdir_if_not_exist(out_dir)

        num = 2000
        out_file_format = os.path.join(out_dir, "cases_{}_{}.txt")
        s_num, e_num = 0, num
        for i, data_line in enumerate(data_lines):
            if s_num <= i < e_num:
                out_file = out_file_format.format(s_num+1, e_num)
            else:
                s_num = e_num
                e_num += num
                e_num = e_num if e_num < len(data_lines) else len(data_lines)
                out_file = out_file_format.format(s_num+1, e_num)
            write_line(out_file, data_line)

        print('[Info] 处理完成!')


def main():
    print('[Info] 处理开始')
    dg = DatasetGenerator()

    # dg.generate()  # 生成数据集
    dg.check_dataset()  # 检查数据集
    # dg.split_prelabeled_data()  # 切分标注数据


if __name__ == "__main__":
    main()
