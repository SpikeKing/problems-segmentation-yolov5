#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 12.1.21
"""

import os
import cv2

from multiprocessing.pool import Pool
from myutils.project_utils import *
from root_dir import DATA_DIR
from myutils.cv_utils import *
from x_utils.vpf_utils import get_problem_segmentation_cv_vpf_service


class SampleProcessor(object):
    def __init__(self):
        self.input_file1 = os.path.join(DATA_DIR, 'page_dataset_raw',
                                        'sanghu.zj_question_cut_sampled_kousuanpigai_url_3w_0104')
        self.input_file2 = os.path.join(DATA_DIR, 'page_dataset_raw',
                                        'sanghu.zj_question_cut_sampled_jueying_url_5k_1229')
        print('[Info] input_file1: {}'.format(self.input_file1))
        print('[Info] input_file2: {}'.format(self.input_file2))
        self.out_dir = os.path.join(DATA_DIR, 'full_page_imgs_1024')
        mkdir_if_not_exist(self.out_dir)

    @staticmethod
    def process_img_url(idx, data_line, names_list, out_dir):
        print('[Info] idx: {}'.format(idx))
        img_url = data_line.split("?")[0]
        img_name = img_url.split('/')[-1]
        if img_name in names_list:
            print('[Info] 已处理: {}'.format(img_name))
            return
        print('[Info] img_url: {}, img_name: {}'.format(img_url, img_name))

        # res_dict = get_problem_segmentation_vpf_service(img_url)
        # oss_url_out1 = res_dict['data']['oss_url_out2']

        # res_dict = get_problem_segmentation_cv_vpf_service(img_url)
        # oss_url = res_dict['data']['oss_url']

        print('[Info] img_url: {}'.format(img_url))
        is_ok, img_bgr = download_url_img(img_url)
        out_path = os.path.join(out_dir, img_name)

        img_bgr = resize_img_fixed(img_bgr, 1024, is_height=True)

        cv2.imwrite(out_path, img_bgr)
        print('[Info] 处理完成 out_path: {}'.format(out_path))

    def process(self):
        data_lines1 = read_file(self.input_file1)
        data_lines2 = read_file(self.input_file2)
        data_lines = data_lines1 + data_lines2
        random.seed(47)
        random.shuffle(data_lines)

        # data_lines = data_lines[0:10]
        print('[Info] 样本数: {}'.format(len(data_lines)))

        paths_list, names_list = traverse_dir_files(self.out_dir)  # 用于避免重复生产
        p = Pool(processes=40)
        for idx, data_line in enumerate(data_lines):
            # SampleProcessor.process_img_url(idx, data_line, names_list, self.out_dir)
            p.apply_async(SampleProcessor.process_img_url, (idx, data_line, names_list, self.out_dir))
            print('[Info] ' + '-' * 50)

        p.close()
        p.join()
        print('[Info] 处理完成: {}'.format(self.input_file1))
        print('[Info] 处理完成: {}'.format(self.input_file2))


def main():
    sp = SampleProcessor()
    sp.process()


if __name__ == '__main__':
    main()
