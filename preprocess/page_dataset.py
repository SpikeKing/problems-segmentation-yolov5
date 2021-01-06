#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 5.1.21
"""

import os
import sys
import random

from myutils.project_utils import *
from root_dir import DATA_DIR


class PageDataset(object):
    def __init__(self):
        self.data_raw_dir = os.path.join(DATA_DIR, 'page_dataset_raw')
        self.data_mini_file = os.path.join(DATA_DIR, 'page_dataset_mini.txt')

    def process(self):
        print('[Info] 待处理文件夹: {}'.format(self.data_raw_dir))
        paths_list, names_list = traverse_dir_files(self.data_raw_dir)
        print('[Info] 数据行数: {}'.format(len(paths_list)))

        data_lines = []
        for path, name in zip(paths_list, names_list):
            sub_lines = read_file(path)
            data_lines += sub_lines

        print('[Info] 文件数: {}'.format(len(data_lines)))

        random.seed(47)
        random.shuffle(data_lines)

        mini_lines = data_lines[:200]
        print('[Info] 小样本数: {}'.format(len(mini_lines)))
        for data_line in mini_lines:
            write_line(self.data_mini_file, data_line)

        print('[Info] 写入文件完成! {}'.format(self.data_mini_file))


def main():
    pd = PageDataset()
    pd.process()


if __name__ == '__main__':
    main()
