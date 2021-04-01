#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 8.3.21
"""

import os
import sys
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from x_utils.vpf_utils import get_ps_service_v2_0, get_ps_service_v2_1, get_ocr_vpf_service, get_ps_service_v3_0
from myutils.project_utils import *
from root_dir import DATA_DIR


class OnlineEvaluation(object):
    def __init__(self):
        self.check_dir = "check_20210317"
        # self.file_name = "图片弯曲"
        # self.file_name = "切块问题"
        self.file_name = "random_1w_urls"
        pass

    @staticmethod
    def make_html_page(html_file, imgs_path, n=4):
        header = """
        <!DOCTYPE html>
        <html>
        <head>
        <title>MathJax TeX Test Page</title>
        <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
        <script type="text/javascript" id="MathJax-script" async
          src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
        </script>
        <style>
        img{
            max-height:640px;
            max-width: 640px;
            vertical-align:middle;
        }
        </style>
        </head>
        <body>
        <table>
        """

        tail = """
        </table>
        </body>
        </html>
        """

        data_lines = read_file(imgs_path)
        print('[Info] 样本行数: {}'.format(len(data_lines)))
        urls_list = []  # url列表
        for data_line in data_lines:
            urls = data_line.split("<sep>")
            urls_list.append(urls)

        with open(html_file, 'w') as f:
            f.write(header)
            for idx, items in enumerate(urls_list):
                f.write('<tr>\n')
                f.write('<td>%d</td>\n' % ((idx / n) + 1))
                for item in items:
                    f.write('<td>\n')
                    if item.startswith("http"):
                        f.write('<img src="%s" width="600">\n' % item)
                    else:
                        f.write('%s' % item)
                    f.write('</td>\n')
                f.write('</tr>\n')
            f.write(tail)

    @staticmethod
    def call_service_url(url, mode):
        """
        处理URL
        """
        oss_url_out1, oss_url_out2, oss_url_out3, oss_url_out4, content = "", "", "", "", ""
        if mode == "v2.0":
            res_data = get_ocr_vpf_service(url)
            data_dict = res_data["data"]
            content = data_dict["data"]["content"]
            oss_url_out1 = data_dict['oss_url_reorder']
            oss_url_out2 = data_dict['oss_url_column']
            oss_url_out3 = data_dict['oss_url_block']
            oss_url_out4 = data_dict['oss_url_preorder']
        elif mode == "v2.1":
            res_data = get_ps_service_v2_1(url)
            data_dict = res_data["data"]
            content = data_dict["data"]["content"]
            oss_url_out1 = data_dict['oss_url_out1']
            oss_url_out2 = data_dict['oss_url_out2']
            oss_url_out3 = data_dict['oss_url_out3']
            oss_url_out4 = data_dict['oss_url_out4']
        elif mode == "v3.0":
            res_data = get_ps_service_v3_0(url)
            data_dict = res_data["data"]
            content = data_dict["data"]["content"]
            oss_url_out1 = data_dict['oss_url_out1']
            oss_url_out2 = data_dict['oss_url_out2']
            oss_url_out3 = data_dict['oss_url_out3']
            oss_url_out4 = data_dict['oss_url_out4']
        return oss_url_out1, oss_url_out2, oss_url_out3, oss_url_out4, content

    @staticmethod
    def process_thread(idx, url, out_file):
        """
        线程处理
        """
        # try:
        oss_x_out1, oss_x_out2, oss_x_out3, oss_x_out4, content_x \
            = OnlineEvaluation.call_service_url(url, mode="v2.1")
        oss_y_out1, oss_y_out2, oss_y_out3, oss_y_out4, content_y \
            = OnlineEvaluation.call_service_url(url, mode="v3.0")
        # print('[Info] url_v20: {}, url_v21: {}'.format(url_v20, url_v21))
        out_line_1 = "<sep>".join([url, oss_x_out1, oss_y_out1])
        out_line_2 = "<sep>".join([url, oss_x_out2, oss_y_out2])
        out_line_3 = "<sep>".join([url, oss_x_out3, oss_y_out3])
        out_line_4 = "<sep>".join([url, oss_x_out4, oss_y_out4])
        # content = "<sep>".join([url, content_x, content_y])
        write_line(out_file, out_line_1)
        write_line(out_file, out_line_2)
        write_line(out_file, out_line_3)
        write_line(out_file, out_line_4)
        # write_line(out_file, content)
        print('[Info] idx: {}, url: {}'.format(idx, url))
        # except Exception as e:
        #     print('[Exception] e: {}, url: {}'.format(e, url))

    def process(self):
        """
        处理
        """
        file_name = self.file_name
        file_path = os.path.join(DATA_DIR, 'test_urls_files', "{}.txt".format(file_name))
        print('[Info] 评估开始! {}'.format(file_path))

        out_dir = os.path.join(DATA_DIR, self.check_dir)
        mkdir_if_not_exist(out_dir)
        print('[Info] 输出文件夹: {}'.format(out_dir))

        out_file = os.path.join(out_dir, '{}_out.csv'.format(file_name))
        create_file(out_file)
        out_urls_file = os.path.join(out_dir, '{}_urls.txt'.format(file_name))
        create_file(out_urls_file)

        data_lines = read_file(file_path)
        print('[Info] 文件行数: {}'.format(len(data_lines)))

        n = 100
        if len(data_lines) > n:
            random.seed(65)
            random.shuffle(data_lines)  # 随机生成
            data_lines = data_lines[:n]
        print('[Info] 处理行数: {}'.format(len(data_lines)))

        pool = Pool(processes=80)
        for idx, data_line in enumerate(data_lines):
            url = data_line.split('?')[0]
            write_line(out_urls_file, url)
            OnlineEvaluation.process_thread(idx, url, out_file)  # 单进程
            # pool.apply_async(OnlineEvaluation.process_thread, (idx, url, out_file))  # 多进程

        pool.close()
        pool.join()

    def process_url(self):
        data_file = os.path.join(DATA_DIR, self.check_dir, '{}_out.csv'.format(self.file_name))
        html_file = os.path.join(DATA_DIR, self.check_dir, '{}_out.html'.format(self.file_name))
        create_file(html_file)
        OnlineEvaluation.make_html_page(html_file, data_file)  # 生成URL
        print('[Info] 处理完成: {}'.format(html_file))


def main():
    oe = OnlineEvaluation()
    # oe.process()
    oe.process_url()


if __name__ == '__main__':
    main()
