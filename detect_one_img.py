#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 28.12.20
"""
import copy
import os

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


from myutils.cv_utils import *
from myutils.project_utils import *
from root_dir import DATA_DIR
from x_utils.vpf_utils import get_ocr_vpf_service, get_next_sentences_vpf_service


class ImgDetector(object):
    """
    图像检测
    """
    def __init__(self):
        self.weights = "mydata/models/best_20201224.pt"

        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45

        self.device = select_device()  # 自动选择环境
        self.is_half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model, self.img_size = self.load_model()  # 加载模型

    def filer_boxes_by_size(self, boxes):
        """
        根据是否重叠过滤包含在内部的框
        """
        size_list = []
        idx_list = []
        for idx, box in enumerate(boxes):
            size_list.append(get_box_size(box))
            idx_list.append(idx)

        size_list, idx_list, sorted_boxes = \
            sort_three_list(size_list, idx_list, boxes, reverse=True)

        n_box = len(sorted_boxes)  # box的数量
        flag_list = [True] * n_box

        for i in range(n_box):
            if not flag_list[i]:
                continue
            for j in range(i+1, n_box):
                box1 = sorted_boxes[i]
                box2 = sorted_boxes[j]
                r_iou = min_iou(box1, box2)
                if r_iou > 0.8:
                    flag_list[j] = False

        new_boxes = []
        for i in range(n_box):
            if flag_list[idx_list.index(i)]:
                new_boxes.append(boxes[i])
        return new_boxes

    def load_model(self):
        """
        加载模型
        """
        # Load model
        model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        if self.is_half:
            model.half()  # to FP16
        img_size = check_img_size(self.img_size, s=model.stride.max())  # check img_size

        # 设置Img Half
        img_tmp = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
        _ = model(img_tmp.half() if self.is_half else img_tmp) if self.device.type != 'cpu' else None  # run once

        return model, img_size

    def preprocess_data(self, img_bgr):
        """
        图像预处理
        """
        # Padded resize
        img_bgr = letterbox(img_bgr, new_shape=self.img_size)[0]

        # Convert
        img_rgb = img_bgr[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_rgb = np.ascontiguousarray(img_rgb)

        # 正则化
        img = torch.from_numpy(img_rgb).to(self.device)
        img = img.half() if self.is_half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img

    def detect_problems(self, img_bgr):
        """
        检测逻辑
        """
        img_ori = copy.copy(img_bgr)

        img = self.preprocess_data(img_bgr)  # 预处理数据
        pred = self.model(img, augment=False)[0]  # 预测图像

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)  # NMS后处理

        box_list = []  # 最终输出
        for i, det in enumerate(pred):  # detections per image
            # 回复图像尺寸
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_ori.shape).round()
            det = det.tolist()

            for *xyxy, conf, cls in reversed(det):  # 绘制图像
                xyxy = [int(i) for i in xyxy]
                conf = round(conf, 4)
                cls = int(cls)
                # output_list.append([xyxy, conf, cls])
                box_list.append(xyxy)

        return box_list

    def filter_word_box(self, img_bgr, problem_boxes, word_boxes):
        print('[Info] 题目块: {}, 文字块: {}'.format(len(problem_boxes), len(word_boxes)))
        # draw_box_list(img_bgr, word_boxes)
        n_problem = len(problem_boxes)
        n_word = len(word_boxes)

        flag_list = [True] * n_word

        for i in range(n_problem):
            box1 = problem_boxes[i]
            for j in range(n_word):
                box2 = word_boxes[j]
                r_iou = min_iou(box1, box2)
                if r_iou > 0.8:
                    flag_list[j] = False

        new_boxes = []
        new_idxes = []
        for i in range(n_word):
            if flag_list[i]:
                new_boxes.append(word_boxes[i])
                new_idxes.append(i)

        draw_box_list(img_bgr, new_boxes)
        print('[Info] 输出文字块数: {}'.format(len(new_boxes)))
        return new_boxes, new_idxes

    @staticmethod
    def parse_pos(pos_list):
        """
        处理点
        """
        point_list = []
        for pos in pos_list:
            x = pos['x']
            y = pos['y']
            point_list.append([x, y])
        return point_list

    @staticmethod
    def get_ocr_service_info(url):
        """
        处理URL
        """
        print('[Info] process url: {}'.format(url))
        res_dict = get_ocr_vpf_service(url)
        # print('[Info] res_dict: {}'.format(res_dict))
        data_dict = res_dict['data']['data']
        # print('[Info] data_dict: {}'.format(data_dict))
        word_num = data_dict['wordNum']
        # print('[Info] word_num: {}'.format(word_num))
        words_info = data_dict['wordsInfo']
        angle = data_dict['angle']

        content = data_dict['content']

        # color_list = generate_colors(word_num)
        box_list, word_list, idx_list = [], [], []

        new_words_info = []
        for idx, words_data in enumerate(words_info):
            # print('[Info] words_data: {}'.format(words_data))
            word = words_data["word"]
            rec_classify = words_data["recClassify"]
            # print('[Info] word: {}'.format(word))
            pos = words_data["pos"]
            # print('[Info] pos: {}'.format(pos))
            prob = words_data["prob"]
            # 直接保留公式
            # if rec_classify != 0:
            #     word = u"$${}$$".format(word)
            # if prob == 0:
            #     continue
            word_rec = ImgDetector.parse_pos(pos)
            # print('[Info] point_list: {}'.format(word_rec))
            box_list.append(word_rec)
            word_list.append(word)
            idx_list.append(idx)
            new_words_info.append(words_data)

        return new_words_info, box_list, word_list, idx_list, data_dict

    def analyze_layout(self, l2r_boxes, h, w, is_ud=True):
        """
        分析版面，从左到右的box列
        """
        r2l_boxes = copy.copy(l2r_boxes)
        r2l_boxes.reverse()
        col_box_list = []

        pre_x_min = -1
        for boxes in r2l_boxes:
            x_list, y_list = [], []
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                x_list.append(x_min)
                x_list.append(x_max)
                y_list.append(y_min)
                y_list.append(y_max)
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            if pre_x_min == -1:
                pre_x_min = x_min
            else:
                if x_max > pre_x_min:  # 避免交叉
                    x_max = pre_x_min
            if not is_ud:
                col_box = [x_min, y_min, x_max, y_max]
            else:
                col_box = [x_min, 0, x_max, h]
            print('[Info] col_box: {}'.format(col_box))
            col_box_list.append(col_box)

        col_box_list.reverse()
        return col_box_list

    def filer_boxes_by_col(self, col_boxes, sub_boxes):
        """
        根据列的范围过滤boxes
        """
        col_sub_boxes = []
        col_sub_idxes = []
        for col_box in col_boxes:
            one_col_boxes = []
            one_col_idxes = []
            for idx, word_box in enumerate(sub_boxes):
                v_iou = min_iou(word_box, col_box)
                if v_iou > 0.9:
                    one_col_boxes.append(word_box)
                    one_col_idxes.append(idx)
            col_sub_boxes.append(one_col_boxes)
            col_sub_idxes.append(one_col_idxes)
        return col_sub_boxes, col_sub_idxes

    def merge_problem_by_sentences(self, img_bgr, x_box_list, x_str_list):
        """
        根据上下句服务合并问题
        """
        # draw_box_list(img_bgr, x_box_list, is_show=True)
        print('[Info] str: {}'.format("\n".join(x_str_list)))
        img_h, img_w, _ = img_bgr.shape
        n_box = len(x_box_list)
        r_thr = 0.008
        thr_idx_list = []
        query_list = []
        for i in range(0, n_box-1):
            box1 = x_box_list[i]
            box2 = x_box_list[i+1]
            str1 = x_str_list[i]
            str2 = x_str_list[i+1]
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            diff = y2_min - y1_max
            r_diff = safe_div(diff, img_h)
            print('[Info] diff: {}, r_diff: {}'.format(diff, r_diff))
            if r_diff < r_thr:
                thr_idx_list.append(i)
                query_list.append({"sentence1": "{}".format(str1),
                                   "sentence2": "{}".format(str2)})
        next_flags = [False] * n_box
        if query_list:
            query_list_str = json.dumps(query_list)
            print(query_list_str)
            service_res = get_next_sentences_vpf_service(query_list_str)
            print('[Info] service_res: {}'.format(service_res))
            prob_list = service_res['data']['query_result']

            for x, i in enumerate(thr_idx_list):
                prob = prob_list[x]
                if prob > 0.5:
                    next_flags[i] = True
                    next_flags[i+1] = True

        s_idx, e_idx = -1, -1
        p_idx_list = []
        for idx, n_flag in enumerate(next_flags):
            if n_flag:
                if s_idx == -1:
                    s_idx = idx
                e_idx = idx
            else:
                if s_idx != -1 and e_idx - s_idx > 3:
                    idx_list = [i for i in range(s_idx, e_idx+1)]
                    p_idx_list.append(idx_list)
                s_idx = -1
                e_idx = -1

        print('[Info] next_flags: {}'.format(next_flags))
        print('[Info] p_idx_list: {}'.format(p_idx_list))

        l_box_list = []
        for p_list in p_idx_list:
            box_list = filter_list_by_idxes(x_box_list, p_list)
            l_box = merge_boxes(box_list)
            l_box_list.append(l_box)

        # draw_box_list(img_bgr, l_box_list, is_show=True,
        #               save_name="test.{}.jpg".format(get_current_time_str()))

        return l_box_list

    def add_tag_to_wordinfo(self, img_bgr, col_boxes, problem_boxes, words_info):
        print('[Info] 栏数: {}'.format(len(col_boxes)))
        print('[Info] 块数: {}'.format(len(problem_boxes)))
        print('[Info] 词数: {}'.format(len(words_info)))

        # 初始化
        for words_data in words_info:
            words_data['columnIdx'] = -1
            words_data['blockIdx'] = -1
            words_data['lineIdx'] = -1

        # draw_box_list(img_bgr, col_boxes, is_show=True)  # 测试版面切分效果
        # draw_box_list(img_bgr, problem_boxes, is_show=True)  # 测试题目切分效果

        # ---------- 切分栏 ----------#
        col_idx_dict = collections.defaultdict(list)
        col_wordinfo_dict = collections.defaultdict(list)
        rank_idx_list = []
        for idx, words_data in enumerate(words_info):
            pos = words_data["pos"]
            word_rec = ImgDetector.parse_pos(pos)
            word_box = rec2box(word_rec)
            current_idx = -1
            for col_idx, col_box in enumerate(col_boxes):
                col_iou = min_iou(col_box, word_box)
                if col_iou > 0.5:
                    words_data['columnIdx'] = col_idx
                    current_idx = col_idx
                    c_tag = "c_{}".format(current_idx)
                    col_idx_dict[c_tag].append(idx)
                    col_wordinfo_dict[c_tag].append(words_data)

            if current_idx == -1:
                rank_idx_list.append("{}".format(idx))
            else:
                c_tag = "c_{}".format(current_idx)
                if c_tag not in rank_idx_list:
                    rank_idx_list.append(c_tag)

        print('[Info] rank_idx_list: {}'.format(rank_idx_list))
        print('[Info] col_idx_dict: {}'.format(col_idx_dict))
        for col_key in col_idx_dict.keys():
            print('[Info] {} : {}'.format(col_key, len(col_idx_dict[col_key])))

        # 切分题目
        col_prb_box_dict = collections.defaultdict(list)
        for prb_box in problem_boxes:
            for col_idx, col_box in enumerate(col_boxes):
                col_iou = min_iou(col_box, prb_box)
                if col_iou > 0.5:
                    c_tag = "c_{}".format(col_idx)
                    col_prb_box_dict[c_tag].append(prb_box)

        res_blocks_list = []  # 每栏的block数
        idx_list, num_list = [], []
        for c_tag in col_prb_box_dict.keys():
            idx = int(c_tag.split('_')[-1])
            idx_list.append(idx)
            num_block = len(col_prb_box_dict[c_tag])
            num_list.append(num_block)
        idx_list, res_blocks_list = sort_two_list(idx_list, num_list)
        print('[Info] idx_list: {}, num_list: {}'.format(idx_list, num_list))

        print('[Info] col_prb_box_dict: {}'.format(col_prb_box_dict))
        print('[Info] ' + "-" * 100)
        # 测试效果
        # for col_prb_boxes in col_prb_box_dict.values():
        #     draw_box_list(img_bgr, col_prb_boxes, is_show=True)
        # ---------- 切分栏 ----------#

        # ---------- 切分题目块 ----------#
        col_pro_words_dict = dict()
        for col_key in col_wordinfo_dict.keys():
            col_words_info = col_wordinfo_dict[col_key]
            col_idxes = col_idx_dict[col_key]
            col_prb_boxes = col_prb_box_dict[col_key]

            col_rank_idx_list = []
            prb_wordinfo_dict = collections.defaultdict(list)
            prb_idxes_dict = collections.defaultdict(list)
            for words_data, idx in zip(col_words_info, col_idxes):
                pos = words_data["pos"]
                word_rec = ImgDetector.parse_pos(pos)
                word_box = rec2box(word_rec)
                current_idx = -1

                for prb_idx, prb_box in enumerate(col_prb_boxes):
                    prb_iou = min_iou(prb_box, word_box)
                    if prb_iou > 0.5:
                        words_data['blockIdx'] = prb_idx
                        p_tag = "p_{}".format(prb_idx)
                        prb_idxes_dict[p_tag].append(idx)
                        prb_wordinfo_dict[p_tag].append(words_data)
                        current_idx = prb_idx

                if current_idx == -1:
                    col_rank_idx_list.append("{}".format(idx))
                else:
                    p_tag = "p_{}".format(current_idx)
                    if p_tag not in col_rank_idx_list:
                        col_rank_idx_list.append(p_tag)
            col_pro_words_dict[col_key] = [col_rank_idx_list, prb_wordinfo_dict, prb_idxes_dict]

            print('[Info] col_rank_idx_list: {}'.format(col_rank_idx_list))
            print('[Info] prb_idxes_dict: {}'.format(prb_idxes_dict))
            print('[Info] ' + "-" * 100)
        # ---------- 切分题目块 ----------#

        # ---------- 切分行块 ----------#
        col_idx_num_dict = dict()
        for col_key in col_wordinfo_dict.keys():
            col_rank_idx_list, prb_wordinfo_dict, prb_idxes_dict = col_pro_words_dict[col_key]

            prb_idx_dict = dict()  # 每个问题的ID
            for prb_key in prb_wordinfo_dict.keys():
                prb_words_info = prb_wordinfo_dict[prb_key]
                prb_idxes = prb_idxes_dict[prb_key]

                boxes_list = []
                for words_data, idx in zip(prb_words_info, prb_idxes):
                    pos = words_data["pos"]
                    word_rec = ImgDetector.parse_pos(pos)
                    word_box = rec2box(word_rec)
                    boxes_list.append(word_box)

                sorted_boxes, sorted_idxes, num_row = sorted_boxes_by_row(boxes_list)
                prb_row_idxes = []
                for row_num, row_idxes in enumerate(sorted_idxes):
                    real_row_idxes = []
                    for idx_i in row_idxes:
                        real_idx = prb_idxes[idx_i]
                        real_row_idxes.append(real_idx)
                        words_data = words_info[real_idx]
                        words_data['lineIdx'] = row_num
                    prb_row_idxes.append(real_row_idxes)
                print('[Info] prb_key: {} prb_row_idxes: {}'.format(prb_key, prb_row_idxes))
                prb_idx_dict[prb_key] = prb_row_idxes

                # row_box_list = []
                # for row_boxes in sorted_boxes:
                #     large_row_box = merge_boxes(row_boxes)
                #     row_box_list.append(large_row_box)
                # draw_box_list(img_bgr, row_box_list, is_show=True)
            print('[Info] col_rank_idx_list: {}'.format(col_rank_idx_list))
            print('[Info] prb_idx_dict: {}'.format(prb_idx_dict))

            # 解析list
            col_idx_num_list = []
            for idx_str in col_rank_idx_list:
                if idx_str in prb_idx_dict.keys():
                    col_idx_num_list.append(prb_idx_dict[idx_str])
                else:
                    col_idx_num_list.append([[int(idx_str)]])
            print('[Info] col_idx_num_list: {}'.format(col_idx_num_list))
            col_idx_num_dict[col_key] = col_idx_num_list
        print('[Info] col_idx_num_dict: {}'.format(col_idx_num_dict))
        # ---------- 切分行块 ----------#

        rank_idx_num_list = []
        for idx_str in rank_idx_list:
            if idx_str in col_idx_num_dict.keys():
                rank_idx_num_list.append(col_idx_num_dict[idx_str])
            else:
                rank_idx_num_list.append([[[int(idx_str)]]])
        print('[Info] rank_idx_num_list: {}'.format(rank_idx_num_list))

        print('[Info] ' + "-" * 100)
        all_box_list = []
        for idx, words_data in enumerate(words_info):
            pos = words_data["pos"]
            word_rec = ImgDetector.parse_pos(pos)
            word_box = rec2box(word_rec)
            all_box_list.append(word_box)

        print('[Info] 栏数: {}'.format(len(rank_idx_num_list)))
        all_idx_list = []
        for col_idx, col_list in enumerate(rank_idx_num_list):
            print('[Info] col_idx: {}, col_list: {} - {}'.format(col_idx, len(col_list), col_list))
            # import itertools
            # merged = list(itertools.chain(*col_list))
            # merged = list(itertools.chain(*merged))
            # col_boxes = [all_box_list[i] for i in merged]
            # col_one_box = merge_boxes(col_boxes)
            # draw_box(img_bgr, col_one_box, is_new=False, is_show=False)
            for prb_idx, prb_list in enumerate(col_list):
                print('\t[Info] prb_idx: {}, prb_list: {} - {}'.format(prb_idx, len(prb_list), prb_list))
                # import itertools
                # merged = list(itertools.chain(*prb_list))
                # prb_boxes = [all_box_list[i] for i in merged]
                # prb_one_box = merge_boxes(prb_boxes)
                # draw_box(img_bgr, prb_one_box, color=(0, 255, 0), is_new=False, is_show=False)
                for row_idx, row_list in enumerate(prb_list):
                    print('\t\t[Info] row_idx: {}, row_list: {} - {}'.format(row_idx, len(row_list), row_list))
                    all_idx_list += row_list
                    # row_boxes = [all_box_list[i] for i in row_list]
                    # row_one_box = merge_boxes(row_boxes)
                    # draw_box(img_bgr, row_one_box, color=(255, 0, 0), is_new=False, is_show=False)

        show_img_bgr(img_bgr, save_name="test_nested.jpg")
        print('[Info] all_idx_list: {}'.format(all_idx_list))
        print('[Info] ' + "-" * 100)

        rank_box_list = []
        rank_word_info = []
        for idx in all_idx_list:
            rank_box_list.append(all_box_list[idx])
            rank_word_info.append(words_info[idx])
        draw_box_list(img_bgr, rank_box_list, is_show=True, save_name="test_all.jpg")

        for words_data in rank_word_info:
            pos = words_data["pos"]
            word_rec = ImgDetector.parse_pos(pos)
            word_box = rec2box(word_rec)
            word_tag_str = "{}_{}_{}".format(words_data['columnIdx'], words_data['blockIdx'], words_data['lineIdx'])
            draw_box(img_bgr, word_box, color=(0, 255, 0), is_show=False, is_new=False)
            draw_text(img_bgr, word_tag_str, org=get_box_center(word_box), color=(0, 0, 255))

        show_img_bgr(img_bgr, save_name="test_info.jpg")
        return rank_word_info, len(col_boxes), res_blocks_list,

    def process(self, img_url):
        """
        核心处理逻辑
        """
        # 第1步, 检测题目框
        is_ok, img_bgr = download_url_img(img_url)
        h, w, _ = img_bgr.shape
        problem_boxes = self.detect_problems(img_bgr)
        if not problem_boxes:
            return img_bgr
        # draw_box_list(img_bgr, problem_boxes, is_show=True)

        # 第2步，上下排序
        problem_boxes = self.filer_boxes_by_size(problem_boxes)
        # draw_box_list(img_bgr, problem_boxes, is_show=True)
        # problem_boxes 是 分栏之后的
        col_problem_boxes, col_sorted_idxes, num_col = sorted_boxes_by_col(problem_boxes)
        print('[Info] 页面栏数: {}'.format(num_col))

        problem_boxes = []  # 去掉分栏信息
        for pcb in col_problem_boxes:
            problem_boxes += pcb
        # draw_box_list(img_bgr, problem_boxes, is_show=True, save_name="1.jpg")

        # 第3步，将页面分成多栏
        large_col_boxes = self.analyze_layout(col_problem_boxes, h, w)
        # draw_box_list(img_bgr, col_boxes, is_show=True)

        # 第4步，获取OCR的文本框
        words_info, word_rec_list, word_str_list, idx_list, data_dict \
            = self.get_ocr_service_info(img_url)
        word_boxes = []
        for rec in word_rec_list:
            word_boxes.append(rec2box(rec))
        # draw_box_list(img_bgr, word_boxes, is_show=True)

        # 第5步，题目+文本框 -> 过滤
        other_word_boxes, new_idxes = self.filter_word_box(img_bgr, problem_boxes, word_boxes)
        other_str_list = filter_list_by_idxes(word_str_list, new_idxes)
        # 调试
        # draw_box_list(img_bgr, other_word_boxes, is_show=True, save_name="test.jpg")
        # print('[Info] str: {}'.format("\n".join(other_str_list)))

        col_sub_boxes, col_sub_idxes = self.filer_boxes_by_col(large_col_boxes, other_word_boxes)

        # 过滤文字
        col_str_list = []
        for sub_idxes in col_sub_idxes:
            sub_str_list = filter_list_by_idxes(other_str_list, sub_idxes)
            col_str_list.append(sub_str_list)

        # 第6步, 处理每一栏的信息
        problem_box_list = []
        for col_boxes, col_str in zip(col_sub_boxes, col_str_list):
            # draw_box_list(img_bgr, col_boxes, is_show=True)
            sorted_boxes, col_sorted_idxes, num_col = sorted_boxes_by_row(col_boxes)
            sorted_str = filter_list_by_idxes(col_str, col_sorted_idxes)
            x_box_list = []
            x_str_list = []
            for boxes, word_strings in zip(sorted_boxes, sorted_str):
                x_box = merge_boxes(boxes)
                x_str = "".join(word_strings)
                x_box_list.append(x_box)
                x_str_list.append(x_str)

            l_box_list = self.merge_problem_by_sentences(img_bgr, x_box_list, x_str_list)
            problem_box_list += l_box_list

        # 第7步，全部排序
        all_problem_boxes = problem_box_list + problem_boxes
        col_problem_boxes, col_sorted_idxes, num_col = sorted_boxes_by_col(all_problem_boxes)
        problem_boxes = []  # 去掉分栏信息
        for pcb in col_problem_boxes:
            problem_boxes += pcb
        img_out = draw_box_list(img_bgr, problem_boxes)

        rank_word_info, column_num, block_num_list = \
            self.add_tag_to_wordinfo(img_bgr, large_col_boxes, problem_boxes, words_info)
        data_dict["wordsInfo"] = rank_word_info
        data_dict["columnNum"] = column_num
        data_dict["blockNumList"] = block_num_list

        res_dict = {
            "url": img_url,
            "data_dict": data_dict
        }

        out_path = os.path.join(DATA_DIR, 'sample.txt')
        res_dict_str = json.dumps(res_dict)
        write_line(out_path, res_dict_str)
        print('[Info] 处理完成: {}'.format(out_path))
        return img_out


def evaluate_urls():
    urls_file = os.path.join(DATA_DIR, 'test_400_res.right.e0.csv')
    data_lines = read_file(urls_file)

    out_dir = os.path.join(DATA_DIR, 'cases-400')
    mkdir_if_not_exist(out_dir)
    _, img_names = traverse_dir_files(out_dir)

    ido = ImgDetector()
    for idx, data_line in enumerate(data_lines):
        if idx == 0:
            continue
        items = data_line.split(',')
        url = items[0]
        img_name = url.split('/')[-1]
        if img_name in img_names:
            print('[Info] 已经处理!')
            continue
        url = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/tmp/test_400_out/{}".format(img_name)
        print('[Info] url: {}'.format(url))
        img_out = ido.process(url)
        out_path = os.path.join(out_dir, img_name)
        cv2.imwrite(out_path, img_out)
        print('[Info] 处理完成: {}'.format(out_path))


def evaluate_case():
    from root_dir import DATA_DIR
    # img_path = os.path.join(DATA_DIR, 'labeled_mini', '9f4c773acbfb2f58cf4e0d7eb2099fa6.jpg')
    # img_path = os.path.join(DATA_DIR, 'test', 'fcf3445e2f6b1cfce533ba272818a24f.jpg')
    # img_bgr = cv2.imread(img_path)
    # show_img_bgr(img_bgr)
    img_url = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/tmp/test1.jpg"
    # img_url = "https://img.alicdn.com/imgextra/i3/6000000000052/O1CN01EvpIvD1CFrEQZjHyI_!!6000000000052-0-quark.jpg"
    # img_url = "https://img.alicdn.com/imgextra/i1/6000000005190/O1CN013t3IMJ1oD4DVlTj7b_!!6000000005190-0-quark.jpg"
    # img_url = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/tmp/test_400_out/O1CN01KseCRE1FsjhJD7ivM_!!6000000000543-0-quark.jpg"
    ido = ImgDetector()
    # ido.detect_with_draw(img_bgr)
    img_out = ido.process(img_url)


def main():
    # evaluate_urls()
    evaluate_case()


if __name__ == '__main__':
    main()
