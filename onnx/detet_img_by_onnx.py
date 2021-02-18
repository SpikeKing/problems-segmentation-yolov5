#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 21.1.21
"""

from PIL.Image import Image
import os
import sys


class ImgDetectorONNX(object):
    def __init__(self):
        pass

    @staticmethod
    def letterbox_image(image, size):
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image



def main():
    pass


if __name__ == '__main__':
    main()
