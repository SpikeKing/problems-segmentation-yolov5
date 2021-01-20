#coding=utf-8
import os
import sys
import cv2
import requests
from glob import glob
from tqdm import tqdm
from urllib.parse import quote
from pylatexenc.latexwalker import LatexWalker, LatexMacroNode, LatexCharsNode, LatexMathNode, LatexSpecialsNode, LatexGroupNode

from myutils.project_utils import read_file
from root_dir import DATA_DIR


def read_lines(filepath):
    with open(filepath) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]
        return lines


def get_image_paths(image_dir, post_fixes = ['.jpg', '.png', '.jpeg', '.bmp', '.webp', '.tif']):
    img_paths = []
    for filename in os.listdir(image_dir):
        filepath = os.path.join(image_dir, filename)
        for post_fix in post_fixes:
            if filepath.lower().endswith(post_fix):
                img_paths.append(filepath)
        if os.path.isdir(filepath):
            img_paths.extend(get_image_paths(filepath, post_fixes))
    return img_paths


if __name__ == '__main__':
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
    tail="""
</table>
</body>
</html>
"""
    # image_dir = sys.argv[1]
    # image_paths = get_image_paths(image_dir)
    # html_file = sys.argv[2]
    # image_paths = ["https://img.alicdn.com/imgextra/i2/6000000003335/O1CN01aNTJ931aVTVuytHXc_!!6000000003335-0-quark.jpg",
    #                "https://img.alicdn.com/imgextra/i2/6000000003335/O1CN01aNTJ931aVTVuytHXc_!!6000000003335-0-quark.jpg"]
    # image_paths2 = ["https://img.alicdn.com/imgextra/i2/6000000003335/O1CN01aNTJ931aVTVuytHXc_!!6000000003335-0-quark.jpg",
    #                "https://img.alicdn.com/imgextra/i2/6000000003335/O1CN01aNTJ931aVTVuytHXc_!!6000000003335-0-quark.jpg"]

    image_file1 = os.path.join(DATA_DIR, "long_text_2020-12-07-15-21-36_家德.out1.txt")
    image_file2 = os.path.join(DATA_DIR, "long_text_2020-12-07-15-21-36_家德.out2.txt")
    image_paths = read_file(image_file1)
    image_paths2 = read_file(image_file2)
    html_file = "xxx.html"
    with open(html_file, 'w') as f:
        f.write(header)
        for index, (image_path, image_path2) in enumerate(zip(image_paths, image_paths2)):
            f.write('<tr>\n')
            f.write('<td>%d</td>\n'%(index + 1))
            f.write('<td>\n')
            f.write('<img src="%s" width="400">\t'%(image_path))
            f.write('<img src="%s" width="400">\n'%(image_path2))
            f.write('</td>\n')
            f.write('</tr>\n')
        f.write(tail)
