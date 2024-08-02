"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_txt_file(filename):
    return filename.endswith('txt')


def is_exr_file(filename):
    return filename.endswith('exr')


def is_jso_file(filename):
    return filename.endswith('json')


def make_dataset(dir, txt=False, exr=False, jso=False):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if exr is True and is_exr_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
            elif txt is True and is_txt_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
            elif jso is True and is_jso_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
            elif is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images
