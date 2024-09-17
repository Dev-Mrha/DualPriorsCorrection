# -*- coding: utf-8 -*-
import urllib.request
import urllib.error
import time
import json
import cv2
import os
from tqdm import tqdm

http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
key = "jQHy36U4TEUYdOD4UwnDHM2mJXIJRzEE"
secret = "LIQb5aMzS_FBhDfy1OwJ7xJwKeKQGAPh"
# filepath = r"D:\\HIT\\correcttion\\datasets\\test\\0001_n21_pred.jpg"
import requests.packages.urllib3.util.ssl_

requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS = 'ALL'
def get_landmarks(img_path):
    boundary = '----------%s' % hex(int(time.time() * 1000))
    data = []
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
    data.append(key)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
    data.append(secret)
    data.append('--%s' % boundary)
    fr = open(img_path, 'rb')
    data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file')
    data.append('Content-Type: %s\r\n' % 'application/octet-stream')
    data.append(fr.read())
    fr.close()
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_landmark')
    data.append('1')
    # data.append('--%s' % boundary)
    # data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
    # data.append(
    #     "gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus")
    data.append('--%s--\r\n' % boundary)

    for i, d in enumerate(data):
        if isinstance(d, str):
            data[i] = d.encode('utf-8')

    http_body = b'\r\n'.join(data)

    # build http request
    req = urllib.request.Request(url=http_url, data=http_body)
    # header
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)

    try:
        resp = urllib.request.urlopen(req, timeout=5)
        qrcont = resp.read()
        output = json.loads(qrcont.decode('utf-8'))
        ldmks = []
        for i in range(int(output['face_num'])):
            del output['faces'][i]['landmark']['left_eye_pupil']
            del output['faces'][i]['landmark']['right_eye_pupil']
            ldmks.append(output['faces'][i]['landmark']) # .pop().pop('right_eye_pupil')
        return ldmks
    except urllib.error.HTTPError as e:
        print(e.read().decode('utf-8'), " ", img_path)
        # print(img_path + "too large")
        return None


def draw_landmark(img_path, ldmks):
    img = cv2.imread(img_path)
    for face in ldmks:
        for ldmk in face.keys():
            cv2.circle(img, tuple([face[ldmk]['x'], face[ldmk]['y']]), 5, (0, 0, 255), -5)
    cv2.imwrite(img_path.replace(".jpg", "_ldmk.jpg"), img)

def save_ldmk(save_pth, ldmks):
    ret = []
    for face in ldmks:
        f = []
        for ldmk in face:
            f.append([face[ldmk]['x'], face[ldmk]['y']])
        ret.append(f)
    with open(save_pth, "w") as f:
        json.dump(ret, f)

def generate_pred_ldmk(imgs_path):
    imgs = os.listdir(imgs_path)
    imgs = [i for i in imgs if "pred" not in i and "landmark" not in i and "mask" not in i]
    imgs = [i for i in imgs if "78" in i]
    # imgs = [i for i in imgs if "pred_mask" in i and "ldmk" not in i and "jpg" in i]
    # imgs = [i for i in imgs if "output" in i and "ldmk" not in i]
    print(len(imgs))
    for img_path in tqdm(imgs):
        ori_img = cv2.imread(imgs_path + img_path)
        new_img = cv2.resize(ori_img, (0,0), fx=0.75, fy=0.75)
        cv2.imwrite(imgs_path + img_path.replace(".jpg", "_small.jpg"), new_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        ldmks = get_landmarks(imgs_path + img_path.replace(".jpg", "_small.jpg"))
        if ldmks is not None:
            save_ldmk(imgs_path + img_path.replace(".jpg", "_landmark.json"), ldmks)
            # draw_landmark(imgs_path+img_path, ldmks)

def see_ldmk(imgs_path):
    imgs = os.listdir(imgs_path)
    imgs = [i for i in imgs if "pred" in i and "ldmk" not in i and "mask" not in i]
    for img_path in tqdm(imgs):
        img = cv2.imread(imgs_path + img_path)
        json_file = open(imgs_path + img_path.replace(".jpg", "_ldmk.json"), "r")
        ldmks = json.load(json_file)
        for face in ldmks:
            for i, ldmk in enumerate(face):
                cv2.circle(img, tuple([int(ldmk[0]), int(ldmk[1])]), 5, (0, 0, 255), -5)
        cv2.imwrite(imgs_path + img_path.replace(".jpg", "_ldmk.jpg"), img)

        # img = cv2.imread(imgs_path + img_path.replace("_pred_mask.jpg", "_stereo.jpg"))
        # json_file = open(imgs_path + img_path.replace("_pred_mask.jpg", "_stereo_landmark.json"), "r")
        # ldmks = json.load(json_file)
        # for i, ldmk in enumerate(ldmks[0]):
        #     cv2.circle(img, tuple([int(ldmk[0]), int(ldmk[1])]), 5, (0, 0, 255), -5)
        # cv2.imwrite(imgs_path + img_path.replace("_pred_mask.jpg", "_stereo_ldmk.jpg"), img)


if __name__ == '__main__':
    imgs_path = "../test/"
    generate_pred_ldmk(imgs_path)
    # see_ldmk(imgs_path)
