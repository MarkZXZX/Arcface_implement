# -*- coding: utf-8 -*-
"""
Created on 20-3-17 下午8:55

@author: markzhou
"""
from __future__ import print_function
import os
import cv2
from models import *
import torch
import numpy as np
import time
from config import Config
from torch.nn import DataParallel


def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])
    return data_list


def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def get_featurs(model, test_list, batch_size=10):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0) # 拼接函数，把image全部连上，images.shape[0]就是图片的数量

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(torch.device("cpu"))
            output = model(data)
            output = output.data.cpu().numpy()
            fe_1 = output[::2] #beg:end:step 
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2)) # 它其实就是水平(按列顺序)把数组给堆叠起来，vstack()函数正好和它相反

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt
def now_get_featurs(model, img):
    data = torch.from_numpy(img)
    data = data.to(torch.device("cpu"))
    output = model(data)
    output = output.data.cpu().numpy()
    fe_1 = output[::2] #beg:end:step 
    fe_2 = output[1::2]
    feature = np.hstack((fe_1, fe_2))
    return feature


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[i] = each
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_feature(model, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    features, cnt = get_featurs(model, img_paths, batch_size=batch_size)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    return features,fe_dict


def prepare():
    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    state_dict = torch.load(opt.test_model_path,map_location='cpu')
    # print(len(state_dict))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    model.eval()
    features,fe_dict = lfw_feature(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
    return model,features,fe_dict

def process(model,features,fe_dict,img1):
    feature= now_get_featurs(model,img1)
    l = []
    for i in range(len(features)):
        sim = cosin_metric(feature, features[i])
        l.append(sim)

    name = 'unknow'
    if max(l)>0.4:
        # print(max(l))
        index = l.index(max(l))
        name = fe_dict[index]
    return name,max(l)

    

if __name__ == '__main__':
    print("process all pic, begin!")
    model,features,fe_dict = prepare()
    print("Well down")
    show_FPS = 1
    cap = cv2.VideoCapture(0)
    count = 0
    s = time.time()
    while(True):
        if time.time()-s>show_FPS:
            print('PFS'+str(count/show_FPS))
            s = time.time()
            count = 0
        count+=1
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        xmlfile ='haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(xmlfile)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(5, 5),
        )
        image_list = []
        # print("发现{0}个目标!".format(len(faces)))
        for (x,y,w,h) in faces:
            img = frame[y:y+h,x:x+w,:]
            img = cv2.resize(img, (128,128))
            # cv2.imwrite('./pic/'+str(count)+'.jpg',img)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image = np.dstack((image, np.fliplr(image)))
            image = image.transpose((2, 0, 1))
            image = image[:, np.newaxis, :, :]
            image = image.astype(np.float32, copy=False)
            image -= 127.5
            image /= 127.5
            name,possible = process(model,features,fe_dict,image)
            cv2.putText(frame, str(name)+'  '+str(possible)[1:6], (x,y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL,1.2, (255, 255, 0) )
            cv2.rectangle(frame, (x, y), (x + w, y + w), (0, 255, 0), 2)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()






