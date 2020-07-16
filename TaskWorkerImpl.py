# -*- coding:utf-8 -*-
# @Time : 2020-2-28 13:54
# @Author: Henry.ZHAO
# @File : TaskWorkerImpl.py
import torch
from PIL import Image
import numpy as np


import cv2
import base64
import re
from io import BytesIO
from torchvision import transforms
from . import model

#入参，两张图，一个alpha,一个输出图片的h,w
class TaskWorkerImpler():
    def __init__(self, taskid):
        self.canTrain = False  # 是否可以训练
        self.canPredict = False  # 是否可以预测
        self.canSDk = False  # 是否支持sdk
        self.model = None
        self.device = None
        self.taskid = taskid
        self.gpucost = 2048  # 显存消耗以Mb为单位，根据任务自行修改
        self.cpucost = 2048  # 内存消耗以Mb为单位，根据任务自行修改

    # 需要你定制函数
    def initImpler(self, initmodelparams):
        # torch.cuda.manual_seed(1)
        # torch.backends.cudnn.deterministic = True
        # 初始化你的模型
        # 绑定你的模型到指定的设备
        # 加载你的模型权重
        # 有用到initmodelparams 记得做合法性判断
        model_state_path = initmodelparams["face_generate_weight"]
        model.g_all.load_state_dict(torch.load(model_state_path))
        model.g_all.eval()
        self.model = model.g_all.to(self.device)
        return 1

    # 默认函数不需要修改
    def setDevice(self, device):
        self.device = device
        return 1

    # 默认函数不需要修改
    def getWorkerCost(self):
        taskinfo = dict()
        taskinfo['gpucost'] = self.gpucost
        taskinfo['cpucost'] = self.cpucost
        return taskinfo

    # 默认函数不需要修改,图像转base64
    # datatype默认表示img是PIL的图像，1的时候表示img为opencv mat的格式
    def image_to_base64(self, img, datatype):
        if datatype == 1:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        output_buffer = BytesIO()
        img.save(output_buffer, format='png')
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data)
        return base64_str.decode()

    # 默认函数不需要修改，base64转图片
    # datatype默认表示是返回PIL的图像，1的时候表示返回opencv mat的格式
    def base64_to_image(self, base64_str, datatype):
        base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
        byte_data = base64.b64decode(base64_data)
        img = None
        if datatype == 1:
            image_data = np.fromstring(byte_data, np.uint8)
            img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        else:
            image_data = BytesIO(byte_data)
            img = Image.open(image_data)
        return img

        # 需要你定制函数


    # base64datalist为图片base64数据列表根据你自己的需要转成你的PIL或者转成 cv2.mat
    def predictPic(self, ):

        latents = torch.randn(1, 512, device=self.device)
        with torch.no_grad():
            imgs = self.model(latents)
            imgs = (imgs.clamp(-1, 1) + 1) / 2.0  # normalization to 0..1 range

        imgs = imgs.cpu().squeeze()
        image = imgs.squeeze(0)
        image = transforms.ToPILImage()(image)  # reconvert into PIL image image
        #image.save("result.jpg")
        output_pic1 = self.image_to_base64(image, 0)
        myresponse = []
        output_pic = dict()
        output_pic['outimg'] = output_pic1
        myresponse.append(output_pic)
        return myresponse

