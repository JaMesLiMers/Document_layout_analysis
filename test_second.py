import os
import sys
import time
import cv2

import torch
import torch.nn as nn
import numpy as np
from backbone_second import DecoderNet, EncoderNet
from Dataset_second import ModelDataset
from model_load import remove_prefix

# init train parameter
SAVE_PATH = 'save'
SAVE_NAME = 'model_second_epoch_369.pkl'
TARGET_IMAGE_PATH = 'test'

# cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# 准备数据集
train_set = ModelDataset(base_path='./test', train=False)

# 建立dataloader
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=1,
                 num_workers=1,
                 shuffle=False)

# 加载模型
model = nn.Sequential(EncoderNet(),
                     DecoderNet(),)
model.eval()

def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   # reshape 为向量
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)

def save_image(img, num, dir):
    img = np.uint8(img) * 255
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) * 255
    img = cv2.resize(img, (400, 600))
    cv2.imwrite(os.path.join(dir, 'result_second_{}.png'.format(num)), img)

if SAVE_NAME:
    try:
        state_dict = torch.load(os.path.join('.', 'save', SAVE_NAME), map_location=torch.device('cpu'))
        state_dict = remove_prefix(state_dict, 'module.')
        model.load_state_dict(state_dict)
    except Exception as e:
        print(e)

for num, i in enumerate(train_set):
    x = torch.Tensor(i['target_image'][np.newaxis, :, :, :])
    output = model(x)

    _, pred = torch.topk(output, 1, dim=1)

    # output = torch.squeeze(output).permute(1, 2, 0).detach().numpy()
    output_onehot = get_one_hot(pred.squeeze(), 4)
    output_onehot = output_onehot[:,:,1::].permute(1,0,2).detach().numpy()

    save_image(output_onehot, num, dir=os.path.join('.', TARGET_IMAGE_PATH, 'Output'))
