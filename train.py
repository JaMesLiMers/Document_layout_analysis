import os
import sys
import time

import torch
import torch.nn as nn
from backbone import DecoderNet, EncoderNet
from Dataset import ModelDataset

import argparse
from model_load import load_pretrain

import logging
from train_utils.log_helper import init_log, add_file_handler, print_speed
from train_utils.average_meter_helper import AverageMeter

# init train parameter
NUM_WORKER = 16
BATCH_SIZE = 8
EPOCHE = 10000
LR = 0.001
PRINT_FREQ = 20
BOARD_PATH = 'board'
SAVE_PATH = 'save'
PRETRAIN = 'alexnet_bn_.pth'
FREEZE = True
START_BACKBONE = 40

# 初始化logger
global_logger = init_log('global', level=logging.INFO)
add_file_handler("global", os.path.join('.', 'train.log'), level=logging.DEBUG)

# 初始化avrager
avg = AverageMeter()

# cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# 准备数据集
train_set = ModelDataset(base_path='./Data', train=True)

# 建立dataloader
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKER,
                 shuffle=True)

# 确定数据集长度
train_lenth = len(train_loader)

global_logger.debug('==>>> total trainning batch number: {}'.format(train_lenth))

def freeze(target_module, train=False):
    for child in target_module.children():
        for param in child.parameters():
            param.requires_grad = train

# 加载模型
model = nn.Sequential(EncoderNet(),
                     DecoderNet(),)

# try:
#     model.load_state_dict(torch.load(os.path.join('.', 'save', PRETRAIN)))
# except Exception as e:
#     print(e)
load_pretrain(model, os.path.join('.', 'save', PRETRAIN))

# freeze backbone
if FREEZE:
    freeze(model._modules['0'], False)

model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).to(device)

# 建立tensorboard的实例
from tensorboardX import SummaryWriter 
writer = SummaryWriter(os.path.join(".", BOARD_PATH))

# 建立优化器
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[5,10,15,20,30], gamma=0.5)

# 建立loss
loss_L1 = nn.MSELoss().to(device)
loss_L2 = nn.L1Loss().to(device)
# crossEntropy
loss_BCE = nn.BCELoss().to(device)

# 训练的部分
for epoch in range(EPOCHE):

    if epoch is START_BACKBONE:
        # freeze backbone
        if FREEZE:
            freeze(model.module._modules["0"], True)

    for step, teget_dic in enumerate(train_loader):
        target_image = teget_dic['target_image']
        target_mask = teget_dic['target_mask']
        step_time = time.time()
        # 打印测试信息
        if epoch is 0 and step is 0:
            global_logger.debug('Input:  {}'.format(target_image.shape))
            global_logger.debug('--- Sample')
            global_logger.debug('Target: {}'.format(target_mask.shape))

        # 放到cuda中
        target_image, target_mask = target_image.to(device), target_mask.to(device)

        # 优化器归零
        optimizer.zero_grad()

        # 送入模型进行推断
        layer_output = model(target_image)

        # loss计算
        train_loss = loss_BCE(layer_output, target_mask)
        with torch.no_grad():
            train_metric = loss_L1(layer_output, target_mask)
        train_loss.backward()

        # 优化器更新
        optimizer.step()

        # 时间
        step_time = time.time() - step_time

        # 将有用的信息存进tensorboard中
        writer.add_scalars('loss/merge', {"train_loss": train_loss, "train_metric":train_metric}, epoch*train_lenth + step + 1)

        # 更新avrager
        avg.update(step_time=step_time, train_loss=train_loss, train_metric=train_metric) # 算平均值

        # 打印结果
        if (step+1) % PRINT_FREQ == 0:
                global_logger.info('Epoch: [{0}][{1}/{2}] {step_time:s}\t{train_loss:s}\t{train_metric:s}'.format(
                            epoch+1, (step + 1) % train_lenth, train_lenth, step_time=avg.step_time, train_loss=avg.train_loss, train_metric=avg.train_metric))
                print_speed(epoch*train_lenth + step + 1, avg.step_time.avg, EPOCHE * train_lenth)

    # save model
    torch.save(model.state_dict(), os.path.join('.', SAVE_PATH, 'model_epoch_{}.pkl'.format(epoch)))

    # scheduler更新
    scheduler.step()

