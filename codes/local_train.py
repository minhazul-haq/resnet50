#code courtesy: https://www.kaggle.com/gxkok21/resnet50-with-pytorch
#author: Mohammad Minhazul Haq
#created on: February 3, 2020

import torch
import torchvision
import os
import os.path as osp
import torch.backends.cudnn as cudnn
import pickle
import numpy as np
import torch.nn as nn
from torch.utils import data
from dataset_loader import WSI_Dataset
from local_utils import transform_pipe_train, transform_pipe_val_test
from tensorboardX import SummaryWriter


EPOCHS = 100
TRAIN_DIR = "data/prepared_dataset/train"
VAL_DIR = "data/prepared_dataset/validation"
MEAN_STD_FILE = 'data/prepared_dataset/mean_std.txt'
SAVED_MODEL_DIR = 'saved_model_resnet50_batchsize8_224'
LOG_PATH = 'logs_resnet50_batchsize8_224'
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
GPU_DEVICE = torch.device("cuda:0")

def validate(model, validation_loader, loss_function):
    loss_sum = 0
    correct_sum = 0
    samples = 0
    softmax = torch.nn.Softmax(dim=1)

    for iter, batch in enumerate(validation_loader):
        image, label, name = batch
        image = image.to(GPU_DEVICE)
        label = label.to(GPU_DEVICE)

        with torch.set_grad_enabled(False):
            prediction = model(image)

            loss = loss_function(prediction, label)
            loss_value = loss.data.cpu().numpy()
            loss_sum += loss_value * BATCH_SIZE

            predicted_class = torch.max(softmax(prediction), 1)[1]
            num_corrects = torch.sum(predicted_class == label).data.cpu().numpy()
            correct_sum += num_corrects

            samples += BATCH_SIZE

            print("iter:{0:4d}, loss:{1:.3f}, label:{2:2d}, predicted_class:{3:2d}, prediction:{4}".format(
                iter + 1, loss_value, label.data.cpu().numpy()[0],
                predicted_class.data.cpu().numpy()[0],
                softmax(prediction)[0].data.cpu().numpy()))

    val_loss = float(loss_sum) / float(samples)
    val_accuracy = float(correct_sum) / float(samples)

    return val_loss, val_accuracy


def train():
    if not osp.exists(SAVED_MODEL_DIR):
        os.makedirs(SAVED_MODEL_DIR)

    if not osp.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    writer = SummaryWriter(log_dir=LOG_PATH)
    cudnn.enabled = True

    # with open(MEAN_STD_FILE, 'rb') as handle:
    #     data_mean_std = pickle.loads(handle.read())
    #
    # mean_train = data_mean_std['mean_train_images']
    # std_train = data_mean_std['std_train_images']
    # mean_val = data_mean_std['mean_val_images']
    # std_val = data_mean_std['std_val_images']
    #
    # print(mean_train)
    # print(std_train)
    # print(mean_val)
    # print(std_val)

    train_loader = data.DataLoader(WSI_Dataset(dir=TRAIN_DIR,
                                               transform=transform_pipe_train),
                                   batch_size=BATCH_SIZE,
                                   shuffle=True)

    validation_loader = data.DataLoader(WSI_Dataset(dir=VAL_DIR,
                                                    transform=transform_pipe_val_test),
                                        batch_size=BATCH_SIZE)

    model = torchvision.models.resnet50(pretrained=True)

    #model.avgpool = torch.nn.AdaptiveAvgPool2d(1)

    #replace the final fully connected layer to suite the problem
    model.fc = torch.nn.Linear(in_features=2048, out_features=4)

    if torch.cuda.device_count() > 1:
        print("using " + str(torch.cuda.device_count()) + "GPUs...")
        model = nn.DataParallel(model)

    model.eval()

    model.to(GPU_DEVICE)
    cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)
    optimizer.zero_grad()

    weight = np.array([1 / float(71), 1 / float(132), 1 / float(411), 1 / float(191)])
    weight_tensor = torch.from_numpy(weight).float().to(GPU_DEVICE)
    mce_loss = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    #mce_loss = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    best_val_accuracy = 0.0
    lowest_val_loss = 1000

    for epoch in range(1, EPOCHS+1):
        model.train()

        samples = 0
        loss_sum = 0
        correct_sum = 0

        for iter, batch in enumerate(train_loader):
            image, label, name = batch
            image = image.to(GPU_DEVICE)
            label = label.to(GPU_DEVICE)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                prediction = model(image)

                loss = mce_loss(prediction, label)
                loss.backward()
                optimizer.step()
                loss_value = loss.data.cpu().numpy()

                predicted_class = torch.max(softmax(prediction), 1)[1]

                loss_sum += loss_value * BATCH_SIZE  #we need to multiple by batch size as loss is the mean loss of the samples in the batch
                samples += BATCH_SIZE
                num_corrects = torch.sum(predicted_class == label).data.cpu().numpy()
                correct_sum += num_corrects

                print("epoch:{0:3d}, iter:{1:4d}, loss:{2:.3f}, label:{3:2d}, predicted_class:{4:2d}, prediction:{5}".format(epoch, iter+1, loss_value,
                                                                                                            label.data.cpu().numpy()[0],
                                                                                                            predicted_class.data.cpu().numpy()[0],
                                                                                                            softmax(prediction)[0].data.cpu().numpy()))

        #train epoch statistics
        epoch_loss = float(loss_sum) / float(samples)
        epoch_accuracy = float(correct_sum) / float(samples)

        print("epoch:{0:3d}, epoch_loss:{1:.3f}, epoch_acc: {2:.3f}".format(epoch, epoch_loss, epoch_accuracy))

        writer.add_scalar('training epoch loss', epoch_loss, epoch)
        writer.add_scalar('training epoch accuracy', epoch_accuracy, epoch)

        #validation
        model.eval()
        print("validating...")
        val_loss, val_accuracy = validate(model, validation_loader, mce_loss)

        print("val_loss: {0:.3f}, val_accuracy: {1:.3f}".format(val_loss, val_accuracy))

        writer.add_scalar('validation loss', val_loss, epoch)
        writer.add_scalar('validation accuracy', val_accuracy, epoch)

        if (val_accuracy > best_val_accuracy):
            best_val_accuracy = val_accuracy
            print('saving best model so far...')
            torch.save(model.state_dict(), osp.join(SAVED_MODEL_DIR, 'best_model_' + str(epoch) + '.pth'))

        scheduler.step(val_accuracy) #update learning rate


if __name__=='__main__':
    train()
