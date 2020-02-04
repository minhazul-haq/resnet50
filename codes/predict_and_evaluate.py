#author: Mohammad Minhazul Haq
#created on: February 3, 2020

import torch
import torchvision
import pickle
import torch.nn as nn
from torch.utils import data
from dataset_loader import WSI_Dataset
from utils import transform_pipe_val_test


BEST_MODEL_PATH = "saved_model_resnet50_batchsize8_512/best_model_18.pth"
BATCH_SIZE = 8
TEST_DIR = "data/prepared_dataset/test"
MEAN_STD_FILE = 'data/prepared_dataset/mean_std.txt'
GPU_DEVICE = torch.device("cuda:0")


def predict():
    model = torchvision.models.resnet50(pretrained=True)
    #model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    model.fc = torch.nn.Linear(in_features=2048, out_features=4)

    if torch.cuda.device_count() > 1:
        print("using " + str(torch.cuda.device_count()) + "GPUs...")
        model = nn.DataParallel(model)

    saved_state_dict = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(saved_state_dict)
    model.eval()

    model.to(GPU_DEVICE)

    # with open(MEAN_STD_FILE, 'rb') as handle:
    #     data_mean_std = pickle.loads(handle.read())
    #
    # mean_val = data_mean_std['mean_val_images']
    # std_val = data_mean_std['std_val_images']
    #
    # print(mean_val)
    # print(std_val)

    test_loader = data.DataLoader(WSI_Dataset(dir=TEST_DIR,
                                              transform=transform_pipe_val_test),
                                  batch_size=BATCH_SIZE)

    correct_sum = 0
    samples = 0
    softmax = torch.nn.Softmax(dim=1)

    for iter, batch in enumerate(test_loader):
        image, label, name = batch
        image = image.to(GPU_DEVICE)
        label = label.to(GPU_DEVICE)

        with torch.set_grad_enabled(False):
            prediction = model(image)
            predicted_class = torch.max(softmax(prediction), 1)[1]
            num_corrects = torch.sum(predicted_class == label).data.cpu().numpy()

            for b in range(image.shape[0]):
                print(name[b] + ": " + str(predicted_class[b].data.cpu().numpy()))

            correct_sum += num_corrects
            samples += image.shape[0]

    accuracy = float(correct_sum) / float(samples)
    print("Accuracy: {0:.3f}".format(accuracy))


if __name__=='__main__':
    predict()
