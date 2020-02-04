#author: Mohammad Minhazul Haq
#created on: February 4, 2020
#this file creates train, val and test dataset for WSI dataset

import os
import random
import pickle
import numpy as np
from PIL import Image


image_height = 1024
image_width = 1024


def copy_resized_images_to_target_dir(source_path, filenames, target_path):
    os.makedirs(target_path)

    for filename in filenames:
        #code courtesy: https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio
        image = Image.open(os.path.join(source_path, filename))
        image_resized = image.resize((image_width, image_height), Image.ANTIALIAS)
        image_resized.save(os.path.join(target_path, filename))


def generate_dataset(source_path, target_path):
    os.makedirs(target_path)

    train_val_test_images = sorted(os.listdir(source_path))
    random.shuffle(train_val_test_images)

    total_train_val_test_images = len(train_val_test_images)
    train_percentage = 0.6
    train_val_percentage = 0.7
    train_index = int(total_train_val_test_images * train_percentage)
    train_val_index = int(total_train_val_test_images * train_val_percentage)

    train_images = train_val_test_images[0 : train_index]
    train_images_target_dir = os.path.join(target_path, 'train')
    copy_resized_images_to_target_dir(source_path, train_images, train_images_target_dir)
    print('Training images: ' + str(len(train_images)))

    val_images = train_val_test_images[train_index : train_val_index]
    val_images_target_dir = os.path.join(target_path, 'validation')
    copy_resized_images_to_target_dir(source_path, val_images, val_images_target_dir)
    print('Validation images: ' + str(len(val_images)))

    test_images = train_val_test_images[train_val_index : ]
    test_images_target_dir = os.path.join(target_path, 'test')
    copy_resized_images_to_target_dir(source_path, test_images, test_images_target_dir)
    print('Test images: ' + str(len(test_images)))

    print('success')


def compute_mean_std(path, filename):
    train_image_path = os.path.join(path, 'train')
    val_image_path = os.path.join(path, 'validation')

    train_image_files = sorted(os.listdir(train_image_path))
    val_image_files = sorted(os.listdir(val_image_path))

    #train_images
    means_train_images = []
    stds_train_images = []

    for i, image_name in enumerate(train_image_files):
        img = np.array(Image.open(os.path.join(train_image_path, image_name)))
        avg = np.mean(img)
        std = np.std(img)
        means_train_images.append(avg)
        stds_train_images.append(std)

    #validation_images
    means_val_images = []
    stds_val_images = []

    for i, image_name in enumerate(val_image_files):
        img = np.array(Image.open(os.path.join(val_image_path, image_name)))
        avg = np.mean(img)
        std = np.std(img)
        means_val_images.append(avg)
        stds_val_images.append(std)

    #save mean, std to file
    data_to_save = {'mean_train_images': np.mean(means_train_images),
                    'std_train_images': np.mean(stds_train_images),
                    'mean_val_images': np.mean(means_val_images),
                    'std_val_images': np.mean(stds_val_images)}

    with open(os.path.join(path, filename), 'wb') as handle:
        pickle.dump(data_to_save, handle)


if __name__ == '__main__':
    source_path = os.path.join('data', 'original_dataset')
    dest_path = os.path.join('data', 'prepared_dataset')
    mean_std_filename = 'mean_std.txt'

    generate_dataset(source_path, dest_path)
    compute_mean_std(dest_path, mean_std_filename)
