import pathlib

import pandas as pd
from loguru import logger

from data_source import get_image_label_gen, load_label_data
from image_processing import image_set2list
from image_processing import print_named_images


if __name__ == "__main__":
    print = logger.info
    print("Loading model")
    path = str(pathlib.Path().absolute())
    folder = f'{path}/plates'
    print("Loading data")
    metadata = pd.read_csv(f"{folder}/files.csv")
    labels = f"{folder}/labels_plates_20200707023447.csv"
    labels = pd.read_csv(labels, sep=',')
    labels = load_label_data(labels)
    labels = labels.rename(columns={'filename': 'image'})
    metadata = metadata.merge(labels, on=['image'], how='left')
    dsize = (768, 768)
    train_metadata = metadata.query("set == 'train'")
    test_metadata = metadata.query("set == 'test'")
    train_metadata = train_metadata.assign(idx=range(len(train_metadata)))
    test_metadata = test_metadata.assign(idx=range(len(test_metadata)))
    x_train, y_train = get_image_label_gen(folder, train_metadata, dsize)
    x_val, y_val = get_image_label_gen(folder, test_metadata, dsize)
    images = image_set2list(x_train, x_val)
    print("images")
    y_train_pred = ((y_train) * 255).round()
    y_val_pred = ((y_val) * 255).round()
    images_pred = image_set2list(y_train_pred, y_val_pred)
    y_train_pred = ((x_train) * 1).round()
    y_val_pred = ((x_val) * 1).round()
    images_x = image_set2list(y_train_pred, y_val_pred)
    print_named_images(images_x, "x")
    print_named_images(images_pred, "y")
