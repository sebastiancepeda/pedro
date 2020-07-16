import os
import pathlib

import pandas as pd
from loguru import logger
from tensorflow import keras

from data_source import get_image_label_gen, load_label_data
from model_definition import get_model_definition


def train_model(params):
    epochs = params['epochs']
    dsize = params['dsize']
    model_file = params['model_file']
    model_folder = params['model_folder']
    labels = params['labels']
    metadata = params['metadata']
    folder = params['folder']
    #
    model, preprocess_input = get_model_definition()
    metadata = pd.read_csv(metadata)
    labels = pd.read_csv(labels, sep=',')
    labels = load_label_data(labels)
    labels = labels.rename(columns={'filename': 'image'})
    metadata = metadata.merge(labels, on=['image'], how='left')
    train_metadata = metadata.query("set == 'train'")
    test_metadata = metadata.query("set == 'test'")
    train_metadata = train_metadata.assign(idx=range(len(train_metadata)))
    test_metadata = test_metadata.assign(idx=range(len(test_metadata)))
    x_train, y_train = get_image_label_gen(folder, train_metadata, dsize)
    x_val, y_val = get_image_label_gen(folder, test_metadata, dsize)
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_file,
            save_weights_only=True,
            save_best_only=True,
            mode='min'),
    ]
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    # Training model
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=1,  # 16,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )
    model.load_weights(model_file)


def get_params():
    path = str(pathlib.Path().absolute())
    folder = f'{path}/plates'
    params = {
        'folder': folder,
        'epochs': 1000,
        'dsize': (768, 768),
        'model_folder': f'{folder}/model',
        'model_file': f'{folder}/model/best_model.h5',
        'labels': f"{folder}/labels_plates.csv",
        'metadata': f"{folder}/files.csv",
    }
    return params


if __name__ == "__main__":
    print = logger.info
    train_model(get_params())
