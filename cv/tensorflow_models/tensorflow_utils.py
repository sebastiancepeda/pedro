import os

from tensorflow import keras


def train_model(x_train, y_train, x_val, y_val, get_model_definition, params,
                logger):
    epochs = params['epochs']
    model_file = params['model_file']
    model_folder = params['model_folder']
    model_params = params['model_params']
    #
    model, preprocess_input = get_model_definition(**model_params)
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_file,
            save_weights_only=True,
            # save_best_only=False,
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
    return model


def train_model_gen(data_train, data_val, model, params, logger):
    epochs = params['epochs']
    model_file = params['model_file']
    model_folder = params['model_folder']
    #
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_file,
            save_weights_only=True,
            save_best_only=False,
            # save_best_only=True,
            mode='min'),
        keras.callbacks.TensorBoard(
            './graphs',
            histogram_freq=1,  # Freq compute activation and weight histograms
            write_graph=True,  # visualize the graph
            write_grads=True,  # visual gradient histogram
            write_images=True,  # visualize weights as an image
            # embeddings_freq=1,
            # embeddings_layer_names=['...'],
            update_freq='epoch'
            # update TensorBoard every epoch
        )
    ]
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    # Training model
    model.fit(
        x=data_train,
        steps_per_epoch=len(data_train),
        epochs=epochs,
        validation_data=data_val,
        validation_steps=len(data_val),
        shuffle=False,
        callbacks=callbacks,
    )
    model.load_weights(model_file)
    return model
