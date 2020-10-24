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
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            model_file,
            save_weights_only=True,
            save_best_only=False,  # save_best_only=True,
            mode='min')
    tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir='./graphs', histogram_freq=0, batch_size=32,
            write_graph=False, write_grads=False, write_images=False,
            embeddings_freq=0, embeddings_layer_names=None,
            embeddings_metadata=None, embeddings_data=None,
            update_freq='epoch')
    callbacks = [
        model_checkpoint_callback,
        tensorboard_callback
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


def compose_fs(functions):
    def composed_function(x):
        for f in functions:
            x = f(x)
        return x

    return composed_function


def identity_function(x):
    return x
