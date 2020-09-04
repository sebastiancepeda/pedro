import tensorflow as tf
from loguru import logger

from io_utils.data_source import (
    get_plates_text_metadata,
    get_image_text_label,
)
from io_utils.image_text_label_generator import ImageTextLabelGenerator
from io_utils.utils import set_index
from cv.tensorflow_models.cnn_encoder import CNN_Encoder
from cv.tensorflow_models.rnn_decoder import RNN_Decoder
from cv.tensorflow_models.unet2text2 import (
    get_model_definition,
    normalize_image_shape
)


def get_params():
    path = '/home/sebastian/projects/pedro/data'
    input_folder = f'{path}/plates/plate_segmentation'
    output_folder = f'{path}/plates/plate_ocr'
    width = 200
    height = 50
    height, width = normalize_image_shape(height, width)
    # height = height + 1
    # width = width + 1
    dsize = (height, width)
    alphabet = ' abcdefghijklmnopqrstuvwxyz0123456789'
    alphabet = {char: idx for char, idx in zip(alphabet, range(len(alphabet)))}
    in_channels = 1
    out_channels = len(alphabet)
    params = {
        'input_folder': input_folder,
        'output_folder': output_folder,
        'epochs': 1 * 1000,
        'dsize': dsize,
        'model_folder': f'{output_folder}/model',
        'model_file': f'{output_folder}/model/best_model.h5',
        'metadata': f"{path}/plates/input/labels/ocr/files.csv",
        'alphabet': alphabet,
        'model_params': {
            'img_height': dsize[0],
            'img_width': dsize[1],
            'in_channels': in_channels,
            'out_channels': out_channels,
        },
    }
    return params


def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def train_ocr_model(params):
    embedding_dim = 10
    units = 20
    vocab_size = len(params['alphabet'])
    num_steps = 10
    alphabet = '*- abcdefghijklmnopqrstuvwxyz0123456789'  # {*: start, -: end}
    word_index = {char: idx for idx, char in enumerate(alphabet)}
    index_word = {idx: char for idx, char in enumerate(alphabet)}

    img_height = 16 * 4
    img_width = 16 * 16
    encoder = CNN_Encoder(embedding_dim, img_height, img_width)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    optimizer = tf.keras.optimizers.Adam()
    loss_plot = []

    @tf.function
    def train_step(img_tensor, target):
        loss = 0
        """
        Reset of the hidden state for each batch
        """
        print(target.numpy())
        batch_size = target.shape[0]
        sentence_len = target.shape[2]
        hidden = decoder.reset_state(batch_size=batch_size)
        dec_input = tf.expand_dims([word_index['*']] * batch_size, 1)
        with tf.GradientTape() as tape:
            features = encoder(img_tensor)
            for idx in range(1, sentence_len):
                # Passing the features through the decoder
                predictions, hidden, _ = decoder(dec_input, features, hidden)
                # print(1, predictions.numpy())
                target_char = tf.reshape(target[0, 0, idx, :], (1, target.shape[-1]))
                target_char = tf.argmax(target_char, axis=1)
                # print(2, target_char.eval())
                partial_loss = loss_function(target_char, predictions)
                loss += partial_loss
                # Using teacher forcing
                dec_input = tf.expand_dims(target_char, 1)
        total_loss = (loss / sentence_len)
        trainable_variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss, total_loss

    epochs = 20

    dsize = params['dsize']
    model_params = params['model_params']
    in_folder = params['input_folder']
    alphabet = params['alphabet']
    #
    in_channels = model_params['in_channels']
    out_channels = model_params['out_channels']
    metadata = get_plates_text_metadata(params)
    metadata.file_name = 'plate_' + metadata.file_name
    metadata.file_name = metadata.file_name.str.split('.').str[0] + '.png'
    train_meta = metadata.query("set == 'train'")
    train_meta = set_index(train_meta)
    model, preprocess_input = get_model_definition(**model_params)
    f_train_params = {
        'folder': in_folder, 'metadata': train_meta, 'dsize': dsize,
        'in_channels': in_channels, 'out_channels': out_channels,
        'alphabet': alphabet
    }
    data_train = ImageTextLabelGenerator(get_image_text_label, preprocess_input, f_train_params)
    for epoch in range(0, epochs):
        total_loss = 0
        for batch, (img_tensor, target) in enumerate(data_train):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss
            # logger.info(f"target.mean(): {target.mean()}")
            loss_debug = batch_loss.numpy() / int(target.shape[1])
            logger.info(f'Epoch {epoch + 1} Batch {batch} Loss {loss_debug}')
        # Storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)
        logger.info(f'Epoch {epoch + 1} Loss {total_loss / num_steps}')


if __name__ == "__main__":
    train_ocr_model(get_params())
