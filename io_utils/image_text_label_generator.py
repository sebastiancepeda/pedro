import tensorflow as tf

from io_utils.data_source import get_image_text_label


class ImageTextLabelGenerator(tf.keras.utils.Sequence):
    def __init__(self, folder, metadata, dsize, in_channels, out_channels,
                 preprocess_input, params):
        self.alphabet = params['alphabet']
        image_name_list = metadata.image_name.unique()
        self.set_size = len(image_name_list)
        self.text_max_len = 13
        self.folder = folder
        self.metadata = metadata
        self.dsize = dsize
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.preprocess_input = preprocess_input
        self.params = params
        self.on_epoch_end()

    def __len__(self):
        return self.set_size

    def __getitem__(self, index):
        x, y = get_image_text_label(
            self.folder, self.metadata, self.dsize,
            self.in_channels, self.out_channels, self.params)
        x = self.preprocess_input(x)
        return x, y

    def on_epoch_end(self):
        pass
