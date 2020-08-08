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
        self.x_all = None
        self.y_all = None
        self.index_im = 0
        self.on_epoch_end()

    def __len__(self):
        return len(self.x_all)

    def __getitem__(self, index):
        x = self.x_all[self.index_im]
        y = self.y_all[self.index_im]
        self.index_im = self.index_im + 1
        x = self.preprocess_input(x)
        return x, y

    def on_epoch_end(self):
        self.index_im = 0
        x_all, y_all = get_image_text_label(
            self.folder, self.metadata, self.dsize,
            self.in_channels, self.out_channels, self.params)
        self.x_all = x_all
        self.y_all = y_all
