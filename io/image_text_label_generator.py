import tensorflow as tf


class ImageTextLabelGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_function, preprocess_input, data_function_params):
        self.x_all = None
        self.y_all = None
        self.index_im = 0
        self.data_function = data_function
        self.preprocess_input = preprocess_input
        self.data_function_params = data_function_params
        self.on_epoch_end()

    def __len__(self):
        return len(self.x_all)

    def __getitem__(self, index):
        x = self.x_all[self.index_im, :, :, :]
        y = self.y_all[self.index_im, :, :, :]
        x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        y = y.reshape(1, y.shape[0], y.shape[1], y.shape[2])
        self.index_im = self.index_im + 1
        if self.index_im >= len(self.x_all):
            self.on_epoch_end()
        x = self.preprocess_input(x)
        return x, y

    def on_epoch_end(self):
        self.index_im = 0
        x_all, y_all = self.data_function(**self.data_function_params)
        self.x_all = x_all
        self.y_all = y_all
