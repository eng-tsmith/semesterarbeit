from validation_task import PredictorTask
import sys
import os
import numpy as np
from keras import backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input, Layer, Dense, Activation, Flatten
from keras.layers import Reshape, Lambda, merge, Permute, TimeDistributed
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks



def pad_sequence_into_array(image, maxlen):
    """
    Padding sequence (list of numpy arrays) into an numpy array
    :param Xs: list of numpy arrays. The arrays must have the same shape except the first dimension.
    :param maxlen: the allowed maximum of the first dimension of Xs's arrays. Any array longer than maxlen is truncated to maxlen
    :return: Xout, the padded sequence (now an augmented array with shape (Narrays, N1stdim, N2nddim, ...)
    :return: mask, the corresponding mask, binary array, with shape (Narray, N1stdim)
    """
    value = 0.
    image_ht = image.shape[0]

    Xout = np.ones(shape=[image_ht, maxlen], dtype=image[0].dtype) * np.asarray(value, dtype=image[0].dtype)

    trunc = image[:, :maxlen]

    Xout[:, :trunc.shape[1]] = trunc

    return Xout

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

class IAM_Predictor(PredictorTask):

    def __init__(self):
        """
    Input shape: X.shape=(B, 1, rows, cols), GT.shape=(B, L)
    :param feadim: input feature dimension
    :param Nclass: class number
    :param loss:
    :param optimizer:
    :return:
    """

        # Input Parameters
        chars = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2',
                      '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E',
                      'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                      'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                      'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}',
                      '~']  # data['chars']

        # Input Parameters
        self.img_h = 64
        self.img_w = 512
        self.absolute_max_string_len = 25 # TODO
        self.output_size = len(chars)
        minibatch_size = 1  #TODO
        words_per_epoch = 16000  # TODO

        # Network parameters
        conv_num_filters = 16
        filter_size = 3
        pool_size_1 = 4
        pool_size_2 = 2
        time_dense_size = 32
        rnn_size = 512
        time_steps = self.img_w / (pool_size_1 * pool_size_2)
        lr = 0.03
        # clipnorm seems to speeds up convergence
        clipnorm = 5
        self.downsampled_width = int(self.img_w / (pool_size_1 * pool_size_2) - 2)

        # Optimizer
        sgd = SGD(lr=lr, decay=3e-7, momentum=0.9, nesterov=True, clipnorm=clipnorm)
        # Activition functrion
        act = 'relu'

        if K.image_dim_ordering() == 'th':
            input_shape = (1, self.img_h, self.img_w)
        else:
            input_shape = (self.img_h, self.img_w, 1)

        #################################################
        # Network archtitecture
        input_data = Input(name='the_input', shape=input_shape, dtype='float32')

        # CNN encoder
        inner = Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                              activation=act, name='conv1')(input_data)
        inner = MaxPooling2D(pool_size=(pool_size_1, pool_size_1), name='max1')(inner)
        inner = Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                              activation=act, name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(pool_size_2, pool_size_2), name='max2')(inner)

        # CNN to RNN convert
        conv_to_rnn_dims = (
        (self.img_h / (pool_size_1 * pool_size_2)) * conv_num_filters, self.img_w / (pool_size_1 * pool_size_2))

        a = conv_to_rnn_dims[0]
        b = conv_to_rnn_dims[1]
        c = [int(a), int(b)]

        inner = Reshape(target_shape=c, name='reshape')(inner)
        inner = Permute(dims=(2, 1), name='permute')(inner)

        # cuts down input size going into RNN:
        inner = TimeDistributed(Dense(time_dense_size, activation=act, name='dense1'))(inner)

        # RNN
        # Two layers of bidirecitonal GRUs
        # GRU seems to work as well, if not better than LSTM:
        gru_1 = GRU(rnn_size, return_sequences=True, name='gru1')(inner)
        gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, name='gru1_b')(inner)
        gru1_merged = merge([gru_1, gru_1b], mode='sum')
        gru_2 = GRU(rnn_size, return_sequences=True, name='gru2')(gru1_merged)
        gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True)(gru1_merged)

        # transforms RNN output to character activations:
        inner = TimeDistributed(Dense(self.output_size, name='dense2'))(merge([gru_2, gru_2b], mode='concat'))
        y_pred = Activation('softmax', name='softmax')(inner)
        # Model(input=[input_data], output=y_pred).summary()

        # LABELS
        labels = Input(name='the_labels', shape=[self.absolute_max_string_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        # CTC layer
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")([y_pred, labels, input_length, label_length])

        # Keras Model of NN
        Model(input=[input_data, labels, input_length, label_length], output=[loss_out]).summary()
        self.model = Model(input=[input_data, labels, input_length, label_length], output=[loss_out])

        # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

        # Init NN done
        print("Compiled Keras model successfully.")
        # OLD PRINT
        # # captures output of softmax so we can decode the output during visualization
        # test_func = K.function([input_data], [y_pred])
        #
        # viz_cb = VizCallback(test_func, img_gen.next_val())

    def train_rnn(self, inputs):
        """

        :param img_feat_vec:
        :param label:
        """
        print('Train...')

        loss = self.model.train_on_batch(self, inputs, class_weight=None, sample_weight=None)  #TODO metrics?

        return loss

    def test_rnn(self, inputs):
        """

        :param img_feat_vec:
        :param label:
        :return:
        """
        print('Test...')
        loss = self.model.test_on_batch(self, inputs, class_weight=None, sample_weight=None)  #TODO metrics?

        return loss

    def predict_rnn(self, img_feat_vec):
        """

        :param img_feat_vec:
        """
        print('Evaluate...')
        pred = self.model.predict_on_batch(img_feat_vec)

        return pred

    def run(self, input_tuple, test_set):
        """ TODO:
        This function takes a normalized image as Input. During predicting following steps are computed:
         1. Feature Extractor
         2. Neural Net

        :param input_tuple:
        :return:
        """
        # NN Preprocessing
        # inputs = {'the_input': X_data,    (1, self.img_h, self.img_w)
        #           'the_labels': labels,   int list
        #           'input_length': input_length,  img_w / (pool_size_1 * pool_size_2) - 2  --> self.downsampled_width
        #           'label_length': label_length,   len label
        #           }
        #
        # outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function

        x_padded = pad_sequence_into_array(input_tuple[0], self.img_w)
        y_with_blank = input_tuple[1]  #TODO blank

        the_input = np.asarray(x_padded, dtype='float32')
        the_labels = np.asarray(y_with_blank, dtype='float32')
        input_length = np.array([self.downsampled_width], dtype='int64')
        label_length = np.array([len(the_labels)], dtype='int64')

        inputs = [the_input[np.newaxis, :, :], the_labels, input_length, label_length]

        outputs = {'ctc': np.zeros([1])}

        print('Input: ', inputs[0].shape)
        print('Label: ', inputs[1].shape)
        print('Input_length: ', inputs[2])
        print('Label_length: ', inputs[3])

        # Neural Net
        if test_set == 0:
            loss, metric = self.train_rnn(inputs)
            # cst, pred = self.train_rnn(feature_vec, input_tuple[1])
        else:
            loss, metric = self.test_rnn(inputs)

        metric = 0

        return [input_tuple[1], loss, metric]  #TODO DIFFERNET OUTPUT

    def save(self, directory):
        print ("Saving myPredictor to ", directory)
