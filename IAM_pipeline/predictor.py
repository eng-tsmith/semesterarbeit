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
import datetime
import itertools
import editdistance


def decode_batch(test_func, word_batch):
    chars = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '.', ',', '\'', '\"']
    n_classes = len(chars)

    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        # 26 is space, 27 is CTC blank char
        outstr = []

        for il, l in enumerate(out_best):
            if (l != n_classes) and (il == 0 or l != out_best[il - 1]):
                outstr.append(chars[l])
        ret = outstr
    return ret


class TimCallback(keras.callbacks.Callback):

    def __init__(self, test_func, num_display_words=1):   # TODO NUM_DISPLAY WORDS
        # def __init__(self, test_func, text_img_gen, num_display_words=6):
        OUTPUT_DIR = 'output'
        self.true_string = ''
        self.test_func = test_func
        self.output_dir = os.path.join(
            OUTPUT_DIR, datetime.datetime.now().strftime('%A, %d. %B %Y %I.%M%p'))
        # self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        os.makedirs(self.output_dir, exist_ok=True)
        print("Callback init")

    def init_true_string(self, label):
        self.true_string = label

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = self.model.validation_data

            num_proc = min(word_batch[0].shape[0], num_left)
            decoded_res = decode_batch(self.test_func, word_batch[0][0:num_proc])
            for j in range(0, num_proc):
                edit_dist = editdistance.eval(decoded_res[j], self.true_string)
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(self.true_string)
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
              % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        print("Callback Aufruf")
        self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % epoch))
        self.show_edit_distance(256)

        word_batch = self.model.validation_data
        # res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])
        # import ipdb
        # ipdb.set_trace()
        res = decode_batch(self.test_func, word_batch[0])

        out_str = []
        for c in res:
            out_str.append(c)
        dec_string = "".join(out_str)

        print('Truth: ', self.true_string, '   ---   Decoded: ', dec_string)

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


def pad_label_with_blank(label, blank_id, max_length):
    """
    Padding sequence (list of numpy arrays) into an numpy array
    :param Xs: list of numpy arrays. The arrays must have the same shape except the first dimension.
    :param maxlen: the allowed maximum of the first dimension of Xs's arrays. Any array longer than maxlen is truncated to maxlen
    :return: Xout, the padded sequence (now an augmented array with shape (Narrays, N1stdim, N2nddim, ...)
    :return: mask, the corresponding mask, binary array, with shape (Narray, N1stdim)
    """
    label_len_1 = len(label[0])
    label_len_2 = len(label[0])

    label_pad = []
    label_pad.append(blank_id)
    for _ in label[0]:
        label_pad.append(_)
        label_pad.append(blank_id)

    while label_len_2 < max_length:
        label_pad.append(-1)
        label_len_2 += 1

    label_out = np.ones(shape=[max_length]) * np.asarray(blank_id)

    trunc = label_pad[:max_length]
    # import ipdb
    # ipdb.set_trace()
    label_out[:len(trunc)] = trunc

    return label_out, label_len_1

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
        # chars = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2',
        #          '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E',
        #          'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
        #          'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
        #          'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}',
        #          '~']  # data['chars']
        chars = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
                 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
                 'x', 'y', 'z', '.', ',', '\'', '\"']

        # Input Parameters
        self.img_h = 64
        self.img_w = 512
        self.absolute_max_string_len = 16  # TODO
        self.output_size = len(chars)

        # Network parameters
        conv_num_filters = 16
        filter_size = 3
        pool_size_1 = 4
        pool_size_2 = 2
        time_dense_size = 32
        rnn_size = 512
        time_steps = self.img_w / (pool_size_1 * pool_size_2)
        lr = 0.001
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
        inner = TimeDistributed(Dense(self.output_size+1, name='dense2'))(merge([gru_2, gru_2b], mode='concat'))
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

        # captures output of softmax so we can decode the output during visualization
        self.test_func = K.function([input_data], [y_pred])

        self.cb = TimCallback(self.test_func)

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

        # self.model.train_on_batch(inputs[0], inputs[1], class_weight=None, sample_weight=None)
        history_callback = self.model.fit(inputs[0], inputs[1], batch_size=1, nb_epoch=1)
        return history_callback

    def test_rnn(self, inputs):
        """

        :param img_feat_vec:
        :param label:
        :return:
        """
        print('Test...')

        history_callback = self.model.fit(inputs[0], inputs[1], batch_size=1, nb_epoch=1, validation_data=inputs, callbacks=[self.cb])
        return history_callback

    def predict_rnn(self, inputs):
        """

        :param img_feat_vec:
        """
        print('Evaluate...')
        pred = self.predict(inputs[0], batch_size=1, verbose=0)
        print('Prediction: ', pred, 'True Label', inputs[0][1])

        return pred

    def run(self, input_tuple, test_set):
        """ TODO:
        This function takes a normalized image as Input. During predicting following steps are computed:
         1. Feature Extractor
         2. Neural Net

        :param  input_tuple [img_norm, label, label_raw] :
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
        y_with_blank, y_len = pad_label_with_blank(np.asarray(input_tuple[1]), self.output_size, self.absolute_max_string_len)  #TODO blank

        in1 = np.asarray(x_padded, dtype='float32')[np.newaxis, np.newaxis, :, :]
        in2 = np.asarray(y_with_blank, dtype='float32')[np.newaxis, :]
        in3 = np.array([self.downsampled_width], dtype='float32')[np.newaxis, :]
        in4 = np.array([y_len], dtype='float32')[np.newaxis, :]

        out1 = np.zeros([1])

        inputs = {'the_input': in1,
                  'the_labels': in2,
                  'input_length': in3,
                  'label_length': in4}
        outputs = {'ctc': out1}

        # print('the_input', in1[0].shape)
        # print('the_labels', in2[0])
        # print('input_length', in3[0])
        # print('label_length', in4[0])

        # Neural Net
        if test_set == 0:
            history = self.train_rnn((inputs, outputs))
            # cst, pred = self.train_rnn(feature_vec, input_tuple[1])
        else:
            # Init true string
            in5 = []
            for c in input_tuple[2]:
                in5.append(c)
            string = "".join(in5)
            self.cb.init_true_string(string)
            history = self.test_rnn((inputs, outputs))

        print(history.history["loss"])

        return [input_tuple[1], history.history["loss"], 0]  #TODO DIFFERNET OUTPUT

    def save(self, directory):
        print ("Saving myPredictor to ", directory)
