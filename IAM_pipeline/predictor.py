from validation_task import PredictorTask
import sys
import os
from keras.models import Model, Graph
from keras.layers import LSTM, TimeDistributed, Dense, Dropout, Input, Convolution2D, MaxPooling2D, Reshape, Permute, \
                         Merge, Activation, BatchNormalization
import numpy as np, time
import gzip, pickle, theano
from CTC_utils import CTC
from theano import tensor

def dim_shuffle(x, x_mask, y, y_mask):
    """

    :param x:
    :param x_mask:
    :param y:
    :param y_mask:
    :return:
    """
    x_dim = x[np.newaxis, :, :]
    x_mask_dim = x_mask  # [np.newaxis, :, :]
    y_dim = y
    y_mask_dim = y_mask
    # print("MASK", x_mask.shape, x_mask_dim.shape)

    return x_dim, x_mask_dim, y_dim, y_mask_dim


def _change_input_shape(floatx='float32'):
    x = tensor.tensor3('input', dtype=floatx)
    y = x.dimshuffle((0, 'x', 1, 2))
    f = theano.function([x], y, allow_input_downcast=True)
    return f

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
    Mask = np.zeros(shape=[image_ht, maxlen], dtype=image.dtype)

    trunc = image[:, :maxlen]

    Xout[:, :trunc.shape[1]] = trunc
    Mask[:, :trunc.shape[1]] = 1

    return Xout, Mask


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
        feadim = 32  #TODO
        chars = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2',
                      '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E',
                      'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                      'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                      'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}',
                      '~']  # data['chars']
        Nclass = len(chars)
        loss = 'ctc_cost_for_train'
        optimizer = 'Adadelta'
        border_mode = 'same'

        self.minNcharPerseq, self.maxNcharPerseq = 2, 10

        net_input = Input(shape=(1, feadim, None))  #net_input = Input(shape=(1, feadim, None))  #TODO maxlength
        cnn0   = Convolution2D( 64, 3, 3, border_mode=border_mode, activation='relu', name='cnn0')(net_input)
        pool0  = MaxPooling2D(pool_size=(2, 2), name='pool0')(cnn0)
        cnn1   = Convolution2D(128, 3, 3, border_mode=border_mode, activation='relu', name='cnn1')(pool0)
        pool1  = MaxPooling2D(pool_size=(2, 2), name='pool1')(cnn1)
        cnn2   = Convolution2D(256, 3, 3, border_mode=border_mode, activation='relu', name='cnn2')(pool1)
        # BN0    = BatchNormalization(mode=0, axis=1, name='BN0')(cnn2)
        cnn3   = Convolution2D(256, 3, 3, border_mode=border_mode, activation='relu', name='cnn3')(cnn2)
        pool2  = MaxPooling2D(pool_size=(2, 1), name='pool2')(cnn3)
        cnn4   = Convolution2D(512, 3, 3, border_mode=border_mode, activation='relu', name='cnn4')(pool2)
        BN0    = BatchNormalization(mode=0, axis=1, name='BN0')(cnn4)
        cnn5   = Convolution2D(512, 3, 3, border_mode=border_mode, activation='relu', name='cnn5')(BN0)
        BN1 = BatchNormalization(mode=0, axis=1, name='BN1')(cnn5)
        pool3  = MaxPooling2D(pool_size=(2, 1), name='pool3')(BN1)
        cnn6   = Convolution2D(512,   2, 2, border_mode='valid', activation='relu', name='cnn6')(pool3)  # MAYBE BORDER MODE

        net_reshape = Permute((1, 0, 3, 2), name='net_reshape')(cnn6)

        #Reshape(input_width, num_filters)

        lstm0  = LSTM(256, return_sequences=True, activation='tanh', name='lstm0')(net_reshape)  # bi lstm missing
        lstm1  = LSTM(256, return_sequences=True, activation='tanh', go_backwards=True, keep_time_order=True, name='lstm1')(lstm0)
        dense0 = TimeDistributed(Dense(Nclass + 1, activation='softmax', name='dense0'))(lstm1)
        self.model  = Model(net_input, dense0)
        self.model.compile(loss=loss, optimizer=optimizer, sample_weight_mode='temporal')

        print("Compiled Keras model successfully.")

        self.reshape_func = _change_input_shape()
        print('reshape_func compiled')

    def FeatureExtractor(self, img):
        """

        :param img:
        :return:
        """
        feature_vec = np.zeros((9, img.shape[1]))

        # needed values
        m = img.shape[0]

        for col in range(img.shape[1]):
            # 1 Number of black Pixel
            feature_vec[0, col] = np.sum(img[:, col]) / 255.0  # TODO /255 oder nicht?

            # 2 Center of Gravity
            feature_vec[1, col] = np.sum(np.arange(1, m + 1) * img[:, col]) / m

            # 3 Second order Center of Gravitiy "Second order moment"
            feature_vec[2, col] = np.sum(np.square(np.arange(1, m + 1)) * img[:, col]) / np.square(m)

            # 4/5 Position top / bottom black
            black_pixels = np.nonzero(img[:, col])
            if np.sum(img[:, col]) == 0:
                feature_vec[3, col] = 1
                feature_vec[4, col] = m
            else:
                feature_vec[3, col] = black_pixels[0][0] + 1
                feature_vec[4, col] = black_pixels[0][-1] + 1

            # 6/ 7 grad top/bottom
            if col == 0:
                feature_vec[5, col] = 0
                feature_vec[6, col] = 0
            else:
                feature_vec[5, col] = feature_vec[3, col] - feature_vec[3, col - 1]
                feature_vec[6, col] = feature_vec[4, col] - feature_vec[4, col - 1]

            # 8 Number of transitions between black and white
            feature_vec[7, col] = np.sum(np.absolute(np.diff(np.asarray(img[:, col])))) / 255

            # 9 How many black between top and black
            if np.sum(img[:, col]) == 0:
                feature_vec[8, col] = 0
            else:
                feature_vec[8, col] = np.sum(
                    img[:, col][black_pixels[0][0]:black_pixels[0][-1] + 1]) / 255.0  # TODO /255 oder nicht?

        # Pad Sequence with '0' so that every sequence has same length
        feature_vec_pad = sequence.pad_sequences(feature_vec, maxlen=self.maxlen)

        return feature_vec_pad

    def train_rnn(self, img_feat_vec, label):
        """

        :param img_feat_vec:
        :param label:
        """
        print('Train...')
        loss, accuracy = self.model.train_on_batch(img_feat_vec, label)

        return loss, accuracy

    def test_rnn(self, img_feat_vec, label):
        """

        :param img_feat_vec:
        :param label:
        :return:
        """
        print('Evaluate...')
        loss, accuracy = self.model.test_on_batch(img_feat_vec, label)

        return loss, accuracy

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
        x_padded, x_mask = pad_sequence_into_array(input_tuple[0], 151)  #TODO MAXLENGTH!!!
        y_padded, y_mask = pad_sequence_into_array(input_tuple[1], 151)  # TODO MAXLENGTH!!!

        # Dim Shuffle to fit Keras [40x150] --> [1 x 40 x 150]
        x_padded, x_mask, y_padded, y_mask = dim_shuffle(x_padded, x_mask, y_padded, y_mask)

        print('Image shape:', x_padded.shape)  # (B, T, D)
        print('Label shape:', y_padded.shape)  # (B, L)

        B, T, D = x_padded.shape  # D = 28
        L = y_padded.shape[1]

        total_seqlen, total_ed = 0.0, 0.0

        time0 = time.time()

        traindata = self.reshape_func(x_padded)
        traindata_mask = np.transpose(x_mask)   # TODO [0, 0::16][:, :-1]

        gt = y_padded
        gt_mask = y_mask

        print('Traindata:', traindata.shape)  # (1x1x150x40)
        print('GT:', gt.shape)  # (1x150)
        print('GT Mask:', gt_mask.shape)  # (1x150)
        print('Traindata Mask:', traindata_mask.shape)  # (40x150)

        ctcloss, score_matrix = self.model.train_on_batch(x=traindata, y=gt, sample_weight=gt_mask, sm_mask=traindata_mask, return_sm=True)

        print('ctcloss = ', ctcloss)
        resultseqs = CTC.best_path_decode_batch(score_matrix, traindata_mask)
        targetseqs = convert_gt_from_array_to_list(gt, gt_mask)
        CER_batch, ed_batch, seqlen_batch = CTC.calc_CER(resultseqs, targetseqs)
        total_seqlen += seqlen_batch
        total_ed += ed_batch
        CER = total_ed / total_seqlen * 100.0
        time1 = time.time()

        print('CER = %0.2f, CER_batch = %0.2f, time = %0.2fs' % (
        CER, CER_batch, (time1 - time0)))

        # 2. Neural Net
        if test_set == 0:
            loss, acc = self.train_rnn(feature_vec, y_train)
            # cst, pred = self.train_rnn(feature_vec, input_tuple[1])
        else:
            loss, acc = self.test_rnn(feature_vec, input_tuple[1])
            # pred = self.classify_rnn(feature_vec)

        return [input_tuple[1], loss, acc]  #TODO DIFFERNET OUTPUT

    def save(self, directory):
        print ("Saving myPredictor to ", directory)
