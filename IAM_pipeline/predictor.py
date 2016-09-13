from validation_task import PredictorTask
# import pickle
import sys
import numpy as np
import theano as th
from rnn_ctc.nnet.neuralnet import NeuralNet
import rnn_ctc.utils as utils
import IAM_pipeline.data_config as data_config


def FeatureExtractor(img):
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
    return feature_vec


class IAM_Predictor(PredictorTask):

    def __init__(self):
        """
        When this funtion is first called it initalizes the net.
        """
        print('Initializing IAM Predictor. Loading   net_config.ast:')

        self.args = utils.read_args(['net_config.ast'])

        self.num_epochs = self.args['num_epochs']
        self.img_ht = self.args['img_ht']
        self.train_on_fraction = self.args['train_on_fraction']

        self.scribe_args = self.args['scribe_args']
        self.nnet_args = self.args['nnet_args'],

        self.chars = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~'] # data['chars']
        self.num_classes = len(self.chars)

        self.labels_print, self.labels_len = self.prediction_printer(self.chars)

        print('\nNetwork Info:'
              '\nInput Dim: {}'
              '\nNum Classes: {}'
              '\nNum Epochs: {}'
              '\nFloatX: {}'
              '\n'.format(self.img_ht, self.num_classes, self.num_epochs, th.config.floatX))

        print('Building the Network')
        self.net = NeuralNet(self.img_ht, self.num_classes, **self.nnet_args)
        print(self.net)
        self.printer = utils.Printer(self.chars)



    def show_all_tim(self, shown_seq, shown_img,
                     softmax_firings=None,
                     *aux_imgs):
        """
        Utility function to show the input and output and debug
        :param shown_seq: Labelings of the input
        :param shown_img: Input Image
        :param softmax_firings: Seen Probabilities (Excitations of Softmax)
        :param aux_imgs: List of pairs of images and names
        :return:
        """
        print('True Label : ', end='')
        self.labels_print(shown_seq)

        if softmax_firings is not None:
            print('Predicted Label  : ', end='')
            maxes = np.argmax(softmax_firings, 0)
            self.labels_print(maxes)


    def prediction_printer(self, chars):  #TODO VON htr-CTC!!!!
        """
        Returns a function that can print a predicted output of the CTC RNN
        It removes the blank characters (need to be set to n_classes),
        It also removes duplicates
        :param list chars: list of characters
        :return: the printing functions
        """
        n_classes = len(chars)

        def yprint(labels):
            labels_out = []
            for il, l in enumerate(labels):
                if (l != n_classes) and (il == 0 or l != labels[il - 1]):
                    labels_out.append(l)
            print(labels_out, " ".join(chars[l] for l in labels_out))

        def ylen(labels):
            length = 0
            for il, l in enumerate(labels):
                if (l != n_classes) and (il == 0 or l != labels[il - 1]):
                    length += 1
            return length

        return yprint, ylen

    def train_rnn(self, img_feat_vec, label):
        """

        :param features:
        :param labels:
        :return:
        """
        print('Training the Network')

        x = np.asarray(img_feat_vec, dtype=th.config.floatX)  # data_x[samp]
        y = np.asarray(utils.insert_blanks(label, self.num_classes), dtype=np.int32)    # data_y[samp]
        # if not samp % 500:            print(samp)

        # print("x: ", x, "   y: ", y)
        cst, pred, aux = self.net.trainer(x, y)

        if np.isinf(cst):
            print('Cost is infinite')
            sys.exit()
        # print(self.net)

        return cst, pred, aux


    def classify_rnn(self, img_feat_vec):
        """

        :param features:
        :return:
        """
        print('Classification')

        x = np.asarray(img_feat_vec, dtype=th.config.floatX)  # data_x[samp]

        pred, aux = self.net.tester(x)

        return pred, aux


    def run(self, input_tuple, test_set):
        """ TODO:
        This function takes a normalized image as Input. During predicting following steps are computed:
         1. Feature Extractor
         2. Neural Net

        :param input_tuple:
        :return:
        """
        # print "Input: ", input_tuple[0]
        print ("Image size: ", input_tuple[0].shape)
        # 1. Feature Extractor
        # feature_vec = FeatureExtractor(input_tuple[0])
        # 2. Neural Net
        if test_set == 0:
            cst, pred, aux = self.train_rnn(input_tuple[0], input_tuple[1])
            # cst, pred, aux = self.train_rnn(feature_vec, input_tuple[1])
        else:
            pred, aux = self.classify_rnn(input_tuple[0])
            # pred, aux = self.classify_rnn(feature_vec)
            cst = 0

        # self.show_all_tim(input_tuple[1], input_tuple[0], pred, (aux > 1e-20, 'Forward probabilities:'))
        # self.show_all_tim(input_tuple[1], feature_vec, pred, (aux > 1e-20, 'Forward probabilities:'))

        return [input_tuple[1], pred, cst,  aux]


    def save(self, directory):
        print ("Saving myPredictor to ", directory)
