from validation_task import PredictorTask
# import pickle
# import sys
import numpy as np
import theano as th
from rnn_ctc.nnet.neuralnet import NeuralNet
import rnn_ctc.utils as utils


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
        # th.config.optimizer = 'fast_compile'
        # th.config.exception_verbosity='high'

        ################################### Main Script ###########################

        print('Initializing IAM Predictor. Loading   net_config.ast:')
        # with open('../rnn_ctc/configs/default', 'rb') as pkl_file:  # TODO which file?
        #     data = pickle.load(pkl_file)

        self.args = utils.read_args(['net_config.ast'])  # TODO reads ast file
        self.num_epochs, self.train_on_fraction = self.args['num_epochs'], self.args['train_on_fraction']
        self.scribe_args, self.nnet_args, = self.args['scribe_args'], self.args['nnet_args'],

        self.chars = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~'] # data['chars']
        self.num_classes = len(self.chars)
        self.img_ht = 9  # TODO
        self.num_samples = 1  # TODO
        self.nTrainSamples = int(self.num_samples * self.train_on_fraction)
        printer = utils.Printer(self.chars)

        print('\nNetwork Info:'
              '\nInput Dim: {}'
              '\nNum Classes: {}'
              '\nNum Samples: {}'
              '\nNum Epochs: {}'
              '\nFloatX: {}'
              '\n'.format(self.img_ht, self.num_classes, self.num_samples, self.num_epochs, th.config.floatX))

        ################################
        print('Building the Network')
        self.net = NeuralNet(self.img_ht, self.num_classes, **self.nnet_args)
        print(self.net)
        self.printer = utils.Printer(self.chars)
        ################################
        # print('Preparing the Data')
        # try:
        #     self.conv_sz = self.nnet_args['midlayerargs']['conv_sz']
        # except KeyError:
        #     self.conv_sz = 1
        #
        # data_x, data_y = [], []
        # bad_data = False
        #
        # for x, y in zip(data['x'], data['y']):
        #     # Insert blanks at alternate locations in the labelling (blank is num_classes)
        #     y1 = utils.insert_blanks(y, num_classes)
        #     data_y.append(np.asarray(y1, dtype=np.int32))
        #     data_x.append(np.asarray(x, dtype=th.config.floatX))
        #
        #     if printer.ylen(y1) > (1 + len(x[0])) // conv_sz:
        #         bad_data = True
        #         printer.show_all(y1, x, None, (x[:, ::conv_sz], 'Squissed'))


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

        # print(self.net)

        return cst, pred, aux

        # OLD:
        # print('Training the Network')
        # for epoch in range(self.num_epochs):
        #     print('Epoch : ', epoch)
        #     for samp in range(self.num_samples):
        #         x = features  # data_x[samp]
        #         y = labels    # data_y[samp]
        #         # if not samp % 500:            print(samp)
        #
        #         if samp < self.nTrainSamples:
        #             if len(y) < 2:
        #                 continue
        #
        #             cst, pred, aux = self.net.trainer(x, y)
        #
        #             if (epoch % 10 == 0 and samp < 3) or np.isinf(cst):
        #                 print('\n## TRAIN cost: ', np.round(cst, 3))
        #                 utils.printer.show_all(y, x, pred, (aux > 1e-20, 'Forward probabilities:'))
        #
        #             if np.isinf(cst):
        #                 print('Exiting on account of Inf Cost...')
        #                 sys.exit()
        #
        #         elif (epoch % 10 == 0 and samp - self.nTrainSamples < 3) \
        #                 or epoch == self.num_epochs - 1:
        #             # Print some test images
        #             pred, aux = self.net.tester(x)
        #             aux = (aux + 1) / 2.0
        #             print('\n## TEST')
        #             utils.printer.show_all(y, x, pred, (aux, 'Hidden Layer:'))
        #
        # print(self.net)
        #
        # prediction = np.random.random((5, 5))
        # return prediction


    def classify_rnn(self):
        """
        """
        pass


    def run(self, input_tuple):
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
        feature_vec = FeatureExtractor(input_tuple[0])
        # 2. Neural Net
        cst, pred, aux = self.train_rnn(feature_vec, input_tuple[1])

        #self.printer.show_all(input_tuple[1], feature_vec, pred, (aux > 1e-20, 'Forward probabilities:'))

        return [input_tuple[1], pred, cst,  aux]


    def save(self, directory):
        print ("Saving myPredictor to ", directory)
