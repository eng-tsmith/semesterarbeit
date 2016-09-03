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
    features =[1,2,3]
    return features



class IAM_Predictor(PredictorTask):

    def __init__(self):
        # th.config.optimizer = 'fast_compile'
        # th.config.exception_verbosity='high'

        ################################### Main Script ###########################
        print('Initializing IAM Predictor. Loading   net_config.ast:')
        # with open('../rnn_ctc/configs/default', 'rb') as pkl_file:  # TODO which file?
        #     data = pickle.load(pkl_file)

        args = utils.read_args(['net_config.ast']) # TODO reads ast file
        num_epochs, train_on_fraction = args['num_epochs'], args['train_on_fraction']
        scribe_args, nnet_args, = args['scribe_args'], args['nnet_args'],

        chars = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~'] # data['chars']
        num_classes = len(chars)
        img_ht = 20 # TODO
        num_samples = 1 # TODO
        nTrainSamples = int(num_samples * train_on_fraction)
        printer = utils.Printer(chars)

        print('\nNetwork Info:'
              '\nInput Dim: {}'
              '\nNum Classes: {}'
              '\nNum Samples: {}'
              '\nNum Epochs: {}'
              '\nFloatX: {}'
              '\n'.format(img_ht, num_classes, num_samples, num_epochs, th.config.floatX))

        ################################
        print('Building the Network')
        self.net = NeuralNet(img_ht, num_classes, **nnet_args)
        print(self.net)
        ################################
        print('Preparing the Data')
        try:
            conv_sz = nnet_args['midlayerargs']['conv_sz']
        except KeyError:
            conv_sz = 1

        data_x, data_y = [], []
        bad_data = False

        for x, y in zip(data['x'], data['y']):
            # Insert blanks at alternate locations in the labelling (blank is num_classes)
            y1 = utils.insert_blanks(y, num_classes)
            data_y.append(np.asarray(y1, dtype=np.int32))
            data_x.append(np.asarray(x, dtype=th.config.floatX))

            if printer.ylen(y1) > (1 + len(x[0])) // conv_sz:
                bad_data = True
                printer.show_all(y1, x, None, (x[:, ::conv_sz], 'Squissed'))


    def train_rnn(self, features, labels):
        # """
        #
        # :param features:
        # :param labels:
        # :return:
        # """
        # print('Training the Network')
        # for epoch in range(num_epochs):
        #     print('Epoch : ', epoch)
        #     for samp in range(num_samples):
        #         x = features  # data_x[samp]
        #         y = labels    # data_y[samp]
        #         # if not samp % 500:            print(samp)
        #
        #         if samp < nTrainSamples:
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
        #         elif (epoch % 10 == 0 and samp - nTrainSamples < 3) \
        #                 or epoch == num_epochs - 1:
        #             # Print some test images
        #             pred, aux = self.net.tester(x)
        #             aux = (aux + 1) / 2.0
        #             print('\n## TEST')
        #             utils.printer.show_all(y, x, pred, (aux, 'Hidden Layer:'))
        #
        # print(self.net)

        prediction = np.random.random((5, 5))
        return prediction


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
        feature_vec = FeatureExtractor(input_tuple)
        # 2. Neural Net
        pred = self.train_rnn(feature_vec, input_tuple)

        return [pred, pred]


    def save(self, directory):
        print ("Saving myPredictor to ", directory)
