from validation_task import PredictorTask
import sys
import numpy as np
import IAM_pipeline.data_config as data_config
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional


class IAM_Predictor(PredictorTask):

    def __init__(self):
        """
        When this funtion is first called it initalizes the net.
        """
        self.max_features = 9
        self.maxlen = 150  # cut texts after this number of words (among top max_features most common words)
        self.batch_size = 1

        self.model = Sequential()
        self.model.add(Embedding(self.max_features, 128, input_length=self.maxlen))
        self.model.add(Bidirectional(LSTM(64)))
        self.model.add(Dense(1, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        self.model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])


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
        # print "Input: ", input_tuple[0]
        print ("Image size: ", input_tuple[0].shape)
        # 1. Feature Extractor
        feature_vec = self.FeatureExtractor(input_tuple[0])

        # 2. Neural Net
        if test_set == 0:
            loss, acc = self.train_rnn(feature_vec, input_tuple[1])
            # cst, pred = self.train_rnn(feature_vec, input_tuple[1])
        else:
            loss, acc = self.test_rnn(feature_vec, input_tuple[1])
            # pred = self.classify_rnn(feature_vec)

        return [input_tuple[1], loss, acc]  #TODO DIFFERNET OUTPUT

    def save(self, directory):
        print ("Saving myPredictor to ", directory)
