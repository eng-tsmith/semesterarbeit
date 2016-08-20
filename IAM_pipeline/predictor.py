from validation_task import PredictorTask
import numpy as np


def FeatureExtractor(img):
    """

    :param img:
    :return:
    """
    features =[1,2,3]
    return features

def NeuralNet(features, labels):
    """

    :param features:
    :param labels:
    :return:
    """
    prediction = np.random.random((5, 5))
    return prediction

class IAM_Predictor(PredictorTask):
    def run(self, input_tuple):
        """ TODO:
        This function takes a normalized image as Input. During predicting following steps are computed:
         1. Feature Extractor
         2. Neural Net

        :param input_tuple:
        :return:
        """
        # print "Input: ", input_tuple[0]
        print "Image size: ", input_tuple[0].shape
        # 1. Feature Extractor
        feature_vec = FeatureExtractor(input_tuple)
        # 2. Neural Net
        pred = NeuralNet(feature_vec, input_tuple)

        return [pred, pred]


    def save(self, directory):
        print "Saving myPredictor to ", directory
