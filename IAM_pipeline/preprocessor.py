from validation_task import PreprocessorTask
import numpy as np

def greyscale(img):
    """

    :param img:
    :return:
    """
    return img


def thresholding(img):
    """

    :param img:
    :return:
    """
    return img


def skew(img):
    """

    :param img:
    :return:
    """
    return img


def slant(img):
    """

    :param img:
    :return:
    """
    return img


def positioning(img):
    """

    :param img:
    :return:
    """
    return img


def scaling(img):
    """

    :param img:
    :return:
    """
    return img


class IAM_Preprocessor(PreprocessorTask):
    def run(self, input_tuple):
        """ TODO:
        This function takes an image as Input. During Pre-Processing following steps are computed:
            1. Greyscale
            2. Thresholding
            3. Skew
            4. Slant
            5. Positioning
            6. Scaling
        :param input_tuple: unedited full Image of text line
        :return: normalized image of text line
        """
        print "Inputs: ", input_tuple
        # 1. Greyscale
        img_grey = greyscale(input_tuple)
        # 2. Thresholding
        img_thresh = thresholding(img_grey)
        # 3. Skew
        img_skew = skew(img_thresh)
        # 4. Slant
        img_slant = slant(img_skew)
        # 5. Positioning
        img_pos = positioning(img_slant)
        # 6. Scaling
        img_norm = scaling(img_pos)

        return [img_norm,img_norm,img_norm]

    def save(self, directory):
        print "Saving myPreprocessor to ", directory

