from validation_task import PreprocessorTask
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import cv2 as cv


def show_img(img):
    plt.imshow(img)
    plt.show()


def greyscale(tupel_filenames):
    """

    :param img:
    :return:
    """
    # img = np.random.random((5,5))
    # img = misc.imread(tupel_filenames[0])
    img = cv.imread(tupel_filenames[0], cv.IMREAD_GRAYSCALE)

    label = "cat"
    return img, label


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
        :param input_tuple: [path to img_file, path to xml]
        :return output_tuple: [normalized image of text line, label]
        """
        print "Inputs: ", input_tuple
        # 1. Greyscale
        img_grey, label = greyscale(input_tuple)
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

        return [img_norm, label]

    def save(self, directory):
        print "Saving myPreprocessor to ", directory

