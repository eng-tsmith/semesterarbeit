from validation_task import PreprocessorTask
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import cv2 as cv
import xml.etree.ElementTree as ET
import rnn_ctc.utils as utils
import rnn_ctc.scribe.scribe as Scribe


def label_preproc(label_string):
    """
    This function is supposed to prepare the label so that it fits the standard of the rnn_ctc network.
    It computes following steps:
    1. make list of integers out of string    e.g. [hallo] --> [8,1,12,12,15]
    2. insert empty class between every int   [8,1,12,12,15] --> [95,8,95,1,95,12,95,12,95,15]
    :param label_string:
    :return:
    """
    print("True Label: ", label_string)

    args = utils.read_args(['net_config.ast'])
    scriber = Scribe.Scribe(**args['scribe_args'])
    alphabet_chars = scriber.alphabet.chars

    label_int = []
    for letter in label_string:
        label_int.append(alphabet_chars.index(letter))

    # print("Int Label: ", label_int)

    return label_int

def show_img(img):
    """
    This function takes an image as input and displays it
    :param img:
    """
    plt.imshow(img)
    plt.show()


def XML_load(filepath, filename):  # TODO: think about this again
    """
    This funtion is used for loading labels out of corresponding xml file
    :param filepath:
    :param filename:
    :return:
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    for line in root.findall('./handwritten-part/'):
        if line.get('id') == filename:
            #  print line.get('text')
            return line.get('text')


def load(tupel_filenames):
    """
    This function ist used for loading images
    :param tupel_filenames:
    :return:
    """
    img = cv.imread(tupel_filenames[0], cv.IMREAD_GRAYSCALE)  # TODO: np float 32 ? img.astype(float)
    label = XML_load(tupel_filenames[1], tupel_filenames[2])

    return img, label


def greyscale(img):
    """
    Makes a greyscale image out of a normal images
    :param img:
    :return:
    """
    # img_grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)# TODO: np float 32 ? img.astype(float)

    return img


def thresholding(img_grey):
    """

    :param img:
    :return:
    """
    # # Adaptive Gaussian
    # img_binary = cv.adaptiveThreshold(img_grey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img_grey, (5, 5), 0)
    ret3, img_binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    ret, thresh1 = cv.threshold(img_binary, 127, 255, cv.THRESH_BINARY_INV)

    # show_img(img_grey)
    # show_img(img_binary)

    return thresh1


def skew(img):
    """

    :param img:
    :return:
    """
    black_pix = np.zeros((2, 1))

    for columns in range(img.shape[1]):
        for pixel in np.arange(img.shape[0]-1, -1, -1):
        # for pixel in np.arange(img.shape[0]):
            if img[pixel][columns] == 255:
                black_pix = np.concatenate((black_pix, np.array([[pixel], [columns]])), axis=1)
                break

    mean_x = np.mean(black_pix[1][:])
    mean_y = np.mean(black_pix[0][:])
    k = black_pix.shape[1]

    a = (np.sum(black_pix[1][:] * black_pix[0][:]) - k * mean_x * mean_y) / (np.sum(black_pix[1][:] * black_pix[1][:]) - k * mean_x * mean_x)

    angle = np.arctan(a) * 180 / np.pi
    print (angle)

    rows, cols = img.shape

    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv.warpAffine(img, M, (cols, rows), flags=cv.INTER_NEAREST)

    # show_img(img)
    # show_img(img_rot)

    return img_rot


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
    baseheight = 40.0
    hpercent = (baseheight / float(img.shape[0]))
    dim = (int(img.shape[1] * hpercent), 40)

    img_scaled = cv.resize(img, dim, interpolation=cv.INTER_NEAREST)

    return img_scaled


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
        print ("Inputs: ", input_tuple)
        print ("imagesize",)
        # 0. Load img and label
        img_raw, label_raw = load(input_tuple)
        # 1. Greyscale
        print("imagesize", img_raw.shape)
        img_grey = greyscale(img_raw)
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
        # 7. Preprocessing of label
        label = label_preproc(label_raw)

        # show_img(img_norm)
        # print(img_norm)
        return [img_norm, label]

    def save(self, directory):
        print ("Saving myPreprocessor to ", directory)

