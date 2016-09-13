from validation_task import PreprocessorTask
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import cv2 as cv
import xml.etree.ElementTree as ET
import rnn_ctc.utils as utils
import rnn_ctc.scribe.scribe as Scribe
from skimage import transform as tf
import IAM_pipeline.data_config as data_config


def label_preproc(label_string):
    """
    This function is supposed to prepare the label so that it fits the standard of the rnn_ctc network.
    It computes following steps:
    1. make list of integers out of string    e.g. [hallo] --> [8,1,12,12,15]
    2. insert empty class between every int   [8,1,12,12,15] --> [95,8,95,1,95,12,95,12,95,15]
    :param label_string: a string of the label
    :return: label_int: the string represented in integers
    """
    # print("True Label: ", label_string)

    # Use built in scriber from RNN-CTC
    args = utils.read_args(['net_config.ast'])
    scriber = Scribe.Scribe(**args['scribe_args'])
    alphabet_chars = scriber.alphabet.chars

    # Iterate through string to find integers
    label_int = []
    for letter in label_string:
        label_int.append(alphabet_chars.index(letter))

    # print("Int Label: ", label_int)

    return label_int

def show_img(img):
    """
    This function takes an image as input and displays it. It is used for testing stages of preprocessing
    :param img: an image import via opencv2
    """
    plt.imshow(img)
    plt.show()


def XML_load(filepath, filename):
    """
    This funtion is used for loading labels out of corresponding xml file
    :param filepath: location of xml file
    :param filename: the name of the current image
    :return: the label of the image
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    # for child in root.findall('./handwritten-part/'):
    #     if child.get('id') == filename:
    #         return child.get('text')
    for child in root.iter('line'):
        if child.get('id') == filename:
            return child.get('text')


def load(tupel_filenames):
    """
    This function returns the raw image and the corresponding label
    :param tupel_filenames:
    :return: img: raw image loaded out of file, label: corresponding label as string
    """
    # img = cv.imread(tupel_filenames[0], cv.IMREAD_GRAYSCALE)
    img = cv.imread(tupel_filenames[0])
    label = XML_load(tupel_filenames[1], tupel_filenames[2])

    return img, label


def greyscale(img):
    """
    Makes a greyscale image out of a normal image
    :param img: colored image
    :return: img_grey: greyscale image
    """
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    return img_grey


def thresholding(img_grey):
    """
    This functions creates binary images using thresholding
    :param img_grey: greyscale image
    :return: binary image
    """
    # # Adaptive Gaussian
    # img_binary = cv.adaptiveThreshold(img_grey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img_grey, (5, 5), 0)
    ret3, img_binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # invert black = 255
    ret, thresh1 = cv.threshold(img_binary, 127, 255, cv.THRESH_BINARY_INV)

    return thresh1


def skew(img):
    """
    This function detects skew in images. It turn the image so that the baseline of image is straight.
    :param img: the image
    :return: rotated image
    """
    # coordinates of bottom black pixel in every column
    black_pix = np.zeros((2, 1))

    # Look at image column wise and in every column from bottom to top pixel. It stores the location of the first black
    # pixel in every column
    for columns in range(img.shape[1]):
        for pixel in np.arange(img.shape[0]-1, -1, -1):
            if img[pixel][columns] == 255:
                black_pix = np.concatenate((black_pix, np.array([[pixel], [columns]])), axis=1)
                break

    # Calculate linear regression to detect baseline
    mean_x = np.mean(black_pix[1][:])
    mean_y = np.mean(black_pix[0][:])
    k = black_pix.shape[1]
    a = (np.sum(black_pix[1][:] * black_pix[0][:]) - k * mean_x * mean_y) / (np.sum(black_pix[1][:] * black_pix[1][:]) - k * mean_x * mean_x)

    # Calculate angle by looking at gradient of linear function
    angle = np.arctan(a) * 180 / np.pi

    # Rotate image and use Nearest Neighbour for interpolation of pixel
    rows, cols = img.shape
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv.warpAffine(img, M, (cols, rows), flags=cv.INTER_NEAREST)

    return img_rot


def slant(img):
    # Load the image as a matrix
    """

    :param img:
    :return:
    """
    # Create Afine transform
    afine_tf = tf.AffineTransform(shear=0.1)  #TODO which factor???

    # Apply transform to image data
    img_slanted = tf.warp(img, afine_tf, order=0)
    return img_slanted


def positioning(img):
    """

    :param img:
    :return:
    """
    return img


def scaling(img):
    """
    This function scale the image down so that height is exactly 40 pixel. Th width of every image may vary.
    :param img:
    :return: resized image
    """
    baseheight = data_config.img_ht
    hpercent = (baseheight / float(img.shape[0]))
    dim = (int(img.shape[1] * hpercent), 40)

    img_scaled = cv.resize(img, dim, interpolation=cv.INTER_NEAREST)

    return img_scaled


class IAM_Preprocessor(PreprocessorTask):
    def run(self, input_tuple):
        """ TODO:
        This function takes an image as Input. During Pre-Processing following steps are computed:
            1. Load image and label
            2. Greyscale
            3. Thresholding
            4. Skew
            5. Slant
            6. Positioning
            7. Scaling
            8. Preprocessing of label
        :param input_tuple: [path to img_file, path to xml]
        :return output_tuple: [normalized image of text line, label]
        """
        print ("Inputs: ", input_tuple)
        # 1. Load img and label
        img_raw, label_raw = load(input_tuple)

        # 2. Greyscale
        img_grey = greyscale(img_raw)

        # 3. Thresholding
        img_thresh = thresholding(img_grey)

        # 4. Skew
        img_skew = skew(img_thresh)

        # 5. Slant
        img_slant = slant(img_skew)

        # 6. Positioning
        img_pos = positioning(img_slant)

        # 7. Scaling
        img_norm = scaling(img_pos)

        # 8. Preprocessing of label
        label = label_preproc(label_raw)

        print("Preprocessing successful!")
        # show_img(img_norm)
        return [img_norm, label]

    def save(self, directory):
        print ("Saving myPreprocessor to ", directory)

