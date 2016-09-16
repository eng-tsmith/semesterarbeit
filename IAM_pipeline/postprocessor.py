from validation_task import PostprocessorTask
import numpy as np


class Printer():
    def __init__(self):
        """
        Creates a function that can print a predicted output of the CTC RNN
        It removes the blank characters (need to be set to n_classes),
        It also removes duplicates
        :param list chars: list of characters
        """
        self.chars = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3',
                 '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[',
                 '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~'] + ['blank']
        self.n_classes = len(self.chars) - 1

    def yprint(self, labels):
        labels_out = []
        for il, l in enumerate(labels):
            if (l != self.n_classes) and (il == 0 or l != labels[il-1]):
                labels_out.append(l)
        labels_list = [self.chars[l] for l in labels_out]
        # print(labels_out, ' '.join(labels_list))
        return labels_out, labels_list




class IAM_Postprocessor(PostprocessorTask):
    def run(self, input_tuple):

        """

        :param input_tuple:  [label, loss, acc]
        :return:
        """
        print("label: ", input_tuple[0])
        print("loss: ", input_tuple[1])
        print("metric: ", input_tuple[2])

        # printer = Printer()
        #
        # labels_out_true, labels_list_true = printer.yprint(input_tuple[0])
        #
        # maxes = np.argmax(input_tuple[1], 0)
        # labels_out_pred, labels_list_pred = printer.yprint(maxes)

        # print('Shown String: ', labels_list_true)
        # print('Seen String: ', labels_list_pred)

        # Predictions, Cost, Shown String, Seen String
        return input_tuple


    def save(self, directory):
        print ("Saving myPostprocessor to ", directory)
