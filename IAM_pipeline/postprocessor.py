from validation_task import PostprocessorTask
import numpy as np
import rnn_ctc.utils as utils


class IAM_Postprocessor(PostprocessorTask):
    def run(self, input_tuple):

        print("True: ", input_tuple[0])
        print("Pred: ", input_tuple[1].shape)
        print("Cost: ", input_tuple[2])
        print("Aux: ", input_tuple[3].shape)

        chars = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3',
                 '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[',
                 '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']
        printer = utils.Printer(chars)

        print('Shown : ', end='')
        labels_out_true, labels_list_true = printer.yprint(input_tuple[0])

        print('Seen  : ', end='')
        maxes = np.argmax(input_tuple[1], 0)
        labels_out_pred, labels_list_pred = printer.yprint(maxes)

        print('True String: ', labels_out_true)
        print('Pred String: ', labels_out_pred)

        return [np.random.random((6, 6))]


    def save(self, directory):
        print ("Saving myPostprocessor to ", directory)
