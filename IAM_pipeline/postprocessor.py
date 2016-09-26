from validation_task import PostprocessorTask
import numpy as np


class IAM_Postprocessor(PostprocessorTask):
    def run(self, input_tuple):
        """

        :param input_tuple:  [label, pred, loss, cer, wer]
        :return:
        """
        return input_tuple

    def save(self, directory):
        print ("Saving myPostprocessor to ", directory)
