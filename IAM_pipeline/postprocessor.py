from validation_task import PostprocessorTask
import numpy as np


class IAM_Postprocessor(PostprocessorTask):
    def run(self, input_tuple):
        """

        :param input_tuple:  [true, pred, loss, cer_abs, cer, wer, wer_lib]
        :return:
        """
        return input_tuple

    def save(self, directory):
        print ("Saving myPostprocessor to ", directory)
