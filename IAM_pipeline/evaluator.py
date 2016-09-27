from validation_task import EvaluatorTask
import numpy as np


class IAM_Evaluator(EvaluatorTask):
    def run(self, postprocessor_output, test_set):
        """

        :param postprocessor_output: [true, pred, loss, cer_abs, cer, wer, wer_lib]
        :param test_set:
        :return:
        """
        return postprocessor_output

    def save(self, directory):
        print ("Saving myEvaluator to ", directory)
