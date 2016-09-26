from validation_task import EvaluatorTask
import numpy as np


class IAM_Evaluator(EvaluatorTask):
    def run(self, postprocessor_output, test_set):
        """

        :param postprocessor_output: [label, pred, loss, cer, wer]
        :param test_set:
        :return:
        """
        return postprocessor_output

    def save(self, directory):
        print ("Saving myEvaluator to ", directory)
