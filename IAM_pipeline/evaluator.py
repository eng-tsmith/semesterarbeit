from validation_task import EvaluatorTask
import numpy as np


class IAM_Evaluator(EvaluatorTask):
    def run(self, input_tuple, evaluator_output):
        """

        :param input_tuple: Predictions, Cost, Shown String, Seen String
        :return:
        """

        if evaluator_output[2] == evaluator_output[3]:
            print("Gleich")
            match = 1
        else:
            print("Nicht")
            match = 0

        return [match]

    def save(self, directory):
        print ("Saving myEvaluator to ", directory)
