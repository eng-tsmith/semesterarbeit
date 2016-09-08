from validation_task import EvaluatorTask
import numpy as np


class IAM_Evaluator(EvaluatorTask):
    def __init__(self):
        self.accuracy = []

    def run(self, postprocessor_output):
        """

        :param input_tuple: Original Data
               postprocessor_output: Predictions, Cost, Shown String, Seen String
        :return:
        """

        if postprocessor_output[2] == postprocessor_output[3]:
            print("Gleich")
            match = 1
        else:
            print("Nicht")
            match = 0

        # self.accuracy.append(match)

        # return [self.accuracy, postprocessor_output[1]]
        return [match, postprocessor_output[1]]

    def save(self, directory):
        print ("Saving myEvaluator to ", directory)
