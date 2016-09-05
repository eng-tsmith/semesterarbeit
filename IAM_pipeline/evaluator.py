from validation_task import EvaluatorTask
import numpy as np


class IAM_Evaluator(EvaluatorTask):
    def run(self, input_tuple):
        """

        :param input_tuple: Predictions, Cost, Shown String, Seen String
        :return:
        """

        if input_tuple[2] == input_tuple[3]:
            print("Gleich")
            match = 1
        else:
            print("Nicht")
            match = 0

        return [match]

    def save(self, directory):
        print ("Saving myEvaluator to ", directory)
