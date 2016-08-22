from validation_task import EvaluatorTask
import numpy as np


class IAM_Evaluator(EvaluatorTask):
    def run(self, volumes):
        print ("Length: ", len(volumes))
        print ("Shape: ", volumes[0].shape)
        return [np.random.random((6, 6)), np.random.random((7, 7))]

    def save(self, directory):
        print ("Saving myEvaluator to ", directory)
