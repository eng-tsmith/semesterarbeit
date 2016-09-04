from validation_task import PostprocessorTask
import numpy as np


class IAM_Postprocessor(PostprocessorTask):
    def run(self, input_tuple):
        print ("Cost: ", input_tuple[0])
        print ("Pred: ", input_tuple[1])
        print ("Aux: ", input_tuple[2])

        return [np.random.random((6, 6))]


    def save(self, directory):
        print ("Saving myPostprocessor to ", directory)
