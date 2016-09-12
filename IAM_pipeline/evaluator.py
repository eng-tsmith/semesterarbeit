from validation_task import EvaluatorTask
import numpy as np


class IAM_Evaluator(EvaluatorTask):
    def __init__(self):
        self.accuracy_sum = 0.0
        self.accuracy_length = 0.0

        self.accuracy_sum_test = 0.0
        self.accuracy_length_test = 0.0

        self.start_testing = 0

    def run(self, postprocessor_output, test_set):
        """

        :param input_tuple: Original Data
               postprocessor_output: Predictions, Cost, Shown String, Seen String
        :return:
        """
        # Check if new testing starts
        if self.start_testing == 0 and test_set == 1:
            self.start_testing = 1
            self.accuracy_sum_test = 0.0
            self.accuracy_length_test = 0.0
        # Check if testing ends
        if self.start_testing == 1 and test_set == 0:
            self.start_testing = 0

        if test_set == 0:
            if postprocessor_output[2] == postprocessor_output[3]:
                print("Gleich")
                match = 1
                self.accuracy_sum=+1
            else:
                print("Nicht")
                match = 0
            self.accuracy_length =+1
            accuracy = self.accuracy_sum/self.accuracy_length

        else:
            if postprocessor_output[2] == postprocessor_output[3]:
                print("Gleich")
                match = 1
                self.accuracy_sum_test = +1
            else:
                print("Nicht")
                match = 0
            self.accuracy_length_test = +1
            accuracy = self.accuracy_sum_test / self.accuracy_length_test 

        return [match, accuracy]

    def save(self, directory):
        print ("Saving myEvaluator to ", directory)
