from validation_task import ReporterTask
import numpy as np


class IAM_Reporter(ReporterTask):
    def run(self, input_tuple, eval_output):
        """

        :param input_tuple:
        :param match: was it a match?
        :return:
        """
        # if eval_output[0][-1] == 1:
        #     print("Die Klassifiierung war richtig: ", input_tuple[0])
        # else:
        #     print("Die Klassifiierung war falsch: ", input_tuple[0])
        # accuracy = np.sum(eval_output[0])/len(eval_output[0])
        print("Accuracy: ", eval_output[0], "Cost: ", eval_output[1])
        return True

    def save(self, directory):
        print ("Saving report to ", directory)
