from validation_task import ReporterTask
import numpy as np


class IAM_Reporter(ReporterTask):
    def run(self, input_tuple, match):
        """

        :param input_tuple:
        :param match: was it a match?
        :return:
        """
        if match == 1:
            print("Die Klassifiierung war richtig: ", input_tuple[0])
        else:
            print("Die Klassifiierung war falsch: ", input_tuple[0])
        return True

    def save(self, directory):
        print ("Saving report to ", directory)
