from validation_task import ReporterTask
import numpy as np


class IAM_Reporter(ReporterTask):
    def run(self, input_tuple):
        print("Die Klassifiierung war richtig: ", input_tuple[0])

        return True

    def save(self, directory):
        print ("Saving report to ", directory)
