from validation_task import ReporterTask
import numpy as np


class IAM_Reporter(ReporterTask):
    def run(self, input_tuple, volumes):
        print "Length: ", len(volumes)
        print "Shape: ", volumes[0].shape
        print "Shape: ", volumes[1].shape
        return [np.random.random((6, 6)), np.random.random((7, 7))]

    def save(self, directory):
        print "Saving report to ", directory
