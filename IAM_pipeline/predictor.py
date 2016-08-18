from validation_task import PredictorTask
import numpy as np

class IAM_Predictor(PredictorTask):
	def run(self, volumes):
		print "Length: ", len(volumes)
		print "Shape: ", volumes[0].shape
		print "Shape: ", volumes[1].shape
		print "Shape: ", volumes[2].shape
		return [np.random.random((5, 5)), np.random.random((6, 6))]


	def save(self, directory):
		print "Saving myPredictor to ", directory