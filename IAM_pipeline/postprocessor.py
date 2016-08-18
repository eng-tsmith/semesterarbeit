from validation_task import PostprocessorTask
import numpy as np

class IAM_Postprocessor(PostprocessorTask):
	def run(self, volumes):
		print "Length: ", len(volumes)
		print "Shape: ", volumes[0].shape
		print "Shape: ", volumes[1].shape
		return [np.random.random((6, 6))]


	def save(self, directory):
		print "Saving myPostprocessor to ", directory