from validation_task import PreprocessorTask
import numpy as np

class IAM_Preprocessor(PreprocessorTask):
	def run(self, input_tuple):
		print "Inputs: ", input_tuple
		return [np.random.random((4, 4)), np.random.random((5, 5)), np.random.random((5, 5))]


	def save(self, directory):
		print "Saving myPreprocessor to ", directory

