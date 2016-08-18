from validation_task import InputIteratorTask
import numpy as np


class IAM_InputIterator(InputIteratorTask):
	def run(self):
		print "====== IAM  Pipeline ======"
		inputs = [(1,2,3), (4,5,6), (6,7,8)]
		for i in inputs:
			yield i
			
	def __len__(self):
		fold_lens = map(lambda fold: len(fold), IAM_config.dataset)
		return reduce(lambda a,b: a+b, fold_lens)