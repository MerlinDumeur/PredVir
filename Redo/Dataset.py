import pandas as pd

class Dataset:

	def __init__(self,base,nmois):

		self.base = base
		self.X = pd.read_pickle(base+rf'/X_classififcation{nmois}')
		self.Y = pd.read_pickle(base+rf'/Y_classification{nmois}')

	