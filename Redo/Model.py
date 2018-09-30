class Model:

	def __init__(self,model):

		# Model needs to implement hyperparameter fitting (similar to GridSearchCV)

		self.model = model

	def test_score(self,X,Y):

		