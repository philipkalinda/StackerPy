import numpy as np
import pandas as pd

class ModelStacker():
	"""
	Model Stacker
	"""
	
	def __init__(self, models, stacker, blending=False):
		self.models = models
		self.stacker = stacker
		self.best_models = {}
		self.best_stacker = {}
		self.blending = blending
	
	def fit(X, Y):
		raise NotImplementedError()
	
	def predict(X):
		raise NotImplementedError()
		

class ExplorativeStacker():
	"""
	Explorative Model Stacker
	"""
	
	def __init__(self, n_models, blending=False, optimize=False):
		self.n_models = n_models
		self.stacker = None
		self.best_models = {}
		self.best_stacker = {}
		self.blending = blending
		self.optimize = optimize
	
	def fit(X, Y):
		raise NotImplementedError()
	
	def predict(X):
		raise NotImplementedError()
		
