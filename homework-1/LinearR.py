import numpy as np
from sklearn.metrics import mean_squared_error

class LinearReg:
	
	def mean(self, values):
		return values.mean()

	def covariance(self, x, mean_x, y, mean_y):
		return x.cov()

	def variance(self, values, mean):
		return values.var()

	def fit(self, x, y):
		x_mean, y_mean = self.mean(x), self.mean(y)
		self.b1 = self.covariance(x, x_mean, y, y_mean) / self.variance(x, x_mean)
		self.b0 = y_mean - self.b1 * x_mean
	
	def predict(self, x):
		return self.b0 + self.b1 * x

	def calculateRMSE(self, actual, predicted):
			sum_error = 0.0
			for i in range(len(actual)):
				prediction_error = predicted[i] - actual[i]
				sum_error += (prediction_error ** 2)
			mean_error = sum_error / float(len(actual))
			return sqrt(mean_error)