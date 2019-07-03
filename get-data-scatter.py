import pandas as pd

if __name__ == '__main__':
	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
	print(df.tail())