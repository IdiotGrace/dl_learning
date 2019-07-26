class Perceptron(object):

	def __init__(self, input_para_num, acti_func):
		self.activator = acti_func
		# 权重向量初始化为0
		self.weights = [0.0 for _ in range(input_para_num)]

	def __str__(self):
		return 'final weights\n\tw0 = {:.2f}\n\tw1 = {:.2f}\n\tw2 = {:.2f}'.format(self.weights[0], self.weights[1], self.weights[2])

	def predict(self, row_vec):
		act_values = 0.0
		for i in range(len(self.weights)):
			act_values += self.weights[i] * row_vec[i]
		return self.activator(act_values)

	def __update__weights(self, input_vec_lable, prediction, rate):
		delta = input_vec_lable[-1] - prediction
		for i in range(len(self.weights)):
			self.weights[i] += rate * delta * input_vec_lable[i]

	def train(self, dataset, iteration, rate):
		for i in range(iteration):
			for input_vec_lable in dataset:
				# 计算感知机再当前权重下的输出
				prediction = self.predict(input_vec_lable)
				# 更新权重
				self.__update__weights(input_vec_lable, prediction, rate)

# 定义激活函数
def func_activator(input_value):
	return 1.0 if input_value >= 0.0 else 0.0

def get_training_dataset():
	# 构建训练数据
	dataset = [[-1, 1, 1, 1], [-1, 0, 0, 0], [-1, 1, 0, 0], [-1, 0, 1, 0]]
	# 期望的输出列表，注意要与输入一一对应
	# [-1, 1, 1] -> 1, [-1, 0, 0] -> 0, [-1, 1, 0] -> 0, [-1, 0, 1] -> 0
	return dataset

def train_and_perceptron():
	p = Perceptron(3, func_activator)
	#获取训练数据
	dataset = get_training_dataset()
	p.train(dataset, 100, 0.01) # 迭代次数100次， 学习率为0.01
	# 返回训练好的感知机
	return p

if __name__ == '__main__':
	# 训练and感知机
	and_perceptron = train_and_perceptron()
	# 打印训练获得的权重
	print(and_perceptron)

	# testing
	print('1 and 1 = %d' % and_perceptron.predict([-1, 1, 1]))
	print('0 and 0 = %d' % and_perceptron.predict([-1, 0, 0]))
	print('1 and 0 = %d' % and_perceptron.predict([-1, 1, 0]))
	print('0 and 1 = %d' % and_perceptron.predict([-1, 0, 1]))


