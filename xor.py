# preceptron


def relu(input_value):
	return 0.0 if input_value < 0.0 else 1.0



def _init_params():
	params = [0.0 for _ in range(3)]

	return params


def forward_prop(row_vec, params):
	act_values = 0.0

	for i in range(len(params)):
		act_values += params[i] * row_vec[i]

	return relu(act_values)


def _update_weights(input_vec_label, prediction, params, rate):
	# 根据预测的输出值调整参数。
	# 计算损失值
	delta = input_vec_label[-1] - prediction

	# 根据损失值调整参数，学习率为rate
	for i in range(len(params)):
		params[i] += rate * delta * input_vec_label[i]


def train(dataset, iteration, rate, params):
	# 训练函数，输入x1, x2和期望的输出值。
    # 设置训练的次数和学习率。
    for i in range(iteration):
    	for input_vec_label in dataset:
    		prediction = forward_prop(input_vec_label, params)
    		_update_weights(input_vec_label, prediction, params, rate)


def get_training_dataset():
	# 输入的序列和期望输出值。
    # [偏置值，x1, x2, y]

    # dataset for AND
    # dataset = [[-1, 1, 1, 1], [-1, 0, 0, 0], [-1, 1, 0, 0], [-1, 0, 1, 0]]

    # dataset for OR
    #dataset = [[-1, 1, 1, 1], [-1, 0, 0, 0], [-1, 1, 0, 1], [-1, 0, 1, 1]]
    
    # dataset for XOR
    # 模型无法收敛，训练失败
    dataset = [[-1, 1, 1, 0], [-1, 0, 0, 0], [-1, 1, 0, 1], [-1, 0, 1, 1]]

    return dataset


if __name__ == '__main__':

	TRAIN_ITERATION = 100
	TRAIN_RATE = 0.01

	W1_hist = []
	W2_hist = []
	b_hist = []
	loss_hist = []
	params = _init_params()
	dataset = get_training_dataset()
	train(dataset, TRAIN_ITERATION, TRAIN_RATE, params)

	print("[W1 -- W2 -- bais]")
	print(params)

	# 模型预测结果输出
	print('input [0, 0] = output %d' % forward_prop([-1, 0, 0], params))
	print('input [0, 1] = output %d' % forward_prop([-1, 0, 1], params))
	print('input [1, 0] = output %d' % forward_prop([-1, 1, 0], params))
	print('input [1, 1] = output %d' % forward_prop([-1, 1, 1], params))
	


















