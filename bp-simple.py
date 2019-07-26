# 计算激活函数的导数
def transfer_derivative(output):
	return output * (1 - output)

# 反向传播误差信息，并存储在神经元中
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['responsibility'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['responsibility'] = errors[j] * transfer_derivative(neuron['output'])

def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i!= 0:
			inputs = [neuron['output'] for neuron in network[i-1]]
			for neuron in network[i]:
				for j in range(len(inputs)):
					neuron['weights'][j] += l_rate * neuron['responsibility'] * inputs[j]
				neuron['weights'][-1] += l_rate * neuron['responsibility']

if __name__ == '__main__':
	#测试反向传播
	network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614], 'output': 0.7105668883115941}, \
	            {'weights': [0.2550690257394217, 0.49543508709194095, 0.4494910647887381], 'output': 0.6691980263750579}], \
	            [{'weights': [0.651592972722763, 0.7887233511355132, 0.0938595867742349], 'output': 0.7473771139195361}, \
	            {'weights': [0.02834747652200631, 0.8357651039198697, 0.43276706790505337], 'output': 0.733450902916955}]]
	expected = [0, 1]
	backward_propagate_error(network, expected)
	for layer in network:
		print(layer)