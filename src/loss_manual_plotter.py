import matplotlib.pyplot as plt
losses = []
while True:
	x = input('Loss:')
	losses.append(float(x))
	for c,i in enumerate(losses):
		plt.scatter(c+1, i)
	plt.show()