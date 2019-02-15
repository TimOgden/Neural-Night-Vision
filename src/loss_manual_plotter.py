import matplotlib.pyplot as plt
import pickle
losses = []
while True:
	x = input('Loss:')
	if x=='show':
		plt.plot(losses)
		plt.show()
	if x=='remove':
		losses.remove(losses[-1])
	else:
		losses.append(float(x))
