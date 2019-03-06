import os
with open('../val.txt', 'r') as f:
	for line in f:
		space = line.index(' ')
		x_train = line[:space].strip()
		y_train = line[space+1:].strip()
		#print(y_train)
		#print(os.path.exists(x_train), os.path.exists(y_train))
		if os.path.exists(x_train) and os.path.exists(y_train):
			print(x_train,y_train)
	print('done')