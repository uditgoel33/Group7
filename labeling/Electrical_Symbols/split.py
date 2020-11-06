import os 

path = os.getcwd()
'''
for i in os.listdir(path):
	if i.endswith(".txt"):
		with open(i,'r+') as f :
			for j in f.readlines():
				line = "0" + j[2:]
				print(line)
				f.write(line)
'''

with open("electrical_train_2.txt", "w") as  f:
	for i in os.listdir(path):
		if i.endswith("png"):
			f.write("./text_labels/" + i + "\n")
