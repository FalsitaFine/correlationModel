readf = open("paired.txt",'r')
line = readf.readline()

word_list = []
file_list = []
index_list = []
index = 0

writef = open("corr_xpy.csv",'a')

while(line):
	line = line.replace('\n','')
	line_split = line.split("^")
	word_list.append(line_split[0])
	file_list.append(line_split[1:])
	if len(file_list) == 0:
		print(word_line)
	index_list.append(index)
	index = index + 1
	line = readf.readline()

first_line = ' / '
for word in word_list:
	first_line = first_line + ','+ word
first_line = first_line + "\n"
writef.write(first_line)


vocab_dic = dict(zip(word_list,index_list))
print(vocab_dic)
print(word_list)
#print(file_list[0])

for word_ref in word_list:
	line = word_ref
	for word_compare in word_list:
		common = list(set(file_list[vocab_dic[word_ref]]).intersection(file_list[vocab_dic[word_compare]]))
		#print()print()
		corr = len(common)/len(file_list[vocab_dic[word_ref]])
		if word_ref == word_compare:

			print(len(common),len(file_list[vocab_dic[word_ref]]),len(file_list[vocab_dic[word_compare]]),corr)
			print(word_ref,word_compare)
			print(common)
			print(file_list[vocab_dic[word_ref]])
			print("----")

		line = line + "," + str(corr)
	line = line + '\n'
	writef.write(line)


