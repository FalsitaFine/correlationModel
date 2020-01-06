import os

file_dir = "./info/"



readpre = open("log.txt",'r')
data = readpre.readline()
data_split = data.split("), ")
abd_list = ["'",'(',")","[",",",'1','2','3','4','5','6','7','8','9','0']

text_abd_list = [',','.','/','!',';','?',':','\n']

#abd_list = ["'",'(',")","[",",",' ']

common_list = []
commonword_list = ['of','with','at','from','into','during','including','until','against','among','throughout','despite','towards','upon','concerning','to','in','for','on','by','about','like','through','over','before','between','after','since','without','under','within','along','following','across','behind','beyond','plus','except','but','up','out','around','down','off','above','near','the','a','an']


file_dir = "./info/"
 
total_count = 300
for word in data_split:
	word_purify = word
	for abd in abd_list:
		word_purify = word_purify.replace(abd,"")
	word_purify = word_purify[:-1]
	if not word_purify in commonword_list: 
		total_count = total_count - 1
		common_list.append(word_purify)
	if total_count == 0:
		break
print(common_list)


#for word in common_list:
	#print("searching for... ",word)

index_list = []
pair_list = [[]]

for i in range(300):
	index_list.append(i)
	pair_list.append([])

print(index_list)
vocab_dictionary = dict(zip(common_list,index_list))

print(vocab_dictionary)


for files in os.listdir(file_dir):
	filename = files
	print(filename)
	
	if filename != ".DS_Store":

		filename = file_dir + str(filename)
		readf = open(filename,'r')
		line = readf.readline()
		while(line):
			line_purify = line
			for abd in text_abd_list:
				line_purify = line_purify.replace(abd,'')
			line_purify = line_purify.lower()
			for word_search in common_list:
				if word_search in line_purify:
					if not(filename in pair_list[vocab_dictionary[word_search]]):
						pair_list[vocab_dictionary[word_search]].append(filename)
			line = readf.readline()
#print(pair_list)

paired_file = "./paired_x.txt"
writef = open(paired_file,'a')

for i in range(300):
	print(common_list[i],len(pair_list[i]))
	info = common_list[i]
	for file in pair_list[i]:
		info = info + "^" + file
	info = info + "\n" 
	writef.write(info)

