import os


file_dir = "./list/"
ref_dir = "./info/"

word_list = []
abd_list = [',','.','/','!',';','?',':','\n']

index = 0
index_list = []

cache_list = []


for files in os.listdir(ref_dir):
	filename = files
	print(filename)
	if filename != ".DS_Store":

		filename = ref_dir + str(filename)
		readf = open(filename,'r')
		line = readf.readline()
		while(line):
			#line_split = line.split(" ")
			line_split = line.split(" ")
			for word in line_split:
				#word = line
				word_purify = word
				for abd in abd_list:
					word_purify = word_purify.replace(abd,'')
				word_purify = word_purify.lower()
				#cache_list.append(word_purify)

				if not(word_purify in word_list):
					word_list.append(word_purify)
					index_list.append(index)
					index = index + 1

				line = readf.readline()


print(len(word_list))
#print(len(index_list))

vocab_dictionary = dict(zip(word_list,index_list))
#print(vocab_dictionary)
cache_list = []

for files in os.listdir(ref_dir):
	filename = files
	#print(filename)
	if filename != ".DS_Store":
		filename = ref_dir + str(filename)
		readf = open(filename,'r')
		line = readf.readline()
		whole_text = ''
		while(line):
			#print(line)
			word_purify = line
			for abd in abd_list:
				word_purify = word_purify.replace(abd,'')
			word_purify = word_purify.lower()
			whole_text = whole_text + word_purify + " "
			line = readf.readline()
		cache_list.append(whole_text)
		#print(whole_text)

count_list = []
for i in range(len(word_list)):
	count_list.append(0)
#print(len(count_list))
total_single = 0
total_multi = 0

index_monitor = 0
for cache in cache_list:
	print(index_monitor, "/", len(cache_list))
	for word in word_list:
		#print("searching...", word)
	#print(vocab_dictionary[cache],cache)
		word_split = word.split(" ")
		if word_split[0] == word:
			total_single = total_single + 1
			cache_split = cache.split(" ")
			if word_split[0] in cache_split:
				count_list[vocab_dictionary[word]] = count_list[vocab_dictionary[word]] + 1
		else:
			total_multi = total_multi + 1
			if word in cache:
				#print("find in:", cache)
				count_list[vocab_dictionary[word]] = count_list[vocab_dictionary[word]] + 1
	index_monitor = index_monitor + 1
		

	'''
	if "diabetes" in cache:
		print("find in:", cache)
		total = total + 1
print(total)
'''
#print(count_list)

vocab_count = dict(zip(word_list,count_list))
sorted_vocab = sorted(vocab_count.items(), key=lambda kv: kv[1], reverse = True)
print(sorted_vocab)
print(total_single,total_multi)

logf = open("log_file.log",'a')
for voc in sorted_vocab:
	logf.write(voc)

