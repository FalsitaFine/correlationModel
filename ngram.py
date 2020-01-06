import re
from nltk.util import ngrams


from gensim.models import Word2Vec
import multiprocessing
import os

input_dir = "./out_list/"
file_list = []
text = []
length = 3000
replace_list = [",","'",'"',"/","?",".","!",":"]


for files in os.listdir(input_dir):
	length = length - 1
	flag = 0
	filename = files
	#print(filename)
	if filename != ".DS_Store":
		filename = input_dir + str(filename)
		file_list.append(filename)
		loadf = open(filename,'r')
		loadfile = loadf.read()
		loadfile_sep = loadfile.split(".")
		for sep in loadfile_sep:
			sep_rep = sep
			for rep in replace_list:
				sep_rep = sep_rep.replace(rep,"")
			#loadfile = loadfile.lower
			#text.append(loadfile)
			text.append(sep_rep)
	if length < 0:
		break

#text = ["Drinking enough water is vital to health and good bodily functioning. However, some research suggests that the temperature of water when a person drinks it is also important. Here, we discuss whether cold water can be bad for health and if there are any risks or benefits of drinking cold water vs. warm water.","Previous studies from our lab have suggested that at least 50% of our metabolism is circadian, and 50% of the metabolites in our body oscillate based on the circadian cycle. It makes sense that exercise would be one of the things that's impacted, says Sassone-Corsi.","Gad Asher, who works in the Department of Biomolecular Sciences at the Weizmann Institute of Science in Rehovot, Israel, is senior author of the first study, while Paolo Sassone-Corsi of the Center for Epigenetics and Metabolism at the University of California (UC), Irvine, is senior author of the second."]

#text = ["Welcome to the planetarium where the stars are waiting for you", "Drink water is good for your health"]
sentences = []
n_grams = []
for line in text:
	sentences.append(line.split())
	s = line
	s = s.lower()
	s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
	tokens = [token for token in s.split(" ") if token != ""]
	n_grams.append(list(ngrams(tokens, 2)))


vocab_list = []
index_list = []




index = 0
for n in n_grams:
	print(n,n[0])
	if not n[0] in vocab_list:
		vocab_list.append(n[0])
		index_list.append(index)
		index += 1
	if not n[1] in vocab_list:
		vocab_list.append(n[1])
		index_list.append(index)
		index += 1

result_array = []
for i in len(vocab_list):
	result_array.append([])
	for j in len(vocab_list):
		result_array[i].append(0)

		
vocab_dic = dict(zip(vocab_list,index_list))


for i in n_grams:
	for j in n_grams:
		result_array[vocab_dic[j[1]]][vocab_dic[i[0]]] += 1

print(n_grams)