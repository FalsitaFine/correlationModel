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
for line in text:
	sentences.append(line.split())

EMB_DIM = 500
print(sentences)

#model = Word2Vec.load("point2.model")

model = Word2Vec(sentences, size=EMB_DIM, window=20, min_count=3, iter=100, sg=1,workers=multiprocessing.cpu_count())
#model.train(sentences,total_examples = len(sentences),epochs = 10)

model.save("point2_cbow.model")


word_vectors = model.wv

result = word_vectors.similar_by_word("cancer")
print(result)