from gensim.models import Word2Vec
import gensim.models
import multiprocessing
import os


model = Word2Vec.load("point1.model")


word_vectors = model.wv

#result = word_vectors.similar_by_word("cancer")
#print(result)

test = ['we', 'could', 'predict' ,'whether' ,'or','not' ,'it' ,'would' ,'survive']
#test = ["cancer", "illness", 'coke', 'diabetes', "cough"]
#test = ['we', 'could', 'predict' ,'whether' ,'or' ,'not' ,'it' ,'would' ,'survive']

#test = ['health','cancer']
print(model.predict_output_word(test,topn =1))
print(model.doesnt_match(test))
print(gensim.models.word2vec.score_sg_pair(model,"we","could"))

'''
prediction_list = []
probability_list = []
pre_dict = {}

for word in test:
	similars = word_vectors.similar_by_word(word, topn = 50)
	for similar_word in similars:
		similar_ins = similar_word[0]
		if not similar_ins in prediction_list:
			prediction_list.append(similar_ins)
			probability_list.append(similar_word[1])
			pre_dict.update({similar_word[0]:similar_word[1]})
		else:
			pre_dict[similar_ins] += similar_word[1]
	#print(similars)

sorted_pre = sorted(pre_dict.items(), key=lambda kv: kv[1], reverse = True)

print(sorted_pre)
#print(prediction_list)
'''