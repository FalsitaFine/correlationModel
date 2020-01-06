from gensim.models import Word2Vec
import multiprocessing

text = ["Drinking enough water is vital to health and good bodily functioning. However, some research suggests that the temperature of water when a person drinks it is also important. Here, we discuss whether cold water can be bad for health and if there are any risks or benefits of drinking cold water vs. warm water.","Previous studies from our lab have suggested that at least 50% of our metabolism is circadian, and 50% of the metabolites in our body oscillate based on the circadian cycle. It makes sense that exercise would be one of the things that's impacted, says Sassone-Corsi."]

sentences = []
for line in text:
	sentences.append(line.split())

EMB_DIM = 300

w2v = Word2Vec(sentences.size=EMB_DIM, window=5, min_count=5, negative=15, iter=10, workers=multiprocessing.cpu_count())

word_vectors = w2v.wv

result = word_vectors.similar_by_word("good")
print(result)