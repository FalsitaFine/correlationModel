import torch
import torch.nn as nn
import numpy as np
from gensim.models import Word2Vec
import multiprocessing
import os
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

def get_index_of_max(input):
    index = 0
    for i in range(1, len(input)):
        if input[i] > input[index]:
            index = i 
    return index

def get_max_prob_result(input, ix_to_word):
    return ix_to_word[get_index_of_max(input)]


CONTEXT_SIZE = 8  # 2 words to the left, 2 to the right
EMDEDDING_DIM = 100

word_to_ix = {}
ix_to_word = {}

raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()



input_dir = "./out_list/"
file_list = []
text = []
length = 100
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
        for rep in replace_list:
            loadfile = loadfile.replace(rep,"")
        text.append(loadfile)
        #text.append(sep_rep)
    if length < 0:
        break

#text = ["Drinking enough water is vital to health and good bodily functioning. However, some research suggests that the temperature of water when a person drinks it is also important. Here, we discuss whether cold water can be bad for health and if there are any risks or benefits of drinking cold water vs. warm water.","Previous studies from our lab have suggested that at least 50% of our metabolism is circadian, and 50% of the metabolites in our body oscillate based on the circadian cycle. It makes sense that exercise would be one of the things that's impacted, says Sassone-Corsi.","Gad Asher, who works in the Department of Biomolecular Sciences at the Weizmann Institute of Science in Rehovot, Israel, is senior author of the first study, while Paolo Sassone-Corsi of the Center for Epigenetics and Metabolism at the University of California (UC), Irvine, is senior author of the second."]

#text = ["Welcome to the planetarium where the stars are waiting for you", "Drink water is good for your health"]
sentences = []
raw_text = []
for line in text:
    sentences.append(line.split())
    for word in line.split():
        raw_text.append(word)




# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

for i, word in enumerate(vocab):
    word_to_ix[word] = i
    ix_to_word[i] = word

data = []

'''
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
'''
for raw_text in sentences:
    print(raw_text)
    for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
        context = [raw_text[i - 8], raw_text[i - 7],raw_text[i - 6], raw_text[i - 5],raw_text[i - 4], raw_text[i - 3],raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2],raw_text[i + 3], raw_text[i + 4],raw_text[i + 5], raw_text[i + 6],raw_text[i + 7], raw_text[i + 8]]
        target = raw_text[i]
        data.append((context, target))

class CBOW(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()

        #out: 1 x emdedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.linear1 = nn.Linear(embedding_dim, 128)

        self.activation_function1 = nn.ReLU()
        
        #out: 1 x vocab_size
        self.linear2 = nn.Linear(128, vocab_size)

        self.activation_function2 = nn.LogSoftmax(dim = -1)
        

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1,-1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def get_word_emdedding(self, word):
        word = torch.LongTensor([word_to_ix[word]])
        return self.embeddings(word).view(1,-1)


#model = CBOW(vocab_size, EMDEDDING_DIM)

model = torch.load("2model_100.model")

#model.eval()

'''
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


for epoch in range(30):
    print("epoch",epoch)
    total_loss = 0
    for context, target in data:
        context_vector = make_context_vector(context, word_to_ix)  
        model.zero_grad()
        log_probs = model(context_vector)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()

        total_loss += loss.data


torch.save(model, "2model_100.model")
'''
# ====================== TEST
#context = ['People','create','to', 'direct']
context = ['we', 'could', 'predict' ,'whether' ,'or','not' ,'it' ,'would']
context_vector = make_context_vector(context, word_to_ix)
a = model(context_vector).data.numpy()
print('Raw text: {}\n'.format(' '.join(raw_text)))
print('Context: {}\n'.format(context))
print('Prediction: {}'.format(get_max_prob_result(a[0], ix_to_word)))