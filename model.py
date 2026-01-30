import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict

nltk.download('punkt')

def build_model(text):
    words = word_tokenize(text.lower())
    model = defaultdict(list)

    for i in range(len(words)-1):
        model[words[i]].append(words[i+1])

    return model

def predict_next_word(model, word):
    word = word.lower()
    if word in model:
        return model[word][0]   # returns first predicted word
    else:
        return "No prediction found"
