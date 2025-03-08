from collections import Counter

def generateWordFreq(doc_vector):
    return Counter(doc_vector).items()