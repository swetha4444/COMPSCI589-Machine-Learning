from starter_code.utils import load_training_set, load_test_set
from collections import Counter
from naive_bayes import NaiveBayes
import itertools


percentage_positive_instances_train = 1
percentage_negative_instances_train = 1
percentage_positive_instances_test = 1
percentage_negative_instances_test = 1
(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
(pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

trainData = {
    "positive":pos_train,
    "negative":neg_train
}
print("check len:pos,neg,both",len(pos_train),len(neg_train),len(pos_train)+len(neg_train))
model = NaiveBayes(laplaceFactor=1,logProb=True)
model.fit(trainData=trainData,bow=vocab)

pred = model.predict(neg_test)
test = list(itertools.repeat("negative", len(neg_test)))
print(model.accuracy(pred=pred,test=test))

pred = model.predict(pos_test)
test = list(itertools.repeat("positive", len(pos_test)))
print(model.accuracy(pred=pred,test=test))