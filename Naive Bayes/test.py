from starter_code.utils import load_training_set, load_test_set
from collections import Counter
from naive_bayes import NaiveBayes
import itertools
from calculate_accuracy import CalculateAccuracy
import progressbar

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
print("check len train:pos,neg,both",len(pos_train),len(neg_train),len(pos_train)+len(neg_train))
model = NaiveBayes(laplaceFactor=0,logProb=False)
model.fit(trainData=trainData,bow=vocab)
print("check len test:pos,neg,both",len(pos_test),len(neg_test),len(pos_test)+len(neg_test))

y = []
y.extend(pos_test)
y.extend(neg_test)
actual = []
actual.extend(list(itertools.repeat("positive", len(pos_test))))
actual.extend(list(itertools.repeat("negative", len(neg_test))))

pred = model.predict(y)
accObj = CalculateAccuracy(test=actual,pred=pred,labels=["positive","negative"])

print("Confusion Matrix")
print(accObj.confusion_matrix())
accObj.plotConfusionMatrix()
print("Accuracy")
print(accObj.accuracy())
print("Precision")
print(accObj.precision())
print("Recall")
print(accObj.recall())
accObj.plotPrecisionRecall()
