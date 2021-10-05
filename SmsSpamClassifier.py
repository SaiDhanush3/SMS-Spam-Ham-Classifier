import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score

nltk.download('stopwords')

messages = pd.read_csv("SpamDataset",sep="\t",names=["label","message"])
ps = PorterStemmer()
corpus = []
for i in range(len(messages)):
    sms = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    sms = sms.lower().split()
    sms = [ps.stem(word) for word in sms if word not in stopwords.words('english')]
    corpus.append(' '.join(sms))

cv = CountVectorizer(max_features=3000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size = 0.20,random_state = 0)
spam_detect_model = MultinomialNB().fit(X_train,y_train)
y_pred = spam_detect_model.predict(X_test)

confusion_matrix = confusion_matrix(y_test,y_pred)
print(confusion_matrix)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of the classifier is : ",accuracy)
