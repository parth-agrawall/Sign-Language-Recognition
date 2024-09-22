import pickle as pkl
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pkl.load(open("dataset.pickle",'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data,labels,test_size=0.2,
                                                    shuffle = True, stratify=labels)

classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)

y_predict = classifier.predict(x_test)

score = accuracy_score(y_test,y_predict)
print(f"{score*100}% of samples were classified successfully!")

file = open("model.pkl","wb")
pkl.dump({"model": classifier},file)
file.close()

print("model created successfully")
