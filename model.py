import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from joblib import dump, load
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import re 
from wrangler import Wrangler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score,f1_score, accuracy_score
import matplotlib.pyplot as plt
from collections import Counter
def dummy_fun(doc):
    return doc
class ClassifierModel:
    def train(self):
        data = pd.read_csv('data/text_dataset.csv') #Dataset generated from the wrangler file
        layer_1 = ['suicide', 'non-suicide']
        layer_2 = ['depressed', 'not depressed']
        layer_3 = ['normal', 'lonely', 'stressed', 'anxious']
        all_labels = layer_1 + layer_2 + layer_3
        train, test = train_test_split(data, random_state=745)
        newTrain = train.copy()
        newtest = test.copy() #Split the dataset into training and test
        train['label'].value_counts()
        test.drop('label', axis=1, inplace=True)
        test.sample(5)
        train['suicide'] = 0
        train['lonely'] = 0
        train['stressed'] = 0
        train['normal'] = 0
        train['non-suicide'] = 0
        train['not depressed'] = 0
        train['depressed'] = 0
        train['anxious'] = 0

        train['suicide'] = train.apply(lambda row: self.changing_label_suicide(row), axis=1)
        train['non-suicide'] = train.apply(lambda row: self.changing_label_non_suicide(row), axis=1)

        train['not depressed'] = train.apply(lambda row: self.changing_label_not_depressed(row), axis=1)
        train['depressed'] = train.apply(lambda row: self.changing_label_depressed(row), axis=1)

        train['lonely'] = train.apply(lambda row: self.changing_label_lonely(row), axis=1)
        train['stressed'] = train.apply(lambda row: self.changing_label_stressed(row), axis=1)
        train['normal'] = train.apply(lambda row: self.changing_label_normal(row), axis=1)
        train['anxious'] = train.apply(lambda row: self.changing_label_anxious(row), axis=1)
        n = train.shape[0]
        newVec =  TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None) #we are already tokenized
        vec = TfidfVectorizer(analyzer = "word", max_features=10000)
        tft = newVec.fit_transform(newTrain["text"])
        tfttest = newVec.transform(newtest["text"])
        X_dtm = vec.fit_transform(train['text'])
        test_X_dtm = vec.transform(test['text'])
        labelencoder = LabelEncoder()
        newtrainfit = labelencoder.fit_transform(newTrain["label"])
        suicide_pred_model = DecisionTreeClassifier()
        for label in layer_1:
            print('... Processing {}'.format(label))
            y = train[label]
            suicide_pred_model.fit(X_dtm, y)
            y_pred_X = suicide_pred_model.predict(X_dtm)
            print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))
        

        depression_pred_model = LinearSVC()
        all_labels_pred_model = LinearSVC()
        clf = all_labels_pred_model.fit(tft, newtrainfit)
        calib = CalibratedClassifierCV(base_estimator=all_labels_pred_model, cv="prefit")
        calib.fit(tft, newtrainfit)
        predicted = calib.predict(tfttest)
        conf_matrix = confusion_matrix(y_true=newtest["label"], y_pred=labelencoder.inverse_transform(predicted))
        print("Average accuracy on test set is={}".format(np.mean(predicted == labelencoder.transform(newtest["label"]))))
    
        print('Precision: %.3f' % precision_score(newtest["label"], labelencoder.inverse_transform(predicted), average="weighted"))
        print('Recall: %.3f' % recall_score(newtest["label"], labelencoder.inverse_transform(predicted), average="weighted"))
        print('F1-Score: %.3f' % f1_score(newtest["label"], labelencoder.inverse_transform(predicted), average="weighted"))
        print('Accuracy: %.3f' % accuracy_score(newtest["label"], labelencoder.inverse_transform(predicted)))
        for label in layer_2:
            print('... Processing {}'.format(label))
            y = train[label]
            depression_pred_model.fit(X_dtm, y)
            y_pred_X = depression_pred_model.predict(X_dtm)
            print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))
        

        dump(suicide_pred_model, 'suicide_prediction.joblib') 
        dump(depression_pred_model, 'depression_prediction.joblib') 
        dump(all_labels_pred_model,'all_labels_prediction.joblib' )
        dump(newVec, "vectorizor.joblib")
        dump(labelencoder, "labelencoder.joblib")
        dump(calib, "calibrated_all_labels.joblib")


    def changing_label_suicide(self,row):
        if row['label'] == 'suicide':
            return 1
        return 0

    def changing_label_lonely(self,row):
        if row['label'] == 'lonely':
            return 1
        return 0

    def changing_label_stressed(self,row):
        if row['label'] == 'stressed':
            return 1
        return 0

    def changing_label_normal(self,row):
        if row['label'] == 'normal':
            return 1
        return 0

    def changing_label_non_suicide(self,row):
        if row['label'] == 'non-suicide':
            return 1
        return 0

    def changing_label_not_depressed(self,row):
        if row['label'] == 'not depressed':
            return 1
        return 0

    def changing_label_depressed(self,row):
        if row['label'] == 'depressed':
            return 1
        return 0

    def changing_label_anxious(self,row):
        if row['label'] == 'anxious':
            return 1
        return 0
    def predict_label(self,text):
        wrang = Wrangler()
        text_tokened = wrang.preparetext(text)
        vector =  load("vectorizor.joblib")
        labelencoder = load("labelencoder.joblib")
        print(text_tokened)
        textVec = vector.transform(text_tokened)
        model = load("calibrated_all_labels.joblib")
        prediction = model.predict(textVec)
        print(prediction)
        true_pred = labelencoder.inverse_transform(prediction)
        if "suicide" in true_pred:
            return "suicide"
        else:
            counts = Counter(true_pred)
            return counts.most_common(1)[0]
        
if __name__ == "__main__":
    mod = ClassifierModel()
    mod.train()
    mod.predict_label("man i want to die")
