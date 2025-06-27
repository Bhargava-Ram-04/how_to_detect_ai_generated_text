from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from nltk.stem import PorterStemmer
import os
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector

global filename, xgb_cls, labels
global X, Y
global dataset
global accuracy, precision, recall, fscore, basic_features
global X_train, X_test, y_train, y_test, scaler, pca, tfidf_vectorizer
#define object to remove stop words and other text processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

main = tkinter.Tk()
main.title("How to Detect AI-Generated Texts") #designing main screen
main.geometry("1300x1200")

def punctuationCount(data):#get punctuation count
    counts = Counter(data)
    punctuation_counts = {k:v for k, v in counts.items() if k in punctuation}
    return sum(punctuation_counts.values())

def getCountDensity(data): #get word count, density and upper letter count
    upper_count = sum(1 for c in data if c.isupper())
    arr = data.split(" ")
    count = len(arr)
    density = len(data) / count
    return count, density, upper_count

def getPOS(data):#get noun and other POS count 
    tokenized = nltk.word_tokenize(data)
    pos = nltk.pos_tag(tokenized)
    counts = Counter(tag for word, tag in pos)
    counts = dict((word, count) for word, count in counts.items())
    noun = 0
    verb = 0
    adj = 0
    adv = 0
    pronoun = 0
    if 'NN' in counts.keys():
        noun = counts['NN']
    if 'VB' in counts.keys():
        verb = counts['VB']
    if 'JJ' in counts.keys():
        adj = counts['JJ']
    if 'RB' in counts.keys():
        adv = counts['RB']
    if 'PRP' in counts.keys():
        pronoun = counts['PRP']     
    return noun, verb , adj, adv, pronoun   

#define function to clean text by removing stop words and other special symbols
def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

 
#fucntion to upload dataset
def uploadDataset():
    global filename, dataset, labels
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset") #upload dataset file
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename) #read dataset from uploaded file
    labels = ['Human', 'AI']
    text.insert(END,"Dataset Values\n\n")
    text.insert(END,str(dataset.head()))
    text.update_idletasks()
    
    label = dataset.groupby('label').size()
    label.plot(kind="bar")
    plt.xlabel('0 (Human), 1 (AI)')
    plt.ylabel('Label Count')
    plt.title("Dataset Class Label Graph")
    plt.show()
    
    
def featuresExtraction():
    text.delete('1.0', END)
    global dataset, scaler, X, Y, basic_features, tfidf_vectorizer
    data = dataset['text'].ravel()
    label = dataset['label'].ravel()
    if os.path.exists("model/basic_features.npy"):
        basic_features = np.load('model/basic_features.npy')
        X = np.load("model/X.npy")
        Y = np.load("model/Y.npy")
        X = np.asarray(X)
    else:
        for i in range(len(data)):
            text_data = data[i]
            punct = punctuationCount(text_data)
            count, density, upper_count = getCountDensity(text_data)
            noun, verb , adj, adv, pronoun = getPOS(text_data)         
            text_data = text_data.strip("\n").strip().lower()
            target = label[i]
            text_data = cleanText(text_data)#clean data
            X.append(text_data)
            Y.append(target)
            basic_features.append([punct, count, density, upper_count, noun, verb, adj, adv, pronoun])
            print(str(i)+" "+str(noun)+" "+str(verb)+" "+str(adj)+" "+str(adv)+" "+str(pronoun))
        Y = np.asarray(Y)
        basic_features = np.asarray(basic_features)
        np.save("model/X", X)
        np.save("model/Y", Y)
        np.save("model/basic_features", basic_features)
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=2000)
    X = tfidf_vectorizer.fit_transform(X).toarray()
    X = np.hstack((X, basic_features))
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]  
    Y = Y[indices]
    text.insert(END,"Basic NLP & TF-IDF features extracted from Dataset\n\n")
    text.insert(END, str(X)+"\n\n")
    text.insert(END,"Total features available in Dataset : "+str(X.shape[1]))

def PCASelection():
    global X, Y, scaler, pca
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    text.insert(END,"Total features available in Dataset before applying PCA: "+str(X.shape[1])+"\n\n")
    X = X[0:1000]
    Y = Y[0:1000]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=100)
    X = pca.fit_transform(X)
    text.insert(END,"Total features available in Dataset after applying PCA: "+str(X.shape[1])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split data into train & test
    text.insert(END,"Dataset Train and Test Split\n\n")
    text.insert(END,"80% dataset records used to train algorithms 38000\n")
    text.insert(END,"20% dataset records used to test algorithms 7200\n")

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()

def trainRF():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    text.delete('1.0', END)
    #defining Randm Forest tuning parameters
    hyperParameters = {'n_estimators': [100]}
    rf = RandomForestClassifier()
    rf = GridSearchCV(rf, hyperParameters, scoring='accuracy')
    rf.fit(X_train, y_train)
    predict = rf.predict(X_test)
    calculateMetrics("Random Forest", predict, y_test)

def trainSVM():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    #defining SVM tuning parameters
    hyperParameters = {'kernel': ['poly']}
    svm_cls = svm.SVC()
    svm_cls = GridSearchCV(svm_cls, hyperParameters, scoring='accuracy')
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM", predict, y_test)

def trainXGBoost() :
    global X_train, X_test, y_train, y_test, xgb_cls
    global accuracy, precision, recall, fscore
    #defining XGBoost tuning parameters
    hyperParameters = {'n_estimators': [100]}
    xgb_cls = XGBClassifier()
    xgb_cls = GridSearchCV(xgb_cls, hyperParameters, scoring='accuracy')
    xgb_cls.fit(X_train, y_train)
    predict = xgb_cls.predict(X_test)
    calculateMetrics("XGBoost", predict, y_test)

def trainDecisionTree():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    #defining XGBoost tuning parameters
    hyperParameters = {'criterion': ['gini']}
    dt_cls = DecisionTreeClassifier()
    dt_cls = GridSearchCV(dt_cls, hyperParameters, scoring='accuracy')
    dt_cls.fit(X_train, y_train)
    predict = dt_cls.predict(X_test)
    calculateMetrics("Decision Tree", predict, y_test)
    

def graph():
    df = pd.DataFrame([['Random Forest','Precision',precision[0]],['Random Forest','Recall',recall[0]],['Random Forest','F1 Score',fscore[0]],['Random Forest','Accuracy',accuracy[0]],
                       ['SVM','Precision',precision[1]],['SVM','Recall',recall[1]],['SVM','F1 Score',fscore[1]],['SVM','Accuracy',accuracy[1]],
                       ['XGBoost','Precision',precision[2]],['XGBoost','Recall',recall[2]],['XGBoost','F1 Score',fscore[2]],['XGBoost','Accuracy',accuracy[2]],
                       ['Decision Tree','Precision',precision[3]],['Decision Tree','Recall',recall[3]],['Decision Tree','F1 Score',fscore[3]],['Decision Tree','Accuracy',accuracy[3]],
                      ],columns=['Algorithms','Performance Output','Value'])
    df.pivot("Algorithms", "Performance Output", "Value").plot(kind='bar')
    plt.show()

def predictText():
    global xgb_cls, pca, scaler, tfidf_vectorizer
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testFiles") #upload dataset file
    data = ""
    with open(filename, "rb") as file:
        data = file.read()
    file.close()
    data = data.decode()
    text.insert(END,"Input Text : "+data+"\n\n")
    basic = []
    punct = punctuationCount(data)
    count, density, upper_count = getCountDensity(data)
    noun, verb , adj, adv, pronoun = getPOS(data)         
    data = data.strip("\n").strip().lower()
    data = cleanText(data)#clean data
    basic.append([punct, count, density, upper_count, noun, verb, adj, adv, pronoun])
    temp = []
    temp.append(data)
    temp = tfidf_vectorizer.transform(temp).toarray()
    basic = np.asarray(basic)
    temp = np.hstack((temp, basic))
    print(temp.shape)
    temp = scaler.transform(temp)
    temp = pca.transform(temp)
    print(temp.shape)
    predict = xgb_cls.predict(temp)
    print(predict)
    predict = predict[0]
    if predict == 0:
        text.insert(END,"Predicted Text is : Hand Written\n")
    if predict == 1:
        text.insert(END,"Predicted Text is : AI Generated\n")
        
        
font = ('times', 16, 'bold')
title = Label(main, text='How to Detect AI-Generated Texts')
title.config(bg='greenyellow', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Human-AI Text Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

tfidfButton = Button(main, text="Basic NLP & TF-IDF Features", command=featuresExtraction)
tfidfButton.place(x=330,y=550)
tfidfButton.config(font=font1) 

pcaButton = Button(main, text="PCA Features Selection", command=PCASelection)
pcaButton.place(x=600,y=550)
pcaButton.config(font=font1)

rfButton = Button(main, text="Train Random Forest Algorithm", command=trainRF)
rfButton.place(x=850,y=550)
rfButton.config(font=font1)

svmButton = Button(main, text="Train SVM Algorithm", command=trainSVM)
svmButton.place(x=50,y=600)
svmButton.config(font=font1)

xgButton = Button(main, text="Train XGBoost Algorithm", command=trainXGBoost)
xgButton.place(x=330,y=600)
xgButton.config(font=font1)

dtButton = Button(main, text="Train Decision Tree Algorithm", command=trainDecisionTree)
dtButton.place(x=600,y=600)
dtButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=850,y=600)
graphButton.config(font=font1)

predictButton = Button(main, text="Human-AI Text Prediction", command=predictText)
predictButton.place(x=50,y=650)
predictButton.config(font=font1)


main.config(bg='LightSkyBlue')
main.mainloop()
