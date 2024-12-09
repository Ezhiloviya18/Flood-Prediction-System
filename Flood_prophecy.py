#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


import numpy as np
x=pd.read_csv("/Users/joel/Pictures/TamilNadu.csv")
y=pd.read_csv("/Users/joel/Pictures/TamilNadu.csv")

y1=list(x["YEAR"])
x1=list(x["Oct-Dec"])
z1=list(x["OCT"])
n1=list(x["NOV"])
w3=list(x["SEP"])


plt.plot(y1, x1,'*')
plt.show()


# In[4]:


flood=[]
nov=[]
sub=[]


for i in range(0,len(x1)):
    if x1[i]>580:
        flood.append('1')
    else:
        flood.append('0')


for k in range(0,len(x1)):
    nov.append(n1[k]/3)


for k in range(0,len(x1)):
    sub.append(abs(w3[k]-z1[k]))


df = pd.DataFrame({'flood':flood})
df1=pd.DataFrame({'per_10_days':nov})

x["flood"]=flood
x["avgnov"]=nov
x["sub"]=sub

x.to_csv("out1.csv")
print((x))


# In[5]:


import scipy
from scipy.stats import spearmanr
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split



# In[6]:



c = x[['AUG','SEP','OCT','NOV']]
c.hist()
plt.show()


# In[7]:


ax = x[['JAN', 'FEB', 'MAR', 'APR','MAY', 'JUN', 'AUG', 'SEP', 'OCT','NOV','DEC']].mean().plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2,figsize=(14,6))
plt.xlabel('Month',fontsize=30)
plt.ylabel('Monthly Rainfall',fontsize=20)
plt.title('Rainfall in Tamil Nadu for all Months',fontsize=25)
ax.tick_params(labelsize=20)
plt.grid()
plt.ioff()


# In[8]:


import seaborn as sns


# In[9]:


sns.pairplot(x[['avgnov','Oct-Dec','sub']])
plt.show()


# In[10]:


sns.jointplot(x[['OCT','NOV']])
plt.show()


# In[11]:


sns.set_theme(style="white")
sns.relplot(x[['OCT','NOV']])
plt.show()


# In[12]:


X = x.loc[:,["Oct-Dec","avgnov","sub"]].values
y1=x.iloc[:,19].values
X_train, X_test, Y_train, Y_test= train_test_split(X, y1 ,random_state=1,test_size=0.4)
from sklearn import preprocessing
minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
minmax.fit(X).transform(X)


# In[13]:


X_train.shape


# In[14]:


X_test.shape


# In[15]:


Y_train.shape


# In[16]:


Y_test.shape


# In[17]:


import plotly.graph_objects as go
trace_specs = [
    [X_train, Y_train, '0', 'Train', 'square'],
    [X_train, Y_train, '1', 'Train', 'circle'],
    [X_test, Y_test, '0', 'Test', 'square-dot'],
    [X_test, Y_test, '1', 'Test', 'circle-dot']
]
fig = go.Figure(data=[
    go.Scatter(
        x=X[y==label, 0], y=X[y==label, 1],
        name=f'{split} Split, Label {label}',
        mode='markers', marker_symbol=marker
    )
    for X, y, label, split, marker in trace_specs
])
fig.update_traces(
    marker_size=12, marker_line_width=1.5,
    marker_color="lightyellow"
)
fig.show()


# In[18]:


#KNN
from sklearn import model_selection,neighbors
clf=neighbors.KNeighborsClassifier()
clf.fit(X_train,Y_train)


# In[19]:


print("Predicted Values for the Floods:")
y_predict=clf.predict(X_test)
y_predict


# In[20]:


print("Actual Values for the Floods:")
print(Y_test)


# In[21]:


print("List of the Predicted Values:")
print(y_predict)


# In[22]:


from sklearn.model_selection import cross_val_score,cross_val_predict
x_train_std= minmax.fit_transform(X_train)
x_test_std= minmax.fit_transform(X_test)
knn_acc=cross_val_score(clf,x_train_std,Y_train,cv=3,scoring='accuracy',n_jobs=-1)
knn_proba=cross_val_predict(clf,x_train_std,Y_train,cv=3,method='predict_proba')


# In[23]:


knn_proba


# In[24]:


knn_acc


# In[25]:


from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix
print("\nAccuracy Score:%f"%(accuracy_score(Y_test,y_predict)*100))
print("ROC score:%f"%(roc_auc_score(Y_test,y_predict)*100))
mat=confusion_matrix(Y_test,y_predict)
sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("True Class")
plt.ylabel("Predicted Class")
plt.show()


# In[26]:


q1=312.6 #Rainfall during OCT-DEC 2017
w=195 #First 10 days in NOV 2017
e1=198 #Diff of oct and sep 2017

l=[[q1,w,e1],[50,300,205]]

f1=clf.predict(l)
print(f1)


# In[27]:


from sklearn.linear_model import LogisticRegression
Lr=LogisticRegression()
x_train_std=minmax.fit_transform(X_train)         # fit the values in between 0 and 1.
y_train_std=minmax.transform(X_test)

Lr.fit(X_train,Y_train)
lr_acc=cross_val_score(Lr,x_train_std,Y_train,cv=3,scoring='accuracy',n_jobs=-1)
lr_proba=cross_val_predict(Lr,x_train_std,Y_train,cv=3,method='predict_proba')
print(lr_acc)
print(lr_proba)
q1=313
w=195
e1=198 

q2=200 
w2=400 
e2=300 

l=[[q1,w,e1],[50,300,205]]

ypred=Lr.predict(X_test)
f1=Lr.predict(l)

print(f1)




# In[28]:


from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix
print("\naccuracy score:%f"%(accuracy_score(Y_test,ypred)*100))
print("roc score:%f"%(roc_auc_score(Y_test,ypred)*100))
mat=confusion_matrix(Y_test,ypred)
sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("True Class")
plt.ylabel("Predicted Class")
plt.show()


# In[29]:


from sklearn.svm import SVC
svc=SVC(kernel='rbf',probability=True)
svc_classifier=svc.fit(X_train,Y_train)
svc_acc=cross_val_score(svc_classifier,x_train_std,Y_train,cv=3,scoring="accuracy",n_jobs=-1)
svc_proba=cross_val_predict(svc_classifier,x_train_std,Y_train,cv=3,method='predict_proba')


# In[30]:


svc_acc


# In[31]:


svc_proba


# In[32]:


svc_scores=svc_proba[:,1]
svc_scores


# In[33]:


y_pred=svc_classifier.predict(X_test)
print("Actual Flood Values:")
print(Y_test)
print("Predicted Flood Values")
print(y_pred)


# In[34]:


from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix
print("\naccuracy score:%f"%(accuracy_score(Y_test,y_pred)*100))
print("roc score:%f"%(roc_auc_score(Y_test,y_pred)*100))
mat=confusion_matrix(Y_test,y_pred)
sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("True Class")
plt.ylabel("Predicted Class")
plt.show()


# In[467]:


from sklearn.metrics import classification_report
print("Classification report\n")
print("K-Nearest Neighbor Classification\n")
print(classification_report(Y_test,y_predict))
print("Logistic Regression\n")
print(classification_report(Y_test,ypred))
print("Support Vector Classification\n")
print(classification_report(Y_test,y_pred))
print("Artificial Neural Network\n")
print(classification_report(Y_test,y_pred))


# In[40]:
def fpredict(input1,input2,input3):

    q1=input1 #Rainfall during OCT-DEC 2017
    w=input2 #First 10 days in NOV 2017
    e1=input3 #Diff of oct and sep 2017
    l=[[q1,w,e1]]
    r1=Lr.predict(l)
    r2=clf.predict(l)
    r3=svc.predict(l)
    return{
        'result1': r1,
        'result2': r2,
        'result3': r3
    }
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import plotly.express as px
y_score = clf.predict_proba(X_test)[:, 1]


fig = px.scatter(
    X_test, x=0, y=1,
    color=y_score, color_continuous_scale='RdBu',
    symbol=Y_test, symbol_map={'0': 'square-dot', '1': 'circle-dot'},
    labels={'symbol': 'label', 'color': 'score of <br>first class'}
)
fig.update_traces(marker_size=12, marker_line_width=1.5)
fig.update_layout(legend_orientation='h')
fig.show()

