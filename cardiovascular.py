#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


cardio=pd.read_csv("cardio_train.csv")
cardio


# In[3]:


cardio.isnull().sum()


# In[4]:


cardio.isnull().values.any()


# In[5]:


cardio.info()


# # Cleaning Data

# In[6]:


cardio.drop('id',axis=1,inplace=True)


# In[7]:


cardio


# In[8]:


cardio['gender'].value_counts()


# In[9]:


sns.distplot(cardio["gender"])


# In[10]:


pd.crosstab(cardio.gender, cardio.cardio).plot(kind="bar",figsize=(10,5),color=['#9781cc','#d48cac' ])
plt.title("frequency of diseases vs gender")
plt.ylabel('range')
plt.xlabel('female vs male')
plt.legend(["Diseasesd","Not Diseasesd"])
plt.show()


# In[11]:


cardio.cardio.value_counts()


# In[12]:


sns.boxplot(x='cardio',y='age',data=cardio)


# In[13]:


from matplotlib import rcParams
rcParams['figure.figsize']=20,17
cardio['years']=(cardio['age'] / 365).round().astype('int')
sns.countplot(x='years',hue='cardio',data=cardio,palette="Set2");


# In[14]:


cardio.describe()


# # remove weights and heights that fall below 2.5% or above 97.5%

# In[15]:


cardio.drop(cardio[(cardio['height'] > cardio['height'].quantile(0.975)) |(cardio['height'] < cardio['height'].quantile(0.025))].index,inplace=True)
cardio.drop(cardio[(cardio['weight'] > cardio['weight'].quantile(0.975)) | (cardio['weight'] < cardio['weight'].quantile(0.025))].index,inplace=True)


# In[16]:


plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.boxplot(x='cardio',y='height',data=cardio,palette='winter')
plt.subplot(1,2,2)
sns.boxplot(x='cardio',y='weight',data=cardio,palette='summer')


# # remove ap_hi and ap_lo that fall below 2.5% or above 97.5%

# ap_hi = systolic blood presure
# ap_lo = Diastolic blood presure

# In[17]:


print("Diastilic pressure is higher than systolic one in {0} cases".format(cardio[cardio['ap_lo']> cardio['ap_hi']].shape[0]))


# In[18]:


cardio.drop(cardio[(cardio['ap_hi'] > cardio['ap_hi'].quantile(0.975)) | (cardio['ap_hi'] < cardio['ap_hi'].quantile(0.025))].index,inplace=True)
cardio.drop(cardio[(cardio['ap_lo'] > cardio['ap_lo'].quantile(0.975)) | (cardio['ap_lo'] < cardio['ap_lo'].quantile(0.025))].index,inplace=True)


# In[19]:


blood_pressure = cardio.loc[:,['ap_lo','ap_hi']]
sns.boxplot(x='variable',y='value',data=blood_pressure.melt())
print("Diastilic pressure is higher than systolic one in {0} cases".format(cardio[cardio['ap_lo']> cardio['ap_hi']].shape[0]))


# In[20]:


cardio.groupby('gender')['alco'].sum()


# In[21]:


cardio['BMI'] = cardio['weight']/((cardio['height']/100)**2)
sns.catplot(x="gender", y="BMI", hue="alco", col="cardio", data=cardio, color = "yellow",kind="box", height=10, aspect=.7);


# In[22]:


corr = cardio.corr()


# In[23]:


sns.heatmap(cardio.corr(),annot=True)
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show


# In[24]:


x=cardio[['weight','ap_hi','ap_lo','cholesterol','years','BMI']]
y=cardio['cardio'].values
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.35,random_state=40)


# In[25]:


Classifier=list()


# In[26]:


from sklearn.neighbors import KNeighborsClassifier


# In[27]:


knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(xtrain,ytrain)
pred=knn.predict(xtest)
accuracy=knn.score(xtest,ytest)


# In[28]:


pred


# In[29]:


accuracy


# In[30]:


from sklearn.svm import SVC


# In[31]:


model=SVC()
model.fit(xtrain,ytrain)


# In[32]:


pred2=model.predict(xtest)
pred2


# In[33]:


accuracy_svm=model.score(xtest,ytest)
accuracy_svm


# In[34]:


from sklearn.ensemble import RandomForestClassifier


# In[36]:


rf=RandomForestClassifier(n_estimators=100)


# In[37]:


rf.fit(xtrain,ytrain)


# In[38]:


pred3=rf.predict(xtest)
pred3


# In[39]:


accuracy_rf=rf.score(xtest,ytest)
accuracy_rf


# In[40]:


from sklearn.tree import DecisionTreeClassifier


# In[41]:


DT=DecisionTreeClassifier()


# In[42]:


DT.fit(xtrain,ytrain)


# In[43]:


pred4=DT.predict(xtest)
pred4


# In[44]:


accuracy_DT=DT.score(xtest,ytest)
accuracy_DT


# In[45]:


from sklearn import tree
tree.plot_tree(DT)


# In[46]:


from sklearn.linear_model import LinearRegression
Lr=LinearRegression()


# In[48]:


Lr.fit(xtrain,ytrain)


# In[50]:


pred5=Lr.predict(xtest)
pred5


# In[51]:


accuracy_Lr=Lr.score(xtest,ytest)
accuracy_Lr


# In[53]:


confusion_matrix = pd.crosstab(ytest,pred5,rownames=['Actual'],colnames=['Predicted'])
print(confusion_matrix)


# In[55]:


from sklearn.metrics import roc_auc_score
print('roc_auc:\n',roc_auc_score(ytest,pred5))


# In[62]:


print('Confusion Matrix =  \n\n',confusion_matrix)


# In[ ]:




