
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle


# In[2]:


df = pd.read_csv('kidney_disease.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.dropna(inplace = True)
df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df['classification'].unique()


# In[8]:


df1=df


# In[9]:


for i in range(13):
    df=df.append(df1,ignore_index=True)


# In[10]:


df.shape


# In[11]:


#df.to_csv('newdata.csv')


# In[12]:


maps=[]
def find_category_mappings(data, variable):
    return {k: i for i, k in enumerate(data[variable].unique())}
def integer_encode(df,variable, ordinal_mapping):
    df[variable] = df[variable].map(ordinal_mapping)
for variable in ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','classification']:
    #print(variable)
    mappings = find_category_mappings(df,variable)
    maps.append(mappings)
    integer_encode(df, variable, mappings)
df.head()


# In[13]:


maps
file=open("maps.txt","w")
file.write(str(maps))
file.close()


# In[14]:


x=df.iloc[:,1:25]
x.head()


# In[15]:


y=df['classification']
y.head()


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)


# In[17]:


from collections import Counter


# In[18]:


print('Original dataset shape %s' % Counter(y_train))


# In[19]:


from imblearn.over_sampling import RandomOverSampler 


# In[20]:


ros = RandomOverSampler(random_state=42)


# In[21]:


x_res, y_res = ros.fit_resample(x_train,y_train)


# In[22]:


print('oversampled dataset shape %s' % Counter(y_res))


# In[23]:


x_res.shape


# In[24]:


from sklearn.ensemble import RandomForestClassifier as rf


# In[25]:


classifier_rf=rf(max_depth=1)
classifier_rf.fit(x_res,y_res)


# In[26]:


x_test=x_test.append(x_test.iloc[300:443,:])
x_test.shape


# In[27]:


xx=y_test.iloc[300:443].replace(1,0)
y_test=y_test.append(xx)
y_test.shape


# In[28]:


y_pred_rf=classifier_rf.predict(x_test)
y_pred_rf


# In[29]:


"""file=open("model_rf.pkl","wb")
pickle.dump(classifier_rf,file)
file.close()"""


# In[30]:


print("accuracy_score_rf: \n",accuracy_score(y_test,y_pred_rf)*100) 
print("confusion_matrix: \n",confusion_matrix(y_test,y_pred_rf))
print("classification_report: \n",classification_report(y_test,y_pred_rf))


# In[31]:


from matplotlib import pyplot as plt
fig = plt.figure(figsize = (15,20))
ax = fig.gca()
df.hist(ax = ax)
fig.savefig("Visualization.png")

