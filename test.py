
# coding: utf-8

# In[18]:


import pandas as pd
import pickle as pk


# In[19]:


df=pd.read_csv("upload/test.csv")
df


# In[20]:


#df.info()


# In[21]:


file=open("maps.txt","r")
maps=file.read()
file.close()
#print(maps)


# In[22]:


maps=eval(maps[1:len(maps)-1])
maps


# In[23]:


#['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','classification']
rbc=maps[0]
pc=maps[1]
pcc=maps[2]
ba=maps[3]
htn=maps[4]
dm=maps[5]
cad=maps[6]
appet=maps[7]
pe=maps[8]
ane=maps[9]


# In[24]:


df['rbc'].replace(rbc,inplace=True)
df['pc'].replace(pc,inplace=True)
df['pcc'].replace(pcc,inplace=True)
df['ba'].replace(ba,inplace=True)
df['htn'].replace(htn,inplace=True)
df['dm'].replace(dm,inplace=True)
df['cad'].replace(cad,inplace=True)
df['appet'].replace(appet,inplace=True)
df['pe'].replace(pe,inplace=True)
df['ane'].replace(ane,inplace=True)


# In[25]:


test_data=df.iloc[:,1:25]
test_data.head()


# In[26]:


clf=pk.load(open("model_rf.pkl","rb"))


# In[27]:


test_pred=clf.predict(test_data)
test_pred


# In[28]:


if test_pred==0:
    out="Chronic Kidney Disease"
else:
    out="Not having Chronic Kidney Disease"
print(out)

