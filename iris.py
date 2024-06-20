
# iris data classification 






import matplotlib.pyplot as plt 
import pandas  as pd
import numpy as np
import seaborn as sns
from  sklearn import datasets


 sns.set()

# data=datasets.load_iris()
# In[222]:


data['target_names']


# In[223]:


df=pd.DataFrame(data['data'],columns=data["feature_names"])


# In[224]:


df['target']=data['target'].astype(float)
df.dtypes


# In[225]:


df.head()


# In[226]:


df.describe()


# Distribution of features and target

# In[227]:


col='sepal length (cm)'
df[col].hist()
plt.suptitle(col)


# In[228]:


df['target_name']=df['target'].map({0:'setosa',1:'versicolor',2:'virginica'})


# In[229]:


df.head()


# In[230]:


col='petal length (cm)'
sns.relplot(x=col,y='target',hue='target',data=df)


# In[231]:


from sklearn.model_selection import train_test_split


# In[232]:


df_train,df_test=train_test_split(df,test_size=0.25)


# In[46]:





# In[233]:


x_train=df_train.drop(columns=['target','target_name']).values
#x_train = df_train.drop(columns=['target']).values


# In[234]:


y_train=df_train['target'].values

#df['target_name']
df.dtypes


# manual modelling 
# 

# In[235]:


sns.pairplot(df,hue='target_name')


# In[237]:


def missing(petal):
    if petal<2.6:
        return 0
    elif petal<4.8:
        return 1
    else:
        return 0
        


# In[238]:


df_train.columns


# In[63]:


x_train[:,2]


# In[239]:


l= np.array([missing(val) for val in x_train[:,2]])


# In[240]:


l==y_train


# BASIC MODELLING 

# In[241]:


np.mean(l==y_train)*100


# logisitic regression

# In[242]:


from sklearn.linear_model import LogisticRegression


# In[243]:


model=LogisticRegression()
model.fit(x_train,y_train)


# In[244]:


model.score(x_train,y_train)


# In[246]:


xt,xv,yt,yv=train_test_split(x_train,y_train,test_size=0.25)


# In[247]:


model.fit(xt,yt)


# In[248]:


y_pred=model.predict(xv)
print(y_pred)


# In[249]:


print("accuracy of training set",np.mean(y_pred==yv)*100,"%")


# In[250]:


model.score(xv,yv)


# cross validation

# In[251]:


from sklearn.model_selection import cross_val_score,cross_val_predict,cross_validate


# In[252]:


model=LogisticRegression(max_iter=200)
print(model)


# In[253]:


c=cross_val_score(model,x_train,y_train ,cv=5,scoring='accuracy')


# In[254]:


print("accuracy after cross validation",np.mean(c)*100)


# missclassified points

# In[255]:


y_pred=cross_val_predict(model,x_train,y_train)

predicted_correctly_mask=y_pred==y_train


# In[256]:


not_predicted_correctly=~predicted_correctly_mask
x_train[not_predicted_correctly]


# In[155]:


df_predictions=df_train.copy()
df_predictions["correct_predictions"]=predicted_correctly_mask


# 

# In[187]:


df_predictions['prediction']=y_pred
df_predictions['prediction_label']=df_predictions['prediction'].map({0:"setosa",1:"versicolor",2:"virginica"})
df_predictions.head()


# In[257]:


#model tuning 
for reg_param in (0.1,0.5,1,1.6,1.7,2,3):
    print(reg_param)
    model=LogisticRegression(max_iter=200,C=reg_param)
    accuracies=cross_val_score(model,x_train,y_train,cv=5,scoring="accuracy")
    print(f"accuracy:{np.mean(accuracies)*100:.2f}%")


# In[258]:


x_test=df_test.drop(columns=['target','target_name']).values
y_test=df_test['target'].values


# In[259]:


df_predictions.head()
df_predictions.dtypes


# In[260]:


model.fit(x_train,y_train)


# In[261]:


y_test_pred=model.predict(x_test)


# In[262]:


test_set_correct=y_test_pred==y_test




ac=np.mean(test_set_correct)





print("accuracy of model after testing is  ",(100*ac),"%")







