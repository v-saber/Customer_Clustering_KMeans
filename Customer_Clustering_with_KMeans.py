#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import ydata_profiling  # pandas_profiling
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from collections import Counter
from sklearn.cluster import KMeans
from kneed import KneeLocator
from pylab import *


# In[2]:


data = pd.read_csv("D:\programming\Machine Learning/Customer_Data.csv")
data


# In[3]:


df = data.replace('C1','', regex=True)
df["CUST_ID"] = df["CUST_ID"].astype("int")
df


# In[4]:


df.shape


# In[5]:


df.describe(include="all")


# <div style=" font-size:14px; line_height:160%">
# With a general look at the values in the datasat, we can guess that the values of the PURCHASES column are equall to ONEOFF PURCHASES + INSTALLMENT PURCHASES values.<br/>
# Let's see if our guess is True...

# In[6]:


df.columns[2] == df.columns[3] + df.columns[4]


# <div style=" font-size:14px; line_height:160%">
# They are not equall.
# Let's see how many samples are different:

# In[7]:


compare = pd.DataFrame()
compare["purchases"] = df["PURCHASES"]
compare["oneoff+installment"] = df["ONEOFF_PURCHASES"] + df["INSTALLMENTS_PURCHASES"]

m = compare["purchases"] == compare["oneoff+installment"]
Counter(m)


# <div style=" font-size:14px; line_height:160%">
# As it can be seen, there are 492 samples in which PURCHASES column values are not equall to ONEOFF PURCHASES + INSTALLMENT PURCHASES.

# <div style=" font-size:14px; line_height:160%">
# Let's try to plot these samples:

# In[8]:


compare["difference"] = compare["purchases"] - compare["oneoff+installment"]

compare


# In[9]:


plt.scatter(compare.difference, compare.purchases)
plt.title("The difference between Purchases and the summation of \nOneoff and Installment Purchases\n", fontsize=20)
plt.xlabel("\ndifference", fontsize=20)
plt.ylabel("purchases\n", fontsize=20)
plt.grid()
plt.show()


# In[10]:


compare[compare["difference"]<-1e-10].sort_values(by="difference")


# <div style=" font-size:14px; line_height:160%">
# There are 19 samples in which there are differences between PURCHASES column values and the summation of the two columns ONEOFF_PURCHASES and INSTALLMENT_PURCHASES.

# <div style=" font-size:14px; line_height:160%">
# I think it would be better if we delete them from the dataset.

# In[11]:


df.drop(index=compare[compare["difference"]<-1e-10].index, inplace=True)
df.reset_index(drop=True, inplace=True)


# In[12]:


df.shape


# In[13]:


df.info()


# <div style=" font-size:14px; line_height:160%">
# It can be seen that there are some missing values in the dataset.<br/>
# Let's check them:

# In[14]:


# df.isnull().any()
df.columns[df.isnull().any()].tolist()


# <div style=" font-size:14px; line_height:160%">
# There are missing values in two columns. CREDIT_LIMIT and MINIMUM_PAYMENTS.<br/>
# In the following cell we can find out how many rows are having missing values:

# In[15]:


df.isnull().sum()


# <div style=" font-size:14px; line_height:160%">
# There are 314 missing values.
# One in CREDIT_LIMIT and 313 in MINIMUM_PAYMENTS.
# Let's fill theme with the mean values of the related column.

# In[16]:


df["MINIMUM_PAYMENTS"].fillna(np.mean(df["MINIMUM_PAYMENTS"]), inplace=True)
df["CREDIT_LIMIT"].fillna(np.mean(df["CREDIT_LIMIT"]), inplace=True)


# In[17]:


df.duplicated().sum()


# <div style=" font-size:14px; line_height:160%">
# There is no duplicated rows in the dataframe.

# <div style=" font-size:14px; line_height:160%">
# We don't need the "CUST_ID" column for the clustering process

# In[18]:


df.drop("CUST_ID", axis=1, inplace=True)


# <div style=" font-size:14px; line_height:160%">
# Using pandas profiling to have a quick look at the features:

# In[19]:


ydata_profiling.ProfileReport(df)


# In[20]:


plt.figure(figsize=(20,10), dpi=80)
sns.heatmap(df.corr(), annot=True, cmap=plt.cm.Blues)
plt.show()


# <div style=" font-size:14px; line_height:160%">
# We have a high correlation between the two features ONEOFF_PURCHASES and PURCHASES but this issue does not have a big impact on the final resul<br/>
# On the other hand, because we want to compare ONEOFF_PURCHASES with other features, we do not remove it.

# <div style=" font-size:18px; line_height:160%">
# Finding outliers:

# In[21]:


def scatter_plots(df_name, x_ax_name, y_ax_name):
    scatter_name = f"{y_ax_name}-{x_ax_name}"
    fig_output_name = scatter_name
    plt.title(f"{x_ax_name} - {y_ax_name}\n")
    scatter_name = plt.scatter(df_name[x_ax_name], df_name[y_ax_name])
    scatter_name.axes.tick_params(gridOn=True, size=12, labelsize=10)
    plt.xlabel(f"\n{x_ax_name}", fontsize=20)
    plt.ylabel(f"{y_ax_name}\n", fontsize=20)
    plt.xticks(fontsize=15, rotation=60)
    plt.yticks(fontsize=15)


# In[22]:


def scatter_subplots(df):
    
    for j in range(len(df.columns)):
        print(f"\nPlotting {df.columns[j]} with other columns:\n")
        i=1
        while i < len(df.columns):
            if j+i==len(df.columns):
                break
            plt.figure(figsize=(20,8), dpi=80)
            for k in range(3):
                plt.subplot(1, 3, k+1)                    
                scatter_plots(df, df.columns[j], df.columns[j+i])
                plt.title(f'"{df.columns[j]} - {df.columns[j+i]}"', fontsize=20)            
                i += 1
                if j+i==len(df.columns):
                    break

            plt.suptitle("Plotting Each Features vs Other Features", size = 30, fontweight = "bold")
            plt.tight_layout()
            plt.show()


# In[23]:


scatter_subplots(df)


# <div style=" font-size:14px; line_height:160%">
# There does not seem to be significant outliers in the dataset based on the plots above.

# <div style=" font-size:18px; line_height:160%">
# Madeling

# <div style=" font-size:14px; line_height:160%">
# Since the KMeans method uses the distances between the values, we should scale the values of the dataset.

# In[24]:


scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)


# In[25]:


scaled_features = pd.DataFrame(scaled_features, columns=df.columns)
scaled_features


# In[26]:


kmeans_parameters = {"init": "random", "n_init": 50, "max_iter": 300, "random_state": 42}


# In[27]:


List_inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_parameters)  # ** open dictionary
    kmeans.fit(scaled_features)
    List_inertia.append(kmeans.inertia_)


# In[28]:


# list of the kmeans.inertia_
List_inertia


# <div style=" font-size:14px; line_height:160%">
# The parameter inertia can be used to determine the proper number of clusters.<br/>
# Look at the plot below:

# In[29]:


plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), List_inertia)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("inertia")
plt.show()


# <div style=" font-size:14px; line_height:160%">
# With KneeLocator we can make sure that we choose a correct number as the number of clusters.

# In[30]:


kl = KneeLocator(range(1, 11), List_inertia, curve="convex", direction="decreasing")
kl.elbow


# In[31]:


plt.style.use("Solarize_Light2")
plt.plot(range(1, 11), List_inertia)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("inertia")
plt.axvline(x=kl.elbow, color="b", label="axvline - full height", ls="--")
plt.show()


# <div style=" font-size:14px; line_height:160%">
# In the following, we use two metrics in order to help us verify our cluster number choosing.

# <div style=" font-size:18px; line_height:160%">
# Metric #1: Silhouette Coefficients

# In[32]:


silhouette_coefficients = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_parameters)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_coefficients.append(score)


# In[33]:


plt.style.use("ggplot")
plt.plot(range(2, 11), silhouette_coefficients )
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficients")
plt.show()


# <div style=" font-size:18px; line_height:160%">
# Metric #2: Calinski Coefficients

# In[34]:


calinski_coefficients = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_parameters)
    kmeans.fit(scaled_features)
    score = calinski_harabasz_score(scaled_features, kmeans.labels_)
    calinski_coefficients.append(score)


# In[35]:


plt.style.use("default")
plt.plot(range(2, 11), calinski_coefficients )
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("calinski Coefficients")
plt.grid()
plt.show()


# <div style=" font-size:14px; line_height:160%">
# By considering the graph of inertias, silhouette scores and calinski scores, we choose number of clusters equal to 4.

# In[36]:


kmeans = KMeans(n_clusters=4, **kmeans_parameters)
kmeans.fit(scaled_features)


# In[37]:


kmeans.labels_


# In[38]:


LABELS = pd.Series(kmeans.labels_, name="LABELS")
LABELS.head(10)


# In[39]:


df1 = pd.concat([df, LABELS], axis=1)


# In[40]:


df1


# In[41]:


the_clusters = sorted(df1['LABELS'].unique())
the_clusters


# In[42]:


plt.figure()
plot = plt.scatter(df1["BALANCE"], df1["PURCHASES"], 
#                    c=kmeans.labels_, 
                   c=df1["LABELS"], 
                   cmap=mpl.cm.get_cmap('viridis', 4), 
                   s=5, 
                  )
plt.title("Balance vs Purchase based on clusters\n")

plt.xlabel("\nBalance", fontsize=15)
plt.ylabel("Purchase\n", fontsize=15)
plt.grid()

plt.colorbar(spacing="uniform", 
             ticks=df1["LABELS"].unique(), 
            )
plt.clim(-0.5, 3.5)


# In[43]:


def plotting(df, x_name, y_name):
    
    colors = mpl.cm.get_cmap('viridis', 4)

    plt.figure(figsize=(10, 10))

    for i in range(len(the_clusters)):

        plt.subplot(2,2,i+1)
        
        plot = plt.scatter(df1[x_name][df1["LABELS"]==i], 
                           df1[y_name][df1["LABELS"]==i], 
                           c=mpl.colors.rgb2hex(colors(i)[:3]),
                           s=5, 
                          )
        
        plt.title(f"Cluster {i+1}\n", fontsize=16)

        plt.xlabel(f"\n{x_name}\n", fontsize=13)
        plt.ylabel(f"{y_name}\n", fontsize=13)
        
        plt.axis([-5, df[x_name].max()+2000, 0, df[y_name].max()+2000])
        plt.grid()
        
        plt.colorbar(spacing="uniform", ticks=df1["LABELS"].unique())
        plt.clim(-0.5, 3.5)        
        
    plt.tight_layout()
    plt.show()


# In[44]:


plotting(df1, df1.columns[0], df1.columns[2])


# In[45]:


results = df1.groupby("LABELS").mean().reset_index().T
results = results.rename(columns={0: "Class 3", 1: "Class 4",  2: "Class 2", 3: "Class 1"})
results = results.sort_values(by="BALANCE", axis=1)
results


#  

#  

# <div style=" font-size:18px; line_height:160%">
# Conclusions:

# <div style=" font-size:16px; line_height:160%">
# Let's discuss about the classes shown in the table.

#  

# <div style=" font-size:16px; line_height:160%">
# Class 1 (Cluster 4 in the plots):<br/>
# <br/>
# <div style=" font-size:14px; line_height:160%">
# Their account balance is the lowest (BALANCE = 894). Their purchase is in the second category (the second buyer). The amount of ONE-OFF purchase and installment purchase is almost equal. Least of all, they take cash from an ATM or bank. Compared to other clusters, they are in the second category of purchase. It means that they are always shopping with their account (card) balance. We said that the amount of their cash and installment purchases is almost the same, but usually they buy more in installments than in cash (in terms of counting). The number of purchase transactions is second among all clusters. The financial level of this group of customers as well as cluster 2 is three times lower than the other two groups because their minimum payment is one third of the other two groups.<br/>
# <br/>  
# <div style=" font-size:16px; line_height:160%">
# Class 2 (Cluster 3 in the plots):<br/>
# <br/> 
# <div style=" font-size:14px; line_height:160%">
# The account balance of this group is slightly more than the previous group. Their one-off purchase is much more than installment purchase. In fact, their installment purchase is very low. That one-off and installment purchases are usually done very rarely, which means that they are not much of a shopper at all. They have about three times more purchases with cash. It can be said that they are not very familiar with technology or that they do not know how to work with a bank card. This group and the previous group have two to two and a half times lower credit limits than the other two groups. This group has the least amount of money to pay for purchase.<br/>
# <br/>
# <div style=" font-size:16px; line_height:160%">
# Class 3 (Cluster 1 in the plots):<br/>
# <br/>    
# <div style=" font-size:14px; line_height:160%">
# This group has a large account balance and therefore a lot of income (they are in the second level). They buy seven times more than group number 1 and twenty eight times more than group 2 (first level)! Their one-off purchase is twice as much as their installment purchase, which is much more than other groups (first level). The amount of cash they take from bank is the same as group 2. They are constantly shopping and their purchase transaction is several dozen times that of other groups. The credit limit from their account is more than others and the reason may be their more transactions. Their minimum payment is three times that of the previous groups. <br/>
# <br/>
# <div style=" font-size:16px; line_height:160%">
# Class 4 (Cluster 2 in the plots):<br/>
# <br/>    
# <div style=" font-size:14px; line_height:160%">
# The account balance of this group is the highest and they always have money in their account, but surprisingly, the amount of their purchases, whether one-off or installments, is very small (one to the last level)! But on the other hand, they take a lot of cash from bank, which they don't like to spend much.<br/>
# 

#  

#  

#  
