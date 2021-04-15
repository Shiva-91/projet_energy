#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns 
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


# on va utiliser le jeu de données national pour prédire une estimation des émissions de carbone générées par la production 
# d'électricité en France métropolitaine
nat = pd.read_csv(r'C:\Users\LENOVO\Desktop\data analyst\Projet\dataclean\eco2mix-national-cons-def.csv', sep =';')
nat.head()


# In[12]:


nat.info()


# In[13]:


# le but est de travailler sur les émissions du CO2, on garde uniquement les taux de CO2, la date et la nature 
# pour eventuellement supprimer les données non-définitives
national=nat[["Nature","Date","Taux de CO2 (g/kWh)"]]
#new = old.drop('B', axis=1)
national.dropna()
national = national.dropna(subset=['Taux de CO2 (g/kWh)'])
national.isna().sum()


# In[14]:


national=national.rename(columns={"Taux de CO2 (g/kWh)": "taux_Co2"})


# In[ ]:





# In[15]:


# calcul de la somme des émissions par jours
d = national.groupby(by='Date').agg({'taux_Co2':'sum'})


# In[16]:


d.index = pd.to_datetime(d.index)
print(d.index)


# In[17]:


plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':60})
plt.plot(d)
plt.title('Evolution des des émissions journalièere de carbone générées par la production d\'électricité en France', fontsize=18)


# In[18]:


d['2020'].resample('W').mean().plot()
plt.show()
# on voie clairement une nette baisse des émissions du CO2 pendant le premier confinement et même après, les émissions de CO2
# ont retrouvées leurs taux d'avant confinement vers mi juin


# In[33]:


#subts = d.loc['2013':'2019']


# In[19]:


CO2_month = d.resample('M').agg(['mean'])
# on calcul la moyenne par mois des émissions par mois


# In[21]:


plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':60})
plt.plot(CO2_month)
plt.title('Evolution des des émissions mensuelle de carbone générées par la production d\'électricité en France', fontsize=18)


# In[22]:


# visualisation de l'autocorrélation 
pd.plotting.autocorrelation_plot(CO2_month)


# In[24]:


# appliquer une  différenciation d'ordre 1
CO2_month_1 = CO2_month.diff().dropna()
pd.plotting.autocorrelation_plot(CO2_month_1)
sm.tsa.stattools.adfuller(CO2_month_1)


# In[25]:


CO2_month_2 = CO2_month_1.diff(periods = 12).dropna()
pd.plotting.autocorrelation_plot(CO2_month_2);
#il y a pas de différence entre une d de 1 et de 2, on reste sur 1 pour ne pas trop différencier notre série


# In[26]:


#le test augmenté de Dickey-Fuller.
sm.tsa.stattools.adfuller(CO2_month_2)


# In[73]:


# Pvalue inferieur au seuil de 5% même si c'est relativement grande. 


# In[27]:


from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
plt.figure(figsize= (14,7))
plt.subplot(121)
plot_acf(CO2_month_2, lags = 36, ax=plt.gca())
plt.subplot(122)
plot_pacf(CO2_month_2, lags = 36, ax=plt.gca())
plt.show()


# In[28]:


model=sm.tsa.SARIMAX(CO2_month,order=(1,1,2),seasonal_order=(1,1,1,12))
results=model.fit()
print(results.summary())


# In[ ]:


# le test de Ljung-Box avec une Pvalue=0;63 on ne peut pas rejetté l'hypothèse null, La résidu est un bruit blanc cad l'esperance
# et les covariances de notres séries sont null
#quand au test de Jarque-Bera la Pvalue est inferieur au seuil de 5% les résidus ne sont pas gaussien


# In[29]:


# vérification des résidus
model_fit = model.fit(disp=0)
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()


# In[31]:


pred = results.predict(110,230 )
predic =pred.to_frame()
predic =predic.rename(columns={"predicted_mean": "taux_Co2"})
#CO2_month =predic.rename(columns={"(taux_Co2 mean)": "taux_Co2"})
#(taux_Co2, mean)


# In[193]:


#predic.index.name="Date"
#CO2_month.index.name="Date"
#CO2_month.head()


# In[192]:


#plt.rcParams.update({'figure.figsize':(12,8), 'figure.dpi':60})
#CO2_month.plot(color='blue', grid=True)
#predic.plot(color='red',grid=True, secondary_y=True)
#plt.show();


# In[32]:


#result = pd.concat([CO2_month, predic], axis=0)
#result = CO2_month.append(predic)
ax = CO2_month.plot(color='blue', grid=True)
predic.plot(ax=ax,color='red',grid=True)


# In[205]:


# les résultats montrent une baisse des des émissions de carbon au fil des années jusqu'à atteindre le 0 vers 2029, 
# ce qui est peu probable, le modèle semble donner des fausses prédictions, celà est probablement due à l'inclusion de 2020 et
#2021 pendant la crise du covid, toutefois la baisse en émissions du CO2 commence bien avant en mi 2018, peut être le modèle donne
# des mauvaises prédictions à cause de l'utilisation des données  mensuelle et pas journalière


# In[ ]:





# In[ ]:




