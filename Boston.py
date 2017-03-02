# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:22:06 2017

@author: neetakhanuja
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")
#-------------------------------------------------------------------------------
check = True
df = pd.read_csv('.../salary.csv')

mules = df['yd']
df['Years After Graduation'] = np.array(pd.cut(np.array(mules), [0,5,10,15,20,25,30,35,40], labels=['0 - 5','6 - 10','11 - 15','16 - 20','21 - 25','26 - 30','31 - 35','36 - 40']))


df_m = df[df['sx'] == 'male']
df_f = df[df['sx'] == 'female']

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lmm = LinearRegression()
lmf = LinearRegression()
x = pd.DataFrame(df.yd)
y = pd.DataFrame(df.sl)

xm = pd.DataFrame(df_m.yd)
ym = pd.DataFrame(df_m.sl)

xf = pd.DataFrame(df_f.yd)
yf = pd.DataFrame(df_f.sl)


lm.fit(x, y)
lmm.fit(xm, ym)
lmf.fit(xf, yf)

fit = lm.predict(x)
fitm = lm.predict(xm)
fitf = lm.predict(xf)

fig, axs = plt.subplots(1, 3, sharey=True)
ax1 = df.plot(kind='scatter', x='yd', y='sl', ax=axs[0],  figsize=(10, 5))
#plt.plot(df.yd, fit, c='black', linewidth=2)
ax1.set_xlabel("Years After Grad. (Total)")
ax1.set_ylabel('Salary')

ax2 = df_m.plot(kind='scatter', x='yd', y='sl', ax=axs[1])
#plt.plot(df_m.yd, fitm, c='blue', linewidth=2)
ax2.set_xlabel("Years After Grad. (Male)")
ax3 = df_f.plot(kind='scatter', x='yd', y='sl', ax=axs[2])
#plt.plot(df_f.yd, fitf, c='red', linewidth=2)
ax3.set_xlabel("Years After Grad. (Female)")

df_m = df_m.drop(['sx','rk','yr'],axis = 1)
df_m = df_m.sort_values(['dg','Years After Graduation'])
#df_m = df_m.set_index(['dg','Years After Graduation','yd'])
#df_m.unstack()
plt.figure()


'''The swarmplot style in seaborn library is distinctively used for classified data to identify clusters and concentrations'''

df = df.sort_values(["Years After Graduation"])
axis = sns.swarmplot(x = df["Years After Graduation"], y = df["sl"], hue = df["sx"])
axis.set_title('Distribution of Income after Graduation(Sex)')
sns.plt.show()

if(check):
    print ('The intercept for in income after graduation is',lm.intercept_)
    print ('The coefficient for income after graduation is',lm.coef_)
    print ('However, the coefficient for males is',lmm.coef_,'while the same for females is',lmf.coef_)

plt.figure()

axis1 = sns.swarmplot(x = df["Years After Graduation"], y = df["sl"], hue = df["dg"])
axis1.set_title('Distribution of Income after Graduation(Degree)')
sns.plt.show()
