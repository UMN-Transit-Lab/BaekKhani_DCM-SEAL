# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
np.random.seed(5723588)

def synthDataGen(nobs=12500):
    '''
    Class1: e.g., female (s1=='s12'), more concerned about in-vehicle safety and crowding (concerns for buses ('a') are small)
    U1=(0.8+0.2)*(mode=='b')+(-0.1-0.025)*x1+(-0.15)*x2+(-0.05)*x3
    Class2: e.g., male (s1=='s11') and non-white (s2!='s21') more concerned about out-of-vehicle safety except high-incomed neighborhoods
    U2=0.8*(mode=='b')+(-0.1)*x1+(-0.15-0.1)*x2+(-0.05)*x3+1*e23
    Class3: the other remaining demographic groups,showing the baseline behavior
    U3=0.8*(mode=='b')+(-0.1)*x1+(-0.15)*x2+(-0.05)*x3
    Embedding utilities are added uniformly to each class for mode b:
    Uadd=(mode=='b')*(1*e11+0.2*e12-0.2*e13+0.1*e21+0.2*e22+0.4*e23)
    '''
    o1 = np.random.choice(['o11', 'o12', 'o13','o14'], size=nobs, p=[0.2,0.3,0.4,0.1]) #e.g., favorite colors
    s1 = np.random.choice(['s11', 's12', 's13'], size=nobs, p=[0.47, 0.48, 0.05]) # e.g., male female others
    s2 = np.random.choice(['s21', 's22', 's23'], size=nobs, p=[0.78, 0.07, 0.15]) # e.g., white black others
    e1 = np.random.choice(['e11', 'e12', 'e13'], size=nobs, p=[0.47,0.40,0.13]) #e.g., purpose: HBW/S HBO NHB
    P_e2_given_s2 = {#e.g., linc minc hinc (col) for white black others (row) 
        's21': [0.2, 0.6, 0.2],
        's22': [0.4, 0.55, 0.05],
        's23': [0.4, 0.5, 0.1],}
    e2 = []
    for s in s2:
        e2.append(np.random.choice(['e21', 'e22','e23'], p=P_e2_given_s2[s]))
    e2 = np.array(e2)

    dist = np.round(2 + np.random.exponential(scale=4, size=nobs),2)
    x1a = np.round(2 * dist + 0.4 * dist * np.random.randn(nobs))  #ivt for mode a: x1
    x1b = np.round(1.5 * dist + 0.1 * dist * np.random.randn(nobs)) #ivt for mode b: x1
    x2a = np.round(5 + 1 * np.random.randn(nobs)) #ovt for mode a: x2
    x2b = np.round(15 + 2 * np.random.randn(nobs))  #ovt for mode b: x2
    x3a = np.round(1.5 + 0.5 * 1*(np.random.uniform(0,1,nobs)>0.8),2) #fare for mode a: x3
    x3b = np.round(2 + 0.75*np.floor(dist/5),2) #fare for mode b: x3

    # to recover ~90% expected prob: use np.exp(3)/(2+np.exp(3))~0.909
    lc1=np.exp(3*(s1=='s12'))
    lc2=np.exp(3*(s1=='s11')*(s2!='s21'))
    lc3=np.exp(3*((lc1+lc2)==2)) # individuals whose lc1 and lc2 were not affected from the above two rows
    pr1=lc1/(lc1+lc2+lc3)
    pr2=lc2/(lc1+lc2+lc3)
    pr3=lc3/(lc1+lc2+lc3)
    prs=np.transpose(np.array([pr1,pr2,pr3]))
    LC = np.array([np.random.choice(len(row), p=row) for row in prs])+1 # Class assignment

    dfC=pd.concat([pd.DataFrame({'id':(1+np.arange(nobs)),'alt':'a','x1':x1a,'x2':x2a,'x3':x3a}),
                   pd.DataFrame({'id':(1+np.arange(nobs)),'alt':'b','x1':x1b,'x2':x2b,'x3':x3b})]).sort_values(['id','alt'])
    dfN=pd.DataFrame({'id':(1+np.arange(nobs)),'s1':s1,'s2':s2,'e1':e1,'e2':e2,'o1':o1,'dist':dist,'LC':LC})

    ASC=0.6 #identified for mode b (train)
    bx1=-0.1 #ivt
    bx2=-0.15 #ovt
    bx3=-0.05 #fare
    be11=1 #HBW/S on mode b
    be12=0.2 #HBO on mode b
    be13=-0.2 #NHB on mode b
    be21=0.1 #linc on mode b
    be22=0.2 #minc on mode b
    be23=0.4 #hinc on mode b
    L1ASC=0.15 #ASC add-on for LC1
    L1x1=-0.025 #ivt add-on for LC1
    L2x2 = -0.1 #ovt add-on for LC2
    L2e23 = -0.5 #hinc utility penalty revert
    eps=0.15

    #endogeneity defined for LC1 for interaction with x1 or iv, LC2 for interaction with ovt but except highincome LC2
    df=pd.merge(dfC,dfN,on='id')
    df['utility']=(df['alt']=='b')*(ASC+L1ASC*(df['LC']==1))+df['x1']*(bx1+L1x1*(df['LC']==1))+df['x2']*(bx2+L2x2*(df['LC']==2))+df['x3']*bx3
    df['utility']+=(df['alt']=='b')*(be11*(df['e1']=='e11')+be12*(df['e1']=='e12')+be13*(df['e1']=='e13')+be21*(df['e2']=='e21')+be22*(df['e2']=='e22')+be23*(df['e2']=='e23'))
    df['utility']+=L2e23*(df['alt']=='b')*(df['LC']==2)*(df['e2']=='e23')
    df['utility']+=eps*np.random.randn(len(df)) #random error

    dfa=df.loc[df.alt=='a','utility']
    dfb=df.loc[df.alt=='b','utility']
    dfchoice=pd.DataFrame({'id':(1+np.arange(nobs)),'choseA':dfa.values>=dfb.values})

    df=pd.merge(df,dfchoice,on='id')
    df['match']=0

    df.loc[(df.alt=='a') & (df.choseA),'match']=1
    df.loc[(df.alt=='b') & (~(df.choseA)),'match']=1
    df=df.drop(columns=['choseA'])

    df['alt']=1*(df.alt=='b') #revert to 0 1 alts
    df=df.rename(columns={'id':'chid'})
    return df

if __name__=="__main__":
    dfSynth=synthDataGen(nobs=12500)
    # some inspection
    dfPerson=dfSynth.loc[dfSynth.alt==0,:].copy().reset_index()
    print(dfPerson.groupby(['LC']).size()/len(dfPerson))
    print(dfSynth.groupby(['alt'])['match'].sum()/len(dfPerson))
    print(dfSynth.groupby(['LC','alt'])['match'].sum()/dfPerson.groupby(['LC']).size().repeat(2).values)
    enc=OneHotEncoder(sparse_output=False)
    dfPerson2=pd.DataFrame(enc.fit_transform(dfPerson[['s1','s2','e1','e2','o1']]),columns=enc.get_feature_names_out())
    dfPerson3=pd.concat([dfPerson[['LC']],dfPerson2],axis=1)
    dfPerson4=pd.concat([dfPerson3.groupby('LC').sum(),(dfPerson.groupby(['LC']).size())],axis=1)
    dfPerson5=round(dfPerson4.div(dfPerson4[0],axis=0),3).iloc[:,:-1]
    print(dfPerson5.transpose())
    # save the file. it will remove the pinned dfIn.csv file in the directory for the replication
    #dfSynth.to_csv('dfIn.csv',index=False)
