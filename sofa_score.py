#!/usr/bin/env python
# coding: utf-8

# In[57]:


import os 
import numpy as np # linear algebra
import datacompy, pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'# data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn import metrics # for the check the error and accuracy of the model
get_ipython().run_line_magic('matplotlib', 'inline')


# # For Regression

# In[2]:


from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor


# # For Classification

# In[3]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


# # Setting Environment

# In[4]:


os.chdir("E:\zbliss")
dataset = pd.read_csv("admissions.csv")


# # Visualize first five row of dataset

# In[5]:


dataset.head()


# # Visualize last five row of dataset

# In[6]:


dataset.tail()


# # Data Cleaning and Preprocessing

# # Shape of Datase

# In[7]:


print("The Dataset has {} rows and {} columns".format(dataset.shape[0], dataset.shape[1]))


# In[8]:


dataset.columns


# # Droping the duplicates if available

# In[9]:


dataset.drop_duplicates()


# In[10]:


dataset.shape


# # After droping the duplicates we can see the shape of dataset is remains same. so we can say that the dataset don't have any duplicates rows.

# # Finding null values

# In[11]:


dataset.isnull().sum()


# # Percentage of missing data in different columns

# In[12]:


percent_missing = dataset.isnull().sum() * 100 / len(dataset)
missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
missing_value_df


# In[13]:


DF = dataset.dropna(how = 'any')
DF.shape


# # Total Number of Admissions 

# In[14]:


dataset.admission_type
TA=len(dataset.admission_type)
print (" Total Number of  Admissions= ",TA)


# # Unique Admissions types

# In[15]:


dataset.admission_type.unique()


# #  Number of Emergency Admissions

# In[16]:


dataset.loc[dataset.admission_type=='EMERGENCY']
TEA= len(dataset.loc[dataset.admission_type=='EMERGENCY'])
print (" Total Number of Emergency Admissions= ",TEA)


# In[17]:


dataset.loc[dataset.admission_type=='ELECTIVE']
TELA= len(dataset.loc[dataset.admission_type=='ELECTIVE'])
print (" Total Number of ELECTIVE Admissions= ",TELA)


# In[18]:


dataset.loc[dataset.admission_type=='URGENT']
TUA= len(dataset.loc[dataset.admission_type=='URGENT'])
print (" Total Number of URGENT Admissions= ",TUA)


# In[19]:


dataset.loc[dataset.admission_type=='NEWBORN']
TNBA= len(dataset.loc[dataset.admission_type=='NEWBORN'])
print (" Total Number of NEWBORN Admissions= ",TNBA)


# # Percentage of Emergency admission

# In[20]:


print("Percentage of Emergency admissions=", (100* (TEA)/ TA))


# # Ploting a graph of the count of various types of Admissions as a bar graph 

# In[21]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
Unique_Admissions_types= ['EMERGENCY', 'ELECTIVE', 'NEWBORN', 'URGENT']
count = [TEA,TELA,TNBA,TUA]
ax.bar(Unique_Admissions_types,count)
plt.show()


# # SOFA SCORE- Sepsis-related Organ Failure Assessment Score
# 
# Sepsis-related Organ Failure Assessment , previously known as the sepsis-related organ failure assessment score, is used to track a person's status during the stay in an intensive care unit (ICU) to determine the extent of a person's organ function or rate of failure. 
# It is stressed that the score is not meant as a direct predictor of mortality but rather a measure of morbidity.
# 
# The score is based on six different scores, one each for the respiratory, cardiovascular, hepatic, coagulation, renal and neurological systems.
# 
# Each system’s result is given a score from 0–4 which causes the scores range to be from 0–24 with 0 being the least severe condition and 24 being the most severe condition and an average having >90% chance of mortality.
# 
# 

# # 1. PaO2/FiO2(mmHg)- Respiration
# 
# PaO2/FiO2>=400 SOFAscore=0
# 
# PaO2/FiO2< 400 SOFAscore=1
# 
# PaO2/FiO2<300 SOFAscore=2
# 
# PaO2/FiO2<200 & ventend Mechanically SOFAscore=3
# 
# PaO2/FiO2<100 & ventend Mechanically SOFAscore=4

# # 2. Creatine(mg/dl)- Kidney Score (Renal)
# 
# creatinine_max >= 5.0  SOFAscore=4
# 
# urineoutput < 200 SOFAscore= 4
# 
# creatinine_max >= 3.5 & creatinine_max < 5.0 SOFAscore 3
# 
# urineoutput < 500 SOFAscore3
# 
# creatinine_max >= 2.0 and creatinine_max < 3.5 SOFAscore 2
# 
# creatinine_max >= 1.2 and creatinine_max < 2.0 SOFAscore 1

# # 3.Neurovascular Score (GCS(Glass Coma Scale))
# 
# mingcs < 6 SOFAscore= 4
# 
# mingcs >= 6 and mingcs <= 9 SOFAscore= 3
# 
# mingcs >= 10 and mingcs <= 12 SOFAscore= 2
# 
# mingcs >= 13 and mingcs <= 14 SOFAscore=1
# 

# # 4. Coagulation
# 
# platelet_min < 20 SOFAscore= 4
# 
# platelet_min < 50 SOFAscore= 3
# 
# platelet_min < 100 SOFAscore= 2
# 
# platelet_min < 150 SOFAscore= 1
# 
# 

# # 4. Liver Score (Bilirubin in mg/dL)
# 
# bilirubin_max >= 12.0 SOFAscore=  4
# 
# bilirubin_max >= 6.0 SOFAscore=  3
# 
# bilirubin_max >= 2.0 SOFAscore=  2
# 
# bilirubin_max >= 1.2 SOFAscore=  1

# # Cardiovascular
# 
# meanbp_min < 70 SOFAscore= 1
# 
# rate_dopamine > 0 or rate_dobutamine > 0 SOFAscore= 2
# 
# rate_dopamine > 5 or rate_epinephrine <= 0.1 or rate_norepinephrine <= 0.1 SOFAscore= 3
# 
# rate_dopamine > 15 or rate_epinephrine > 0.1 or rate_norepinephrine > 0.1 SOFAscore= 4
# 
# 

# In[22]:


import psycopg2 as pg
import sqlalchemy


# In[23]:


host = '3.7.155.14' 
port = 5432
user = 'datascientist'
password = 'candidate' 
name = 'mimic'
schema = 'mimiciii'


# In[24]:


connect = pg.connect(database = name , user = user , password = password,host =host ,port = port  )
cur = connect.cursor()
cur.execute("select count(*) from public.wt")
admission_count = int(cur.fetchall()[0][0])
print('Total number of Admissions = %d' %admission_count  )


# In[25]:


engine = sqlalchemy.create_engine("postgresql://datascientist:candidate@3.7.155.14:5432/mimic")


# In[26]:


blood_gas_art = pd.read_sql_table('blood_gas_first_day_arterial',engine,schema='public')
blood_gas_art.head()


# In[27]:


resp=blood_gas_art[['icustay_id','subject_id','pao2fio2',]]
resp.head()


# In[28]:


labs_first = pd.read_sql_table('labs_first_day',engine,schema='public')
labs=labs_first[['icustay_id','bilirubin_max','creatinine_max', 'platelet_min']]
labs.head()


# In[29]:


vaso = pd.read_sql_table('vaso_mv',engine,schema='public')


# In[30]:


echo=pd.read_sql_table('echo_dat',engine,schema='public')
bp=echo[['subject_id','bp']]


# In[31]:


gcs=pd.read_sql_table('gcs_first_day',engine,schema='public')
neuro=gcs[['subject_id','icustay_id','mingcs']]


# In[32]:


left = labs.set_index(['icustay_id'])
right =resp.set_index(['icustay_id'])
rlu=left.join(right,lsuffix='_', rsuffix='_')
rlu.head()


# In[33]:


rul=labs.merge(resp, left_on='icustay_id', right_on='icustay_id', suffixes=(False, False))
rul.head()


# In[34]:


nrl=rul.merge(neuro, left_on='icustay_id', right_on='icustay_id', suffixes=('_left', '_right'))
nrl.tail()


# In[35]:


vnr=nrl.merge(vaso, left_on='icustay_id', right_on='icustay_id',  suffixes=(False, False))
vnr.tail()


# In[36]:


bvnr=vnr.merge(bp, left_on='subject_id_left', right_on='subject_id',  suffixes=(False, False))
bvnr.tail()


# In[132]:


conditions1 = [
    (bvnr['platelet_min'] <20),
    (bvnr['platelet_min'] <50),
    (bvnr['platelet_min'] <100),
    (bvnr['platelet_min'] <150)
    ]

coagulation = ['4', '3', '2', '1']
bvnr['coagulation'] = np.select(conditions1, coagulation)
bvnr.head()


# In[38]:


conditions2 = [
    (bvnr['pao2fio2'] <100),
    (bvnr['pao2fio2'] <200),
    (bvnr['pao2fio2'] <300),
    (bvnr['pao2fio2'] <400)
    ]

respiration = ['4', '3', '2', '1']
bvnr['respiration'] = np.select(conditions2, respiration)
bvnr.head()


# In[109]:


conditions3 = [
    (bvnr['bilirubin_max'] <12.0),
    (bvnr['bilirubin_max'] <6.0),
    (bvnr['bilirubin_max'] <2.0),
    (bvnr['bilirubin_max'] <1.2)
    ]

liver = ['4', '3', '2', '1']
bvnr['liver'] = np.select(conditions3, liver)
bvnr.head()

 


# In[155]:


conditions4 = [
    (bvnr['creatinine_max'] <5.0),
    (bvnr['creatinine_max'] <3.5),
    (bvnr['creatinine_max'] <2.0),
    (bvnr['creatinine_max'] <1.2)
    ]

renal = ['4', '3', '2', '1']
bvnr['renal'] = np.select(conditions4, renal)
bvnr.head()


# In[156]:


conditions5 = [
    (bvnr['mingcs'] >=13) & (bvnr['mingcs'] <=14),
    (bvnr['mingcs'] >=10) &  (bvnr['mingcs'] <=12),
    (bvnr['mingcs'] >=6) &  (bvnr['mingcs'] <=9),
    (bvnr['mingcs'] <6)
    ]

cns = ['1', '2', '3', '4']
bvnr['cns'] = np.select(conditions5, cns)
bvnr.head()


# In[157]:


conditions6 = [
    (bvnr['rate_dopamine' ]> 15) | (bvnr['rate_epinephrine']>0.1) | (bvnr['rate_norepinephrine']> 0.1),
    (bvnr['rate_dopamine' ]> 5) | (bvnr['rate_epinephrine']<=0.1) | (bvnr['rate_norepinephrine']<= 0.1),
    (bvnr['rate_dopamine' ]> 0) | (bvnr['rate_dobutamine']>0),
    (bvnr['bp'] <70)
    ]

Cardiovascular = ['4', '3', '2', '1']
bvnr['Cardiovascular'] = np.select(conditions6, Cardiovascular)
bvnr.head()



# In[158]:


SOFA=bvnr[['icustay_id','coagulation','respiration','liver','renal','cns','Cardiovascular']]
SOFA.head()


# In[159]:


SOFA.isnull().sum()


# In[160]:


SOFA.shape


# In[161]:


a = np.arange(352716).reshape(50388,7)
 a.sum(axis=1)


# In[171]:


Sum=sum['coagulation', 'respiration', 'liver', 'renal', 'cns', 'Cardiovascular']


# In[113]:



SOFA=['coagulation', 'respiration', 'liver', 'renal', 'cns', 'Cardiovascular']
SOFA['SOFAscore']=SOFA[Sum].sum(axis=1)
SOFA.head()


# In[ ]:





# In[139]:


column_list = list(SOFA)


# In[140]:



column_list.remove("icustay_id")


# In[141]:



column_list


# In[146]:


type(column_list)


# In[162]:


SOFA = np.arange(352716).reshape(50388,7)


# In[165]:


SOFAscore  =np.sum(column_list),axis=1


# In[151]:


SOFA['SOFAscore']=SOFA[column_list]np.sum(column_list,axis=1)
SOFA.head()


# In[ ]:


.sum(axis=1)
array([18, 22, 26])


# In[163]:


SOFA['SOFAscore']=SOFA[column_list].sum(axis=1)
SOFA.head()


# In[126]:


SOFA.reset_index(drop=True)
SOFAscore=[SOFA.coagulation+SOFA.respiration+SOFA.liver+SOFA.renal+SOFA.cns+SOFA.Cardiovascular]


# In[127]:


SOFA['SOFAscore'] = SOFAscore
SOFAscore.head()


# In[72]:



SOFAscore=[SOFA.coagulation+SOFA.respiration+SOFA.liver+SOFA.renal+SOFA.cns+SOFA.Cardiovascular]




  
# Define a dictionary containing Students data 
data = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj'], 
        'Height': [5.1, 6.2, 5.1, 5.2], 
        'Qualification': ['Msc', 'MA', 'Msc', 'Msc']} 
  
# Convert the dictionary into DataFrame 
df = pd.DataFrame(SOFA) 
  
# Using DataFrame.insert() to add a column 
df.insert(SOFAscore, True) 
  
# Observe the result 
df


# In[ ]:


SOFA["SOFAscore"] =SOFA.column_list.sum(axis=1)
SOFA.tail()
print(df)



importing pandas as pd 
import pandas as pd 
  
# Creating the DataFrame 
df = pd.DataFrame({'Date':['10/2/2011', '11/2/2011', '12/2/2011', '13/2/2011'], 
                    'Event':['Music', 'Poetry', 'Theatre', 'Comedy'], 
                    'Cost':[10000, 5000, 15000, 2000]}) 
  
# Create a new column 'Discounted_Price' after applying 
# 10% discount on the existing 'Cost' column. 
  
# create a new column 
df['Discounted_Price'] = df['Cost'] - (0.1 * df['Cost']) 
  
# Print the DataFrame after  
# addition of new column 
print(df) 


# In[1]:


connect = pg.connect(database = name , user = user , password = password,host =host ,port = port  )
cur = connect.cursor()

query = """with wt AS(SELECT ie.icustay_id             
    ,avg(CASE 
        WHEN itemid IN (762, 763, 3723, 3580, 226512)
          THEN valuenum                                      
        WHEN itemid IN (3581)
          THEN valuenum * 0.45359237
        WHEN itemid IN (3582)
          THEN valuenum * 0.0283495231
        ELSE null
      END) AS weight

  FROM mimiciii.icustays As ie
  left join mimiciii.chartevents As c
    on ie.icustay_id = c.icustay_id
  WHERE valuenum IS NOT NULL
  AND itemid IN
  (
    762, 763, 3723, 3580,                     
    3581,                                    
    3582,                                     
    226512                                                      
  )
  AND valuenum != 0
  and charttime between DATETIME_SUB(ie.intime, INTERVAL '1' DAY) and DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
                                                               
  AND (c.error IS NULL OR c.error = 0)
  group by ie.icustay_id
)
                                     
, echo2 as(
  select ie.icustay_id, avg(weight * 0.45359237) as weight
  FROM mimiciii.icustays As ie
  left join mimiciii.echodata As echo
    on ie.hadm_id = echo.hadm_id
    and echo.charttime > DATETIME_SUB(ie.intime, INTERVAL '7' DAY)
    and echo.charttime < DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
  group by ie.icustay_id
)
, vaso_cv as
(
  select ie.icustay_id
                                        
    , max(case
            when itemid = 30047 then rate / coalesce(wt.weight,ec.weight) 
            when itemid = 30120 then rate         
            else null
          end) as rate_norepinephrine

    , max(case
            when itemid =  30044 then rate / coalesce(wt.weight,ec.weight)       
            when itemid in (30119,30309) then rate                              
            else null
          end) as rate_epinephrine

    , max(case when itemid in (30043,30307) then rate end) as rate_dopamine
    , max(case when itemid in (30042,30306) then rate end) as rate_dobutamine

  FROM mimiciii.icustays As ie
  inner join mimiciii.inputevents_cv As cv
    on ie.icustay_id = cv.icustay_id and cv.charttime between ie.intime and DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
  left join wt
    on ie.icustay_id = wt.icustay_id
  left join echo2 ec
    on ie.icustay_id = ec.icustay_id
  where itemid in (30047,30120,30044,30119,30309,30043,30307,30042,30306)
  and rate is not null
  group by ie.icustay_id
)
, vaso_mv as
(
  select ie.icustay_id
                                       
    , max(case when itemid = 221906 then rate end) as rate_norepinephrine
    , max(case when itemid = 221289 then rate end) as rate_epinephrine
    , max(case when itemid = 221662 then rate end) as rate_dopamine
    , max(case when itemid = 221653 then rate end) as rate_dobutamine
  FROM mimiciii.icustays As ie
  inner join mimiciii.inputevents_mv As mv
    on ie.icustay_id = mv.icustay_id and mv.starttime between ie.intime and DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
  where itemid in (221906,221289,221662,221653)
                                        
  and statusdescription != 'Rewritten'
  group by ie.icustay_id
)
, pafi1 as
(
                                        
  select bg.icustay_id, bg.charttime
  , pao2fio2
  , case when vd.icustay_id is not null then 1 else 0 end as isvent
  from mimiciii.bloodgasfirstdayarterial As bg
  left join mimiciii.ventdurations As vd
    on bg.icustay_id = vd.icustay_id
    and bg.charttime >= vd.starttime
    and bg.charttime <= vd.endtime
  order by bg.icustay_id, bg.charttime
)
, pafi2 as
(
                                
  select icustay_id
  , min(case when isvent = 0 then pao2fio2 else null end) as pao2fio2_novent_min
  , min(case when isvent = 1 then pao2fio2 else null end) as pao2fio2_vent_min
  from pafi1
  group by icustay_id
)
                                 
, scorecomp as
(
select ie.icustay_id
  , v.meanbp_min
  , coalesce(cv.rate_norepinephrine, mv.rate_norepinephrine) as rate_norepinephrine
  , coalesce(cv.rate_epinephrine, mv.rate_epinephrine) as rate_epinephrine
  , coalesce(cv.rate_dopamine, mv.rate_dopamine) as rate_dopamine
  , coalesce(cv.rate_dobutamine, mv.rate_dobutamine) as rate_dobutamine

  , l.creatinine_max
  , l.bilirubin_max
  , l.platelet_min

  , pf.pao2fio2_novent_min
  , pf.pao2fio2_vent_min

  , uo.urineoutput

  , gcs.mingcs
FROM mimiciii.icustays As ie
left join vaso_cv cv
  on ie.icustay_id = cv.icustay_id
left join vaso_mv mv
  on ie.icustay_id = mv.icustay_id
left join pafi2 pf
 on ie.icustay_id = pf.icustay_id
left join mimiciii.vitalsfirstday As v
  on ie.icustay_id = v.icustay_id
left join mimiciii.labsfirstday As l
  on ie.icustay_id = l.icustay_id
left join mimiciii.uofirstday As uo
  on ie.icustay_id = uo.icustay_id
left join mimiciii.gcsfirstday As gcs
  on ie.icustay_id = gcs.icustay_id
)
, scorecalc as
(
                                
select icustay_id

  , case
      when pao2fio2_vent_min   < 100 then 4
      when pao2fio2_vent_min   < 200 then 3
      when pao2fio2_novent_min < 300 then 2
      when pao2fio2_novent_min < 400 then 1
      when coalesce(pao2fio2_vent_min, pao2fio2_novent_min) is null then null
      else 0
    end as respiration

  , case
      when platelet_min < 20  then 4
      when platelet_min < 50  then 3
      when platelet_min < 100 then 2
      when platelet_min < 150 then 1
      when platelet_min is null then null
      else 0
    end as coagulation


  , case

        when bilirubin_max >= 12.0 then 4
        when bilirubin_max >= 6.0  then 3
        when bilirubin_max >= 2.0  then 2
        when bilirubin_max >= 1.2  then 1
        when bilirubin_max is null then null
        else 0
      end as liver


  , case
      when rate_dopamine > 15 or rate_epinephrine >  0.1 or rate_norepinephrine >  0.1 then 4
      when rate_dopamine >  5 or rate_epinephrine <= 0.1 or rate_norepinephrine <= 0.1 then 3
      when rate_dopamine >  0 or rate_dobutamine > 0 then 2
      when meanbp_min < 70 then 1
      when coalesce(meanbp_min, rate_dopamine, rate_dobutamine, rate_epinephrine, rate_norepinephrine) is null then null
      else 0
    end as cardiovascular


  , case
      when (mingcs >= 13 and mingcs <= 14) then 1
      when (mingcs >= 10 and mingcs <= 12) then 2
      when (mingcs >=  6 and mingcs <=  9) then 3
      when  mingcs <   6 then 4
      when  mingcs is null then null
  else 0 end
    as cns


  , case
    when (creatinine_max >= 5.0) then 4
    when  urineoutput < 200 then 4
    when (creatinine_max >= 3.5 and creatinine_max < 5.0) then 3
    when  urineoutput < 500 then 3
    when (creatinine_max >= 2.0 and creatinine_max < 3.5) then 2
    when (creatinine_max >= 1.2 and creatinine_max < 2.0) then 1
    when coalesce(urineoutput, creatinine_max) is null then null
  else 0 end
    as renal
  from scorecomp
)
select ie.subject_id, ie.hadm_id, ie.icustay_id
                                                                   
  , coalesce(respiration,0)
  + coalesce(coagulation,0)
  + coalesce(liver,0)
  + coalesce(cardiovascular,0)
  + coalesce(cns,0)
  + coalesce(renal,0)
  as SOFA
, respiration
, coagulation
, liver
, cardiovascular
, cns
, renal
FROM mimiciii.icustays As ie
left join scorecalc s
  on ie.icustay_id = s.icustay_id
order by ie.icustay_id"""

sofa_score = pd.read_sql_query(query,connect)
sql.close()


# In[ ]:


Sofa_score


# In[ ]:


sofa_score['SOFA'] = sofa_score.sofa
sofa_score.drop('sofa',axis =1, inplace = True)
sofa_score.head()


# In[ ]:


plt.figure(figsize=(10,10))
ax = sns.histplot(x= 'SOFA' , data=sofa_score)
ax.set_title('Histogram Plot For Sofa Score')


# In[ ]:


df_expls = pd.read_sql(query_schema+'select * from explicit_sepsis', con)
df_expls = df_expls.groupby('subject_id')[['severe_sepsis', 'septic_shock', 'sepsis']].max()
df_expls.sum()


# In[ ]:




