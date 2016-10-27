import pandas as p
import numpy as np
import json
import us

#code to map states to typical regions (e.g. mid-west, south)
region_dict = {}
region_mapping = open("state-mapping.txt","r").read().split("\n")[:-1]
states = [k.split("/")[-1].strip() for k in region_mapping]
region = [k.split(" ")[0].strip() for k in region_mapping]
for idx in xrange(len(states)):
 region_dict[states[idx]]=region[idx]

#load sample of census data to match our distribution to
census_sample = json.load(open("census_sample.json","r"))

#functions to add regions and to split ages into brackets
def regionize(st):
 return region_dict[st] 

def age_bracket(age):
 bracket= int(age)/10
 return bracket

 """
 if age<=29:
  return '29'
 if age<=49:
  return '49'
 if age<=65:
  return '65'
 else:
  return '65+'
 """

#add additional derived fields to data
def make_derivative_properties(s):
 if 'region' not in s:
  s['region'] = regionize(s['state']) 
 s['age_discrete'] = age_bracket(s['age'])

#find closest matches to a given census sample in the survey data
#and increment its weight
def nn_weight_increment(s1,data,neighbors):
 distance(s1,data)
 shuffled = data.reindex(np.random.permutation(data.index))
 shuffled = shuffled.sort_values('dist',kind='mergesort')
 idx = shuffled.index[:neighbors]
 data.loc[idx,'weights']+=1.0

#calculate distances between sample and the survey data
#note: a little naive -- likely a more principled way to
#determine difference between demographic samples...
def distance(s1,data):
 data['dist'] = np.zeros(data.shape[0])

 for key,importance in [('gender',1.0),('race',3.0),('mar',1.0),('region',2.0),('age_discrete',3.0),('educ',4.0)]:
  idx = data[key]!=s1[key]
  data.loc[idx,'dist']+=importance
 return data['dist']  

#map survey data into intermediate matching representation
#that can be compared to census data 
def create_intermediate(sample):
    age = sample['age']
    _gender = sample['gender']
    _mar = sample['marital']

    if 'state' in sample:
     _state = sample['state']
    else:
     _state = ''
   
    _race = sample['race']
    _latino = sample['latino']
    _educ = sample['education']
    

    gender='other'
    try:
     if _gender.count('Male')>0:
      gender='male'
     if _gender.count('Female')>0:
      gender='female'
    except:
     pass

    educ='other'
    if _educ.count("Did not")>0:
     educ = "no-hs"
    if _educ.count("GED")>0:
     educ = "hs"
    if _educ.count("Some college")>0:
     educ = "some-college"
    if _educ.count("Four year")>0:
     educ = "college"
    if _educ.count("Some postgraduate")>0 or _educ.count("Postgraduate")>0:
     educ = "post-grad"

    mar = None
    if _mar.count("Married")>0:
     mar = "married"
    if _mar.count("Widowed")>0:
     mar = "widowed"
    if _mar.count("Separated")>0 or _mar.count("Divorced")>0:
     mar = "divorced"
    if _mar.count("Living")>0 or _mar.count("Never")>0:
     mar = "single"
  
    race = 'other'
    _race = _race.lower()
    if _race.count("white")>0:
     race = 'white'
    if _race.count("asian")>0:
     race = 'asian'
    if _race.count("black")>0:
     race = 'black'
    if _race.count("native american")>0:
     race = 'native'
    if _race.count("hawaiian")>0:
     race = 'hawaiian'
    try:
     if _latino.count("Yes")>0:
      race = 'latino'    
    except:
     pass

    state = str(us.states.lookup(unicode(_state)).abbr)

    intermed = {'age':age,'mar':mar,'state':state,'race':race,'educ':educ,'gender':gender} 
    if 'region' in sample:
     intermed['region']=sample['region']

def pull_sample(data,normalized_weights):
    idx = data.index
    sample = np.random.choice(idx,p=normalized_weights)
    return data.loc[sample] 
