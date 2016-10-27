import pandas as p
import pdb
import numpy as np
import us
import json
from collections import Counter
from survey_utils import *

#getting rid of MTURK ids
if False:
 data = p.read_csv("new_poll2.csv")
 headers = open("compact_headers.txt").read().split("\n")[:-1]
 data.columns = headers
 data['id']=xrange(len(data['id']))
 data.to_csv("raw_data.csv")

#read in raw survey data
data = p.read_csv("raw_data.csv")

#some constants to help transform surve data
NO_PREF = 3
candidates = ['hc','dt','gj','js']
pref_map = {'Very satisfied':5,'Satisfied':4,'Neutral':3,'Dissatisfied':2,'Very dissatisfied':1,'I have no preference about this candidate':NO_PREF}
cand_map = {'Hillary Clinton (Democrat)':'hc','Gary Johnson (Libertarian)':'gj','Donald Trump (Republican)':'dt','Jill Stein (Green)':'js'}

census_dataframe = p.DataFrame(census_sample) 

#set to true if you want to regenerate weights
#that try to adapt our distribution of survey-takers
#to the national census
generate_weights = True

#generate weights for our survey data
#to try and match it to the distribution
#of us citizens in the census
if generate_weights:
 survey_intermediates = []
 for _ in xrange(data.shape[0]):
  survey_intermediates.append(create_intermediate(data.loc[_]))

 [make_derivative_properties(k) for k in census_sample]
 census_intermediates = p.DataFrame(census_sample) 

 [make_derivative_properties(k) for k in survey_intermediates]
 survey_intermediates = p.DataFrame(survey_intermediates)
 
 survey_intermediates['weights']=0.0

 for idx in xrange(census_intermediates.shape[0]):
  if idx%1000==0:
   print idx
  dists = nn_weight_increment(census_intermediates.loc[idx],survey_intermediates,neighbors=1)

 weights = np.array(survey_intermediates['weights'])
 np.save("survey_weights.npy",weights)
else:
 #or load presaved weights
 weights = np.load("survey_weights.npy")

#limit how highly influential a single 
weights = np.clip(weights,0,20)

do_weights=False

if do_weights:
 data['weights']=weights
else:
 data['weights']=np.ones(data.shape[0])

show_age = False

#show a plot 
from pylab import *
if show_age:
 _bins = [k*10 for k in range(1,10)]
 subplot(311)
 hist(data['age'],bins=_bins)
 title("Raw unweighted survey data (age)")
 subplot(312)
 hist(data['age'],weights=data['weights'],bins=_bins)
 title("Survey data weighted match census (age)")
 subplot(313)
 hist(census_dataframe['age'],bins=_bins)
 title("Census data (age)")
 show()

#process survey data now...
def process_survey_data(data):
 #only consider people who are registered 
 for idx in xrange(data.shape[0]):
  if (data.loc[idx,'registered_to_vote'].count("not")>0):
   data.loc[idx,'weights']=0.0
  if (data.loc[idx,'intend_to_vote'].count("No")>0):
   data.loc[idx,'weights']=0.0
 
 #map candidate answers to short-form encoding
 vote_counter_dict = {}

 #look both at who voters would vote for today, and 
 #who they truly prefer most of all
 for column in ['vote_today_candidate','preferred_candidate']:
  vote_counter = Counter()
  string_data = data[column]
  row = []
  for idx in xrange(len(string_data)):
   try:
    string = string_data[idx] 
    row.append(cand_map[string])
   except:
    #special case here for "bernie" because he was such a popular write-in
    #no one else had more than a handful of write-in
    #retrospectively, should have anticipated this...
    if string.lower().count("bernie")>0:
     row.append('bs')
    else:
     row.append('other')
   vote_counter[row[-1]]+= data.loc[idx]['weights']
  vote_counter_dict[column]=vote_counter

 print vote_counter_dict

 #map preferences to quantitative encoding
 for candidate in candidates:
  string_data = data[candidate+"_pref"]
  row = np.zeros( (len(string_data),) )
  row[:] = -1
  for idx in xrange(row.shape[0]):
   try:
    row[idx] = pref_map[string_data[idx]]
   except:
    pass

  data.loc[:,"%s_pref_num" % candidate] = p.Series(row,index=data.index)

 #calculate average preference for each candidate
 for candidate in candidates:
  print candidate, np.average( data['%s_pref_num' % candidate],weights=data['weights'])

 #weighted greater-than operator
 def weighted_gt(data,c1,c2):
  weights_total = data['weights'].sum() 
  counter=0.0
  for x in xrange(data.shape[0]): 
   if data.loc[x,'%s_pref_num'%c1] > data.loc[x,'%s_pref_num'%c2]:
    counter+=data.loc[x,'weights']
  return counter/weights_total
  
 #weighted greater-than operator
 def weighted_max(data):
  weights_total = data['weights'].sum() 
  counter=Counter()
  for x in xrange(data.shape[0]):
   vals = [data.loc[x,'%s_pref_num'%candidate] for candidate in candidates]
   if vals.count(max(vals))>1:
    continue
   else:
    print vals
    candidate = candidates[vals.index(max(vals))]
    weight=data.loc[x,'weights']
    counter[candidate]+=weight
  for k in counter:
   print k

 subplot(311)
 title("Hillary Clinton")
 hist(data["hc_pref_num"],weights=data["weights"])
 subplot(312)
 title("Donald Trump")
 hist(data["dt_pref_num"],weights=data["weights"])
 subplot(313)
 title("Gary Johnson")
 hist(data["gj_pref_num"],weights=data["weights"])

 #print float(sum(data["hc_pref_num"]>data["gj_pref_num"]))
 print weighted_gt(data,"hc","dt")
 print weighted_gt(data,"gj","dt")
 #print weighted_gt(data,"hc","gj")
 p#rint weighted_gt(data,"js","dt")

process_survey_data(data)
