import pandas as p
import pdb
import numpy as np
import us
import json
from collections import Counter
from survey_utils import *

#set to true if you want to regenerate weights
#that try to adapt our distribution of survey-takers
#to the national census
generate_weights = False

#do we want to use the calculated weights, or weight each response equally?
do_weights=True

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
NO_PREF = -1
candidates = ['hc','dt','gj','js']
pref_map = {'Very satisfied':5,'Satisfied':4,'Neutral':3,'Dissatisfied':2,'Very dissatisfied':1,'I have no preference about this candidate':NO_PREF}
cand_map = {'Hillary Clinton (Democrat)':'hc','Gary Johnson (Libertarian)':'gj','Donald Trump (Republican)':'dt','Jill Stein (Green)':'js'}
cand_names = dict ( zip([cand_map[k] for k in cand_map.keys()],[k.split("(")[0] for k in cand_map.keys()]) )

census_dataframe = p.DataFrame(census_sample) 

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
#weights = np.clip(weights,0,20)


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
  vote_counter = {'raw':Counter(),'weighted':Counter()}
  string_data = data[column]
  row = []
  for idx in xrange(len(string_data)):
   try:
    string = string_data[idx] 
    row.append(cand_map[string])
   except:
    #special case here for "bernie" because he was a 2-3% write-in
    #no one else had more than a handful of write-in
    #retrospectively, should have anticipated this...
    if string.lower().count("bernie")>0:
     row.append('bs')
    else:
     row.append('other')
   vote_counter['weighted'][row[-1]]+= data.loc[idx]['weights']
   vote_counter['raw'][row[-1]]+= 1

  for k in vote_counter['weighted'].keys():
   vote_counter['weighted'][k] /= data['weights'].sum()
   vote_counter['raw'][k] /= float(data.shape[0])
   
  vote_counter_dict[column]=vote_counter
 
 plot_votes = False 
 if plot_votes:
  plt.clf()
  for cand in cand_names:
   target_dict = vote_counter_dict['vote_today_candidate']['weighted']
   #target_dict = vote_counter_dict['preferred_candidate']['weighted']
   objects = cand_names.keys()
   labels = [cand_names[k] for k in objects]
   y_pos = np.arange(len(objects))
   performance = [100*target_dict[k] for k in objects]
   print performance
   plt.title("Who would you vote for today?")
   #plt.title("Who would you most prefer be president?")
   plt.bar(y_pos, performance, align='center', alpha=0.5)
   plt.xticks(y_pos, labels)
   plt.ylabel('Percentage of Respondants')
   plt.savefig("vote_today.png") 
   #plt.savefig("prefer.png") 
   plt.show()
 
 plot_most_preferred = False
 if plot_most_preferred:
   plt.clf()
   target_dict1 = vote_counter_dict['vote_today_candidate']['weighted']
   target_dict2 = vote_counter_dict['preferred_candidate']['weighted']
   cand_subset = ['gj','js']
   objects1 = cand_subset
   labels1 = ["Vote "+ cand_names[k] for k in objects1]
   objects2 = cand_subset
   labels2 = ["Prefer "+ cand_names[k] for k in objects2]

   performance1 = [100*target_dict1[k] for k in objects1]
   performance2 = [100*target_dict2[k] for k in objects2]
  
   objects = []
   performance = []
   labels =[]
   for k in xrange(len(objects1)):
    objects.append(objects1[k])
    objects.append(objects2[k])
    labels.append(labels1[k])
    labels.append(labels2[k])
    performance.append(performance1[k])
    performance.append(performance2[k])

   y_pos = np.arange(len(objects))

   print performance
   plt.title("Divergence between Vote and Preference?")
   plt.bar(y_pos, performance, align='center', alpha=0.5,color=['r','g','r','g'])
   plt.xticks(y_pos, labels)
   plt.ylabel('Percentage of Respondants')
   plt.savefig("vote_pref1.png") 
   plt.show()


 for column in ['affiliation']:
  option_counter = Counter()
  option_counter_unweighted = Counter()
  string_data = data[column]
  for idx in xrange(len(string_data)):
   option_counter[string_data[idx]]+= data.loc[idx]['weights']
   option_counter_unweighted[string_data[idx]]+= 1
 
 for k in option_counter:
  option_counter[k]/=data['weights'].sum()
  option_counter_unweighted[k]/=float(data.shape[0])

 plot_affiliation = False
 if plot_affiliation:
  plt.subplot(211)
  objects = option_counter.keys()
  y_pos = np.arange(len(objects))
  performance = [100*option_counter[k] for k in objects]
  plt.title("Reweighted Sample")
  plt.bar(y_pos, performance, align='center', alpha=0.5)
  plt.xticks(y_pos, objects)
  plt.ylabel('Percentage of Respondants')
  plt.subplot(212)
  objects = option_counter.keys()
  y_pos = np.arange(len(objects))
  performance = [100*option_counter_unweighted[k] for k in objects]
  plt.title("Raw unweighted Sample")
  plt.bar(y_pos, performance, align='center', alpha=0.5)
  plt.xticks(y_pos, objects)

  plt.savefig("affiliation.png") 
  plt.show()


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

 plot_pref_hist=False
 if plot_pref_hist:
  plt.clf()
  bins=[0,1.01,2.01,3.01,4.01,5.01] 

 avg_pref = {}
 #calculate average preference for each candidate
 for idx,candidate in enumerate(candidates[:]):
  have_preference_criteria = data['%s_pref_num'%candidate] != -1
  have_preference = data[have_preference_criteria]

  if plot_pref_hist:
   plt.subplot(2,1,idx+1)
   plt.title(cand_names[candidate])
   plt.xticks([0.5,1.5,2.5,3.5,4.5,5.5],['V. Dissatisfied','Dissatisfied','Neutral','Satisfied','V. Satisfied'])
   plt.hist(have_preference['%s_pref_num'%candidate],weights=have_preference['weights'],bins=bins,normed=True)

  avg_pref[candidate]= np.average(have_preference['%s_pref_num'%candidate],weights=have_preference['weights'])

 if plot_pref_hist:
  plt.savefig("preference_histogram22.png")
  plt.show()

 plot_average_preference = True
 if plot_average_preference:
  objects = avg_pref.keys()
  labels = [cand_names[k] for k in objects]
  y_pos = np.arange(len(objects))
  performance = [avg_pref[k] for k in objects]
  plt.bar(y_pos, performance, align='center', alpha=0.5)
  plt.xticks(y_pos, labels)
  plt.ylim((2,3))
  plt.ylabel('Average Satisfaction Level')
  plt.savefig("average_pref.png")
  plt.show()
 
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
