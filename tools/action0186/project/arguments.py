from pathlib import Path
def predecessor_arguments(i,temporary_directory):
 p=Path(temporary_directory)/'recursive.zip'
 with p.open('wb') as f:
  for j in range(2):
   with Path(i['scientific_part'+str(j)]).open('rb') as source:
    for b in iter(lambda:source.read(1<<20),b''):f.write(b)
 args={'branch':i['branch_catalog'],'admission':i['pair_admission'],'parent_archive':i['parent_archive'],'pair_archive':i['pair_archive'],'previous_catalog':i['catalog0140'],'recursive_admission':i['admission'],'predecessor_catalog':i['catalog'],'recursive_archive':p}
 for name,cat in [('node','0141'),('o2','0142'),('o3','0143'),('o4','0144'),('o5','0145'),('o6','0146'),('o7','0147')]:args={name+'_admission':i[name+'_admission'],name+'_archive':i[name+'_archive'],'previous_catalog':i['catalog'+cat],'previous_arguments':args}
 args={'additional_admission':i['additional_admission'],'additional_archive':i['additional_archive'],'previous_catalog':i['catalog0148'],'previous_arguments':args}
 return {'admission':i['g1_admission'],'archive':i['g1_archive'],'previous_catalog':i['catalog0149'],'previous_arguments':args}
