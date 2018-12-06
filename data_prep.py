'''
Code built to prep the UNSW NB-15 Cyberattack Anomaly detectiond dataset.

Contains function that load the data for 


'''
import pandas as pd
from sklearn.model_selection import train_test_split

def load_agg_data(data_dir='./data/'):
	'''
    Load data found in full UNSW-NB15 .csv files.


	'''
	dfs = []
	for i in range(1,5):
	    path = data_dir+'UNSW-NB15_{}.csv'
	    dfs.append(pd.read_csv(path.format(i), header = None))
	all_data = pd.concat(dfs).reset_index(drop=True)
	all_data.columns = pd.read_csv('./data/UNSW-NB15_features.csv').Name.apply(lambda x: x.lower())


	## Column cleaning steps: some of the CSV's leave the point blank for zero values.
	## This results in Pandas loading in NaN values in columns where it otherwise expects numeric values. 
	# Fill all NaN attack categories w/ value: 'normal'
	all_data['attack_cat'] = all_data.attack_cat.fillna(value='normal').apply(lambda x: x.strip().lower())
	all_data['attack_cat'] = all_data.attack_cat.replace(value='backdoors', to_replace='backdoor')
	# Replace blank spaces with zero
	all_data['ct_ftp_cmd'] = all_data.ct_ftp_cmd.replace(to_replace=' ', value=0).astype(int)
	# Replace NaN with zero
	all_data['ct_flw_http_mthd'] = all_data.ct_flw_http_mthd.fillna(value=0)
	# Replace NaN with zero and all values > 0 with 1
	all_data['is_ftp_login'] = (all_data.is_ftp_login.fillna(value=0) >0).astype(int)

	## Reduce categorical features into smaller sets:
	## Ex: 135 unique values in `proto` become "tcp", "udp", "arp", "unas", and "other"
	transformations = {
	    'proto':['tcp', 'udp', 'arp', 'unas'],
	    'state':['fin', 'con', 'int'],
	    'service':['-', 'dns']
	}
	for col, keepers in transformations.items():
	    all_data[col] = all_data[col].apply(reduce_column, args=(keepers,))

	# Return with non-informative data eliminated
	drop_cols = ['srcip', 'sport', 'dstip', 'dsport','stcpb', 'dtcpb']
	return all_data.drop(columns=drop_cols)

def load_agg_Xy(path='./data/', sample_size=0.25, strat_cat='label'):
	'''
	Wrapper function for loading smaller subset of UNSW-NB15 dataset.
	---
	Inputs:

	'''
	df = load_agg_data(path=path)
	_, X, _, y = train_test_split(df.iloc[:,:-1],df[strat_cat],
		stratify=df[strat_cat], test_size=sample_size)
	return X, y



def load_csv_data(path, strategy='basic', to_exclude='object'):
	"""
	Wrapper function for loading data from a provided csv.
	Assumes the csv has a header row and an index column when
	creating the dataframe to return.
	"""

	data = pd.read_csv(path,header=0, index_col=0)

	features = data.select_dtypes(exclude='object').iloc[:,:-1]

	if strategy == 'basic':
		# Drop all columns of a certain type, in this case non-numeric.
		# Return features and class labels as separate objects
		return features, data.label

	elif strategy == 'anomaly':
		return features, data.label.apply(y_anomaly_format)

def y_anomaly_format(y):
    '''
    Accepts a target value of 0 or 1 and reformats to the appropriate class for sklearn's anomaly models.
    inlier: 0 -> 1
    outlier: 1 -> -1
    '''
    if y == 0:
        return 1
    elif y == 1:
        return -1
    else:
        print('Unexpected input:', y)
        raise ValueError ('Unexpected input')

def reduce_column(s, to_keep):
    '''
    Reduces the string values found in a column
    to the values provided in the list 'to_keep'.
    ---
    Input:
        s: string
        to_keep: list of strings
    Returns:
        string, s if s should be kept, else 'other'
    '''
    s = s.lower().strip()
    if s not in to_keep:
        return 'other'
    else:
        return s

attack_to_num = {
	'normal':0,
	'analysis':1,
	'backdoors':2,
	'dos':3,
	'exploits':4,
	'fuzzers':5,
	'generic':6,
	'reconnaissance':7,
	'shellcode':8,
	'worms':9

}