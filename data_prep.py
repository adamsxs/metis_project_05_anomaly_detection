'''
Code built to prep the UNSW NB-15 Cyberattack Anomaly detectiond dataset.

Contains function that load the data for 


'''
import pandas as pd

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