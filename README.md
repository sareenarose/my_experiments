# my_experiments
Manuscript Title: Identifying the leverage points using a blend of machine 
learning and statistical approaches


Authors: Sareena Rose, Dr. S.Nickolas, Dr. S. M. Sunoj & Dr. S. Sangeetha


Name of the program: Clustering using quantile regression and POOLED DISTANCES ALGORITHM (PDA)

PDA uses DetectMultVarOutliers (https://www.github.com/AntonSemechko/Multivariate-Outliers) with some modifications

Steps to run the Python code
1. Configure Training data file location in RO_VO_GLP_BLP.py file.
2. The algorithm implemented in the file RO_VO_GLP_BLP.py imports funtions from PDA.py and DTBR.py
3. RO_VO_GLP_BLP.py starts by loading the data. Then
	a. Use 'decision_tree_bagging_regressor_quantile_clusters' function implemented in DTBR.py file to produce clusters according to quantile values with respect to prediction.
	b. Then pass the records associated with those clusters to PDA.py to get group_of_inliers, group_of_ambigious_records, group_of_outliers.
After getting group_of_inliers, group_of_ambigious_records, group_of_outliers the file will generate RO, VO, GLP and BLP in the variables gr1_RO, gr1_VO, gr1_GLP and gr1_BLP respectivelly.


Pakages required for running the code are:
	a. pandas
	b. numpy
	c. sklearn
	d. scipy

detailed description of pakages are imcluded in requirements.txt


The code is also developed in Matlab 2018b and uses Statistical & Machine Learning ToolBox also the open source code in Python is also added

The above two programs is used to train the dataset and creates a framework for Regular Observations, Vertical Outliers, Good and Bad Leverage points.

Then the test data set is classified using the above framework using any simple distance metric

Use the files in this order 1) Clustering using quantile regression and 2) POOLED DISTANCES ALGORITHM (PDA)

The first phase returns some clusters and for each cluster PDA should be run separately
