import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('C://Users//91991//Desktop//ML//Dry_Bean_Dataset.csv')

#performing EDA analysis on the data
corr=data.corr()
sns.heatmap(corr,annot=True)
plt.show()
#plotting the histogram
data.hist()
plt.show()

#plotting the scatter plot
data.plot(kind='scatter',x='Perimeter',y='Class')
plt.show()
