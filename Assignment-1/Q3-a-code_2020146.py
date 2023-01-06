import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv('C://Users//91991//Desktop//ML//Dry_Bean_Dataset.csv')#set the location of the data
#plot pie chart for class
data['Class'].value_counts().plot(kind='pie',autopct='%1.1f%%')
plt.show()