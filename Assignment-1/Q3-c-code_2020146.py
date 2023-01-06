import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('C://Users//91991//Desktop//ML//Dry_Bean_Dataset.csv')
d=data.drop(['Class'],axis=1)#dropping the class column
#preprocessing the data
transformed_data=StandardScaler().fit_transform(d)
#implementing t-SNE
model=TSNE(n_components=2)#reducing the dimension to 2
#fitting the data
tsne=model.fit_transform(transformed_data)
#storing the data in a new dataframe
tsne=np.vstack((tsne.T,data['Class'])).T
#creating a new dataframe
tsne_data_frame=pd.DataFrame(data=tsne,columns=('Data_Dimension_1','Data_Dimension_2','Class'))
#plotting the t-SNE
sns.FacetGrid(tsne_data_frame,hue='Class').map(plt.scatter,'Data_Dimension_1','Data_Dimension_2').add_legend()
plt.show()