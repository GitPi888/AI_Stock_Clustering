import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition  import PCA


def tranform_data(): 
    file_path = "D:\Workspace\Data"
    list_df=[]
    data_frame=[]
    os.chdir(file_path)
    for files in os.listdir(file_path):
    # Check if the item is a file before attempting to read it
        if os.path.isfile(files):
            file_name=os.path.splitext(files)[0]
            df=pd.read_csv(files)
            if len(df.columns)==1:
                df.columns=[file_name]
        else:
            df=df.iloc[:,0]
            df.name=file_name
        data_frame.append(df)
    combine_df=pd.concat(data_frame,axis=1)
    means=round(np.mean(combine_df))
    combine_df.fillna(means,inplace=True)
    combine_df.to_csv("D:\Workspace\Data_final.csv",index=False)



combine_df=pd.read_csv("D:\Workspace\Data_final.csv")





# header=list(combine_df.columns)
# cvmax=np.std(combine_df[header[0]])
# c_max_label=header[0]
# for i in range(1,len(header)):
#     if cvmax< np.std(combine_df[header[i]]):
#         cvmax=np.std(combine_df[header[i]])
#         c_max_label=header[i]

# # print(cvmax)
# # print(c_max_label)


# def init_centroid():
#     pass

inerteria=[]
def elbow(data):
    for i in range(1,20):
        kmean=KMeans(n_clusters=i)
        kmean.fit(data)
        inerteria.append(kmean.inertia_)
    plt.scatter(range(1,20),inerteria)
    # plt.show()

elbow(combine_df)


df_scaler=StandardScaler().fit_transform(combine_df)

pca=PCA(n_components=2)
principal_components=pca.fit_transform(df_scaler)
df_pca = pd.DataFrame(data=principal_components, columns=['principal_component_1', 'principal_component_2'])

k=3
kmeans = KMeans(n_clusters=k)
kmeans.fit(df_pca)

df_pca['Cluser']=kmeans.labels_

plt.figure(figsize=(10, 7))
plt.scatter(df_pca['principal_component_1'], df_pca['principal_component_2'], c=df_pca['Cluser'], cmap='viridis', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA and K-Means Clustering of Companies')
plt.show()




# describe=combine_df.describe()
# print(describe)





# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(combine_df)

# k = 3
# kmeans = KMeans(n_clusters=k, random_state=0)
# kmeans.fit(df_scaled)

# # Gán nhãn cụm vào dataframe
# combine_df['Cluster'] = kmeans.labels_

# # In ra kết quả


# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# principal_components = pca.fit_transform(df_scaled)
# df_pca = pd.DataFrame(data=principal_components, columns=['principal_component_1', 'principal_component_2'])
# df_pca['Cluster'] = kmeans.labels_

# # Vẽ scatter plot
# plt.figure(figsize=(10, 8))
# sns.scatterplot(x='principal_component_1', y='principal_component_2', hue='Cluster', data=df_pca, palette='viridis', s=100)
# plt.title('K-means Clustering với 3 cụm')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend(title='Cluster')
# plt.show()