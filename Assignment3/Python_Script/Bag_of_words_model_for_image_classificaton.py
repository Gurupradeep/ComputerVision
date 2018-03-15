
#
import cv2
import numpy as np
import scipy.io as scio

np.random.seed(0)





import matplotlib.pyplot as plt
from scipy.misc import imsave
from scipy import ndimage, misc
from numpy import unravel_index
from operator import sub
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


import scipy.io as scio
import numpy as np    
import os
import matplotlib.pyplot as plt
import math
import re
from scipy.misc import imsave
from scipy import ndimage, misc
from numpy import unravel_index
from operator import sub
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


# In[7]:


def atoi(text) : 
    return int(text) if text.isdigit() else text


# In[8]:


def natural_keys(text) :
    return [atoi(c) for c in re.split('(\d+)', text)]


# In[1284]:


descriptor_list = []
labels = []
label_count  = 0


# In[1285]:


root_path = ""
filenames = []
for root, dirnames, filenames in os.walk("Selected_Categories/airplanes/"):
    filenames.sort(key = natural_keys)
    rootpath = root


# In[1286]:


for filename in filenames :
    gray = cv2.imread("/home/guru/Desktop/ComputerVision/Assignment3/Selected_Categories/airplanes/" + filename, 0)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints,descriptors = sift.detectAndCompute(gray,None);
    descriptor_list.append(descriptors)
    labels.append(label_count)


# In[1287]:


label_count = label_count + 1


# In[1288]:


len(descriptor_list)


# In[1289]:


root_path = ""
filenames = []
for root, dirnames, filenames in os.walk("Selected_Categories/kangaroo/"):
    filenames.sort(key = natural_keys)
    rootpath = root


# In[1290]:


for filename in filenames :
    gray = cv2.imread("/home/guru/Desktop/ComputerVision/Assignment3/Selected_Categories/kangaroo/" + filename, 0)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints,descriptors = sift.detectAndCompute(gray,None);
    descriptor_list.append(descriptors)
    labels.append(label_count)


# In[1291]:


len(descriptor_list)


# In[1292]:


label_count = label_count + 1


# In[1293]:


root_path = ""
filenames = []
for root, dirnames, filenames in os.walk("Selected_Categories/Motorbikes/"):
    filenames.sort(key = natural_keys)
    rootpath = root


# In[1294]:


for filename in filenames :
    gray = cv2.imread("/home/guru/Desktop/ComputerVision/Assignment3/Selected_Categories/Motorbikes/" + filename, 0)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints,descriptors = sift.detectAndCompute(gray,None);
    descriptor_list.append(descriptors)
    labels.append(label_count)


# In[1295]:


len(descriptor_list)


# In[1296]:


len(labels)


# In[1297]:


np.unique(labels)


# In[1298]:


train_indices = np.random.choice(10,5,replace= False)


# In[1299]:


train_indices


# In[1300]:


image_count = len(descriptor_list)


# In[1301]:


image_count


# In[1302]:


def train_and_test(split = 0.75) :
    train_descriptors = []
    test_descriptors = []
    train_labels = []
    test_labels = []
    train_indices = np.random.choice(image_count, int(split * image_count), replace = False)
    for i in train_indices :
        train_descriptors.append(descriptor_list[i])
        train_labels.append(labels[i])
    
    test_indices = [x for x in range(image_count) if x not in train_indices]
    for i in test_indices :
        test_descriptors.append(descriptor_list[i])
        test_labels.append(labels[i])
    
    return (train_descriptors, train_labels, test_descriptors, test_labels)


# In[1303]:


train_descriptors, train_labels, test_descriptors, test_labels = train_and_test(0.5)


# In[1304]:


train_count = len(train_descriptors)


# In[1305]:


len(train_labels)


# In[1306]:


test_count = len(test_descriptors)


# In[1307]:


len(test_labels)


# In[1308]:


def preprocess_for_clustering(train_descriptors):
    processed_descriptors = np.array(train_descriptors[0])
    for remaining in train_descriptors[1:] :
        processed_descriptors = np.vstack((processed_descriptors, remaining))
    return processed_descriptors


# In[1309]:


processed_descriptors = preprocess_for_clustering(train_descriptors)


# In[1310]:


processed_descriptors.shape


# In[1311]:


np.save("descriptors.npy", processed_descriptors)


# In[1312]:


random_descriptor_indices = np.random.choice(len(processed_descriptors),20000, replace = False)


# In[1313]:


sample_descriptors = []
for i in random_descriptor_indices :
    sample_descriptors.append(processed_descriptors[i])
    


# In[1314]:


sample_descriptors = np.array(sample_descriptors)


# In[1315]:


sample_descriptors.shape


# In[1316]:


def silhouette_score_for_cluster(X) :
    for i in range(5,30) :
        clusterer = KMeans(n_clusters = i, random_state = 6, n_jobs = -1)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", i ,"The average silhouette_score is :", silhouette_avg)


# In[1317]:


silhouette_score_for_cluster(sample_descriptors)


# In[722]:


best_choices = [7,12,19,27]


# In[723]:


for n_clusters in best_choices:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(sample_descriptors) + (n_clusters + 1) * 10])
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=6, n_jobs = -1)
    cluster_labels = clusterer.fit_predict(sample_descriptors)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(sample_descriptors, cluster_labels, sample_size = 100)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(sample_descriptors, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(processed_descriptors[:, 0], processed_descriptors[:, 1],
                marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()
    fig.savefig("S_score" + str(n_clusters)+".png")


# In[1223]:


no_of_clusters = 7


# In[1224]:


kmeans = KMeans(n_clusters = no_of_clusters, random_state=6, n_jobs = -1)
kmeans_ret = kmeans.fit_predict(processed_descriptors)


# In[1225]:


np.unique(kmeans_ret)


# In[1226]:


train_histogram = np.array([np.zeros(no_of_clusters) for i in range(train_count)])


# In[1227]:


old_count  = 0
for i in range(train_count) :
    l = len(train_descriptors[i])
    for j in range(l) :
        index = kmeans_ret[old_count + j]
        train_histogram[i][index] += 1
    old_count += l


# In[1228]:


train_histogram.shape


# In[1229]:


train_histogram[0]


# In[1230]:


def plot_histogram(no_of_clusters, histogram, name) :
    x_scalar = np.arange(no_of_clusters)
    y_scalar = np.array([abs(np.sum(histogram[:,h], dtype=np.int32)) for h in range(no_of_clusters)])
    plt.bar(x_scalar, y_scalar)
    fig = plt.gcf()
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar)
    plt.show()
    fig.savefig(name)


# In[1231]:


plot_histogram(no_of_clusters, train_histogram,"Vocabulary_before_normalisation.png")


# In[1232]:


def Scaling_histogram(histogram) :
    for i in range(len(histogram)) :
        sum_value = 0
        for j in range(20) :
            sum_value += histogram[i][j]
        for j in range(20) :
            histogram[i][j] = histogram[i][j] / (sum_value)
    return histogram


# In[1233]:


def Normalised_histogram(histogram) :
    scale = StandardScaler().fit(histogram)
    return scale.transform(histogram)


# In[1234]:


len(train_histogram)


# In[1235]:


train_histogram.shape


# In[1236]:


normalised_train_histogram = Normalised_histogram(train_histogram)


# In[1237]:


plot_histogram(no_of_clusters, normalised_train_histogram, "Vocabulary_after_normalisation.png")


# In[1238]:


normalised_train_histogram[0]


# In[1239]:


from sklearn.naive_bayes import GaussianNB


# In[1240]:


clf = GaussianNB()


# In[1241]:


clf.fit(normalised_train_histogram, train_labels)


# In[1242]:


train_labels[69]


# In[1243]:


normalised_train_histogram[6].shape


# In[1244]:


ans = clf.predict(normalised_train_histogram[69].reshape((1,no_of_clusters)))


# In[1245]:


ans


# In[1246]:


test_histogram = np.array([np.zeros(no_of_clusters) for i in range(test_count)])


# In[1247]:


len(test_descriptors)


# In[1248]:


kmeans.predict(test_descriptors[0]).shape


# In[1249]:


for i in range(test_count) :
    predictions = kmeans.predict(test_descriptors[i])
    for j in range(len(predictions)) :
        test_histogram[i][predictions[j]] += 1


# In[1250]:


Normalised_test_histogram = Normalised_histogram(test_histogram)


# In[1251]:


predictions = clf.score(Normalised_test_histogram, test_labels)


# In[1252]:


predictions


# In[1253]:


c1 = 0
c2 = 0
c3 = 0
for i in range(len(train_labels)) :
    if(train_labels[i] == 0) :
        c1 += 1
    elif(train_labels[i] == 1) :
        c2 += 1
    else :
        c3 += 1


# In[1254]:


print(c1)
print(c2)
print(c3)


# In[1255]:


c1 = 0
c2 = 0
c3 = 0
for i in range(len(test_labels)) :
    if(test_labels[i] == 0) :
        c1 += 1
    elif(test_labels[i] == 1) :
        c2 += 1
    else :
        c3 += 1


# In[1256]:


print(c1)
print(c2)
print(c3)


# In[1257]:


from sklearn.metrics import confusion_matrix
import itertools


# In[1258]:


predictions = clf.predict(Normalised_test_histogram)


# In[1259]:


predictions


# In[1260]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,name = "Cf.png",
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    fig = plt.gcf()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(name)
    plt.show()
    


# In[1261]:


cnf_matrix = confusion_matrix(test_labels, predictions)


# In[1262]:


cnf_matrix


# In[1263]:


class_names = ['Aeroplanes', 'Faces', 'Motor_bikes']


# In[1264]:


plot_confusion_matrix(cnf_matrix, classes=class_names,normalize= False,
                      name = "NB_Confusion_matrix_without_normalisation.png",
                      title='Confusion matrix without Normalisation')


# In[1265]:


plot_confusion_matrix(cnf_matrix, classes=class_names,normalize= True,
                      name = "NB_Confusion_matrix_with_normalisation.png",
                      title='Confusion matrix with Normalisation')


# In[1266]:


from sklearn import tree


# In[1267]:


clf = tree.DecisionTreeClassifier()


# In[1268]:


clf = clf.fit(normalised_train_histogram, train_labels)


# In[1269]:


accuracy = clf.score(Normalised_test_histogram, test_labels)


# In[1270]:


accuracy


# In[1271]:


Normalised_test_histogram.shape


# In[1272]:


predictions = clf.predict(Normalised_test_histogram)


# In[1273]:


cnf_matrix = confusion_matrix(test_labels, predictions)


# In[1274]:


plot_confusion_matrix(cnf_matrix, classes=class_names,normalize= False,
                      name = "Tree_Confusion_matrix_without_normalisation.png",
                      title='Confusion matrix without Normalisation')


# In[1275]:


plot_confusion_matrix(cnf_matrix, classes=class_names,normalize= True,
                      name = "Tree_Confusion_matrix_with_normalisation.png",
                      title='Confusion matrix with Normalisation')


# In[1276]:


from sklearn import svm


# In[1277]:


clf = svm.SVC()


# In[1278]:


clf.fit(normalised_train_histogram, train_labels)


# In[1279]:


clf.score(Normalised_test_histogram, test_labels)


# In[1280]:


predictions = clf.predict(Normalised_test_histogram)


# In[1281]:


cnf_matrix = confusion_matrix(test_labels, predictions)


# In[1282]:


plot_confusion_matrix(cnf_matrix, classes=class_names,normalize= False,
                      name = "SVM_Confusion_matrix_without_normalisation.png",
                      title='Confusion matrix without Normalisation')


# In[1283]:


plot_confusion_matrix(cnf_matrix, classes=class_names,normalize= True,
                      name = "SVM_Confusion_matrix_with_normalisation.png",
                      title='Confusion matrix with Normalisation')

