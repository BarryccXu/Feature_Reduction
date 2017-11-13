import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix, f1_score

#%% loading data and preprocessing
breast_data = pd.read_csv('data.csv', header = 0)
breast_data = breast_data.drop('Unnamed: 32', 1)
## converting categorical labels to integer
label_dict, label = np.unique(breast_data.iloc[:,1], return_inverse=True)
breast_data.iloc[:,1] = label
# normalizing features 
breast_data_norm = breast_data
for i in range(2,breast_data.shape[1]):
    breast_data_norm.iloc[:,i] = (breast_data.iloc[:,i] - np.mean(breast_data.iloc[:,i])) \
                        /np.std(breast_data.iloc[:,i])
                        

## generate training and test data
np.random.seed(1258)
permut = np.random.permutation(len(breast_data_norm))
train_len = np.int(len(breast_data_norm)*0.8)
train = breast_data_norm.iloc[permut[:train_len],1:]
test = breast_data_norm.iloc[permut[train_len:],1:]
##　converting dataframe to matrix
train = pd.DataFrame.as_matrix(train)
test = pd.DataFrame.as_matrix(test)
X_train = train[:,1:]
y_train = train[:,0]
X_test = test[:,1:]
y_test = test[:,0]
X_all = np.asmatrix(breast_data_norm.iloc[:,2:])
y_all = breast_data_norm.iloc[:,1]
#%%　logistic regression
def score(label, pred):
    delta = np.equal(label, pred)
    score = sum(delta)/len(label)
    return score
    
clf_LR_l2 = LogisticRegression(C=1, penalty='l2')
clf_LR_l2.fit(X_train, y_train)
# scores = cross_val_score(clf_LR_l2, X_train, y_train, scoring='roc_auc', n_jobs = -1, cv = 5)
pred = clf_LR_l2.predict(X_test)
score(y_test, pred)

coef = clf_LR_l2.coef_
index = np.asarray(coef.argsort()[-3:][::-1])
features_30 = breast_data.columns.values[2:]
#top5 negitive
top5_negitive = features_30[index[:,:5]]
top5_postive = features_30[index[:,-5:]]
# make a plot
f = plt.figure(num='Logistic Regression',figsize=(16,16))
confusion = confusion_matrix(y_test, pred) 
Accuracy = score(y_test, pred)
title = 'Logistic Regression' + '\n'+ 'Accuracy:'+ np.str(round(Accuracy,3))
plt.subplot(2,2,1)
plt.title(title)
plt.imshow(confusion, interpolation = 'None')
plt.colorbar()
plt.xticks([0,1], ['B','M'])
plt.yticks([0,1], ['B','M'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
f.savefig("Logistic Regression.pdf", bbox_inches='tight')  

tmp = np.equal(y_test, pred)
wrong_LR = np.where(tmp == False)


#%% 30 features
n_features_30 = 30
clf_LR_l2 = LogisticRegression(C=1, penalty='l2')
pred_i = []; 
score_i = [];
X_train_i = np.zeros((len(X_train),2))
X_test_i = np.zeros((len(X_test),2))
for i in range(n_features_30):
    X_train_i[:,0] = X_train[:,i]
    X_test_i[:,0] = X_test[:,i]
    clf_LR_l2.fit(X_train_i, y_train)
    pred_tmp = clf_LR_l2.predict(X_test_i)
    score_tmp = score(y_test, pred_tmp)
    pred_i.append(pred_tmp); score_i.append(score_tmp)

# make a plot
features_30 = breast_data.columns.values[2:]
f = plt.figure(num='confusion_mat_30',figsize=(30,30))
for i in range(n_features_30):
    confusion = confusion_matrix(y_test, pred_i[i]) 
    plt.subplot(5,8,i+1)
    title = np.str(features_30[i]) + '\n'+ 'Accuracy:'+ np.str(round(score_i[i],4))
    plt.title(title)
    plt.imshow(confusion, interpolation = 'None')
    plt.colorbar()
    plt.xticks([0,1], ['B','M'])
    plt.yticks([0,1], ['B','M'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
f.savefig("confusion_mat_30.pdf", bbox_inches='tight')  
#%% 10 kinds of features analysis

n_features_10 = 10
clf_LR_l2 = LogisticRegression(C=1, penalty='l2')
pred_i = []; 
score_i = [];
for i in range(n_features_10):
    col_i = [i,i+10,i+20]
    X_train_i = X_train[:,col_i]
    X_test_i = X_test[:,col_i]
    clf_LR_l2.fit(X_train_i, y_train)
    pred_tmp = clf_LR_l2.predict(X_test_i)
    score_tmp = score(y_test, pred_tmp)
    pred_i.append(pred_tmp); score_i.append(score_tmp)
# make a plot
 
features_10 = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', \
            'concavity', 'concave points', 'symmetry', 'fractal_dimension']
f = plt.figure(num='confusion_mat_10',figsize=(16,16))
for i in range(n_features_10):
    confusion = confusion_matrix(y_test, pred_i[i]) 
    plt.subplot(3,5,i+1)
    title = np.str(features_10[i]) + '\n'+ 'Accuracy:'+ np.str(round(score_i[i],4))
    plt.title(title)
    plt.imshow(confusion, interpolation = 'None')
    plt.colorbar()
    plt.xticks([0,1], ['B','M'])
    plt.yticks([0,1], ['B','M'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
f.savefig("confusion_mat_10.pdf", bbox_inches='tight')    
    
#%% 3 kinds of features analysis    
n_features_3 = 3
clf_LR_l2 = LogisticRegression(C=1, penalty='l2')
pred_i = []; 
score_i = [];
for i in range(n_features_3):
    col_i = np.arange(i,i+10)
    X_train_i = X_train[:,col_i]
    X_test_i = X_test[:,col_i]
    clf_LR_l2.fit(X_train_i, y_train)
    pred_tmp = clf_LR_l2.predict(X_test_i)
    score_tmp = f1_score(y_test, pred_tmp)
    pred_i.append(pred_tmp); score_i.append(score_tmp)
# make a plot    
features_3= ['mean', 'standard error', 'worst']
f = plt.figure(num='confusion_mat_3',figsize=(16,16))
for i in range(n_features_3):
    confusion = confusion_matrix(y_test, pred_i[i])    
    plt.subplot(2,3,i+1)
    title = np.str(features_3[i]) + '\n'+ 'Accuracy:'+ np.str(round(score_i[i],4))
    plt.title(title)
    plt.imshow(confusion, interpolation = 'None')
    plt.colorbar()
    plt.xticks([0,1], ['B','M'])
    plt.yticks([0,1], ['B','M'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
f.savefig("confusion_mat_3.pdf", bbox_inches='tight')    

#%% PCA 67 2 4 40, A:0.9469
f = plt.figure(num='confusion_mat_PCA',figsize=(16,16))
confusion = [[67,2], [4,40]]
Accuracy = 0.9469
title = 'PCA_2' + '\n'+ 'Accuracy:'+ np.str(0.9469)
plt.subplot(2,3,1)
plt.title(title)
plt.imshow(confusion, interpolation = 'None')
plt.colorbar()
plt.xticks([0,1], ['B','M'])
plt.yticks([0,1], ['B','M'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
f.savefig("confusion_mat_PCA.pdf", bbox_inches='tight')  



#%% bottleneck
n_feature = 30;
bottleneckLayer_size = 5
hiddenLayer_size = 15
tf.reset_default_graph()
input_data = tf.placeholder(tf.float32, shape=[None, n_feature], name='x')

hidden_layer_1 = tf.get_variable('hidden_layer_1', shape=[n_feature,hiddenLayer_size])
hidden_bias_1= tf.get_variable('hidden_bias_1', shape=[hiddenLayer_size])

bottleneckLayer = tf.get_variable('bottleneckLayer', shape=[hiddenLayer_size,bottleneckLayer_size])
bottleneckLayer_bias= tf.get_variable('bottleneckLayer_bias', shape=[bottleneckLayer_size])

hidden_layer_3 = tf.get_variable('hidden_layer_3', shape=[bottleneckLayer_size,hiddenLayer_size])
hidden_bias_3= tf.get_variable('hidden_bias_3', shape=[hiddenLayer_size])

out_layer = tf.get_variable('out_layer', shape = [hiddenLayer_size,n_feature])
out_bias = tf.get_variable('out_bias', shape = [n_feature])

h1 = tf.nn.tanh(tf.matmul(input_data, hidden_layer_1) + hidden_bias_1)
h2 = tf.nn.sigmoid(tf.matmul(h1, bottleneckLayer) + bottleneckLayer_bias)
h3 = tf.nn.tanh(tf.matmul(h2, hidden_layer_3) + hidden_bias_3)
out = tf.matmul(h3, out_layer) + out_bias

loss = tf.nn.l2_loss(out - input_data)/tf.to_float(tf.shape(input_data)[0])
optimizer = tf.train.AdamOptimizer(0.001)
train_op = optimizer.minimize(loss)
session = tf.Session()
session.run(tf.global_variables_initializer())

costs = []
iteration = 20000
for i in range(iteration):
    #output, _ = session.run([out, train_op], 
    #                   {pixels: train})
    c, _ = session.run([loss, train_op], 
                      {input_data: X_all})
    costs.append(c)
    if (0 == i % 10):
        print(i,'/',iteration)

f = plt.figure(num='Loss',figsize=(16,16))
plt.subplot(2,3,1)
plt.title('Loss of the BottleNeck Neural Network')
plt.plot(np.log10(costs))
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.show()
f.savefig("Loss.pdf", bbox_inches='tight')  


#%% autoencoer for classification

X_encoder = session.run(h2, {input_data: X_all})
#X_ = X_encoder
#for i in range(2,len(X_encoder[0,:])):
#    X_[:,i] = (X_encoder[:,i] - np.mean(X_encoder[:,i])) \
#                        /np.std(X_encoder[:,i])
np.random.seed(1258)
permut = np.random.permutation(len(breast_data_norm))
train_len = np.int(len(X_encoder)*0.8)
X_train_ = X_encoder[permut[:train_len],:]
X_test_ = X_encoder[permut[train_len:],:]
y_train_ = y_all[permut[:train_len]]
y_test_ = y_all[permut[train_len:]]
X_train_ = np.asmatrix(X_train_)
X_test_ = np.asmatrix(X_test_)


clf_LR_l2 = LogisticRegression(C=1, penalty='l2')
clf_LR_l2.fit(X_train_, y_train_)
#scores = cross_val_score(clf_LR_l2, X_train, y_train, scoring='roc_auc', n_jobs = -1, cv = 5)
pred = clf_LR_l2.predict(X_test_)
score(y_test_, pred)

f = plt.figure(num='confusion_mat_NN',figsize=(16,16))
confusion = confusion_matrix(y_test_, pred) 
Accuracy = 0.9469
title = 'Bottlneck NN' + '\n'+ 'Accuracy:'+ np.str(0.9469)
plt.subplot(2,2,1)
plt.title(title)
plt.imshow(confusion, interpolation = 'None')
plt.colorbar()
plt.xticks([0,1], ['B','M'])
plt.yticks([0,1], ['B','M'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
f.savefig("confusion_mat_NN.pdf", bbox_inches='tight')  
tmp = np.equal(y_test_, pred)
wrong_NN = np.where(tmp == False)
































    
    
    
