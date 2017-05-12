import csv
import re, os
import sys, code
import random
from collections import Counter
#from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn import metrics
from sklearn.metrics import zero_one_loss

from sklearn import svm
from sklearn import decomposition
from sklearn import grid_search
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from nltk.stem.porter import PorterStemmer
import numpy as np
import math
#from word_clusters import identifyCluster
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV, RFE

from textblob import TextBlob

image_feat_dir = 'data/image_feats/'
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def keyboard(banner=None):
    ''' Function that mimics the matlab keyboard command '''
    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back
    print "# Use quit() to exit :) Happy debugging!"
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        code.interact(banner=banner, local=namespace)
    except SystemExit:
        return

def main():
  image_feat_dir = '/Users/emjtang/Dropbox/_spr17/geweldig/data/image_feats/'
  filename = 'museum_data_short.csv'

  with open(filename) as listings_file:
    reader = csv.DictReader(listings_file)
    listings = list(reader);
    #id,image_url,principalOrFirstMaker,title,longTitle,width,height
    all_artists = list(set([row['principalOrFirstMaker'].lower() for row in listings]))

    for row in listings:
      image_out = image_feat_dir + row['id'] + '.npy'
      if os.path.isfile(image_out):
        visual_feat = np.load(image_feat_dir + row['id'] + '.npy')
      else:
        visual_feat = np.zeros(1000)
      row['visual_feature'] = visual_feat
    
    for row in listings:

      fv = np.array(row['visual_feature'])
      row['feature_vector'] = fv

    X = np.array([row['feature_vector'] for row in listings])
    y = np.array([row['label'] for row in listings])
    
    train_set_data, test_set_data, train_set_labels, test_set_labels = train_test_split(X,y,test_size = 0.2, random_state = 0)
    #train_set_data, devtest_set_data, train_set_labels, devtest_set_labels = train_test_split(X,y,test_size = 0.80, random_state = 0)
    #dev_set_data, test_set_data, dev_set_labels, test_set_labels = train_test_split(devtest_set_data, devtest_set_labels, test_size = 0.5, random_state =0)

    # print 'running svm'
    # classifier = svm.SVC(gamma=0.001, C=1., class_weight='balanced')

    print 'running svm'
    classifier = svm.SVC(kernel='linear',C=10., class_weight='auto', probability=True)
    classifier.fit(train_set_data, train_set_labels)
    print 'svm training score', classifier.score(train_set_data, train_set_labels)
    print 'svm test score', classifier.score(test_set_data, test_set_labels)

    pred = classifier.predict(test_set_data)
    probs = classifier.predict_log_proba(test_set_data)

    # print 'precision', metrics.precision_score(test_set_labels, pred)
    # print 'recall', metrics.recall_score(test_set_labels, pred)

    # print 'confusion matrix'
    # conf_matrix = confusion_matrix(test_set_labels, pred)
    # print conf_matrix
    # print 'plotting roc curve'
    # fpr, tpr, thresholds = metrics.roc_curve(test_set_labels, probs[:,1])
    # roc_auc = metrics.auc(fpr, tpr)
    # plt.title('Receiver Operating Characteristic for Price Prediction')
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.3f'% roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0,1],[0,1],'r--')
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()

    # print 'computing learning curves'
    # skf = StratifiedKFold(y, n_folds=3)

    # title = "Learning Curve for Predicting Prices (SVM Linear Kernel)"
    # plot_learning_curve(classifier, title, X, y, cv=skf)

  
    print 'done'
if __name__ == '__main__':
  main()
