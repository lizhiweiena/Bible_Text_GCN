# -*- coding: utf-8 -*-
"""
evaluate the results and misclassified labels
"""

import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
import os
from text_GCN import gcn
from sklearn.metrics import confusion_matrix
import pandas as pd


def load_pickle(filename):
    completeName = os.path.join("./data/",filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


def save_as_pickle(filename, data):
    completeName = os.path.join("./data/",filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)


def evaluate(output, labels_e):
    _, labels = output.max(1); labels = labels.numpy()
    return sum([(e-1) for e in labels_e] == labels)/len(labels)


def misclassified(df_data, pred_labels_test_idx, labels_not_selected, test_idxs, book_dict):
    for actual, pred, test_idx in zip([(e-1) for e in labels_not_selected], \
                             list(pred_labels_test_idx.max(1)[1].numpy()),\
                             test_idxs):
        if actual != pred:
            print("Book: %s\nChapter: %s" % (book_dict[actual+1], df_data.loc[test_idx]["c"]))
            print("Predicted as %s" % book_dict[pred+1]); print("\n")


if __name__=="__main__":
    base_path = "./data/"
    # Loads bible data
    df = pd.read_csv(os.path.join(base_path,"t_bbe.csv"))
    df.drop(["id", "v"], axis=1, inplace=True)
    df = df[["t","c","b"]]
    # one chapter per document, labelled by book
    df_data = pd.DataFrame(columns=["c", "b"])
    for book in df["b"].unique():
        dum = pd.DataFrame(columns=["c", "b"])
        dum["c"] = df[df["b"] == book].groupby("c").apply(lambda x: (" ".join(x["t"])).lower())
        dum["b"] = book
        df_data = pd.concat([df_data,dum], ignore_index=True)
    del df
    book_dict = pd.read_csv(os.path.join(base_path, "key_english.csv"))
    book_dict = {book.lower():number for book, number in zip(book_dict["field.1"], book_dict["field"])}
    book_dict = {v:k for k,v in zip(book_dict.keys(),book_dict.values())}
    # Loads graph data
    G = load_pickle("text_graph.pkl")
    A = nx.to_numpy_matrix(G, weight="weight"); A = A + np.eye(G.number_of_nodes())
    degrees = []
    for d in G.degree(weight=None):
        if d == 0:
            degrees.append(0)
        else:
            degrees.append(d[1]**(-0.5))
    degrees = np.diag(degrees)
    X = np.eye(G.number_of_nodes()) # Features are just identity matrix
    A_hat = degrees@A@degrees
    f = X  # (n X n) X (n X n) x (n X n) X (n X n) input of net
    f = torch.from_numpy(f).float()
    
    # Loads labels
    test_idxs = load_pickle("test_idxs.pkl")
    selected = load_pickle("selected.pkl")
    labels_selected = load_pickle("labels_selected.pkl")
    labels_not_selected = load_pickle("labels_not_selected.pkl")
    
    # Loads best model ###
    checkpoint = torch.load(os.path.join(base_path,"model_best.pth.tar"))
    net = gcn(X.shape[1], A_hat)
    net.load_state_dict(checkpoint['state_dict'])
    
    # labels distribution
    fig = plt.figure(figsize=(15,17))
    ax = fig.add_subplot(111)
    ax.hist([(e-1) for e in labels_not_selected] + [(e-1) for e in labels_selected], bins=66)
    ax.set_title("Class label distribution for data set", fontsize=20)
    ax.set_xlabel("Class label", fontsize=17)
    ax.set_ylabel("Counts", fontsize=17)
    [x.set_fontsize(15) for x in ax.get_xticklabels()]; [x.set_fontsize(15) for x in ax.get_yticklabels()]
    plt.savefig(os.path.join("./data/", "data_idxs_dist.png"))
    
    fig = plt.figure(figsize=(15,17))
    ax = fig.add_subplot(111)
    ax.hist([(e-1) for e in labels_not_selected], bins=66)
    ax.set_title("Class label distribution for test set", fontsize=20)
    ax.set_xlabel("Class label", fontsize=17)
    ax.set_ylabel("Counts", fontsize=17)
    [x.set_fontsize(15) for x in ax.get_xticklabels()]; [x.set_fontsize(15) for x in ax.get_yticklabels()]
    plt.savefig(os.path.join("./data/", "test_true_idxs_dist.png"))
    # Inference
    net.eval()
    pred_labels = net(f)
    c_m = confusion_matrix([(e-1) for e in labels_not_selected], list(pred_labels[test_idxs].max(1)[1].numpy()))
    fig = plt.figure(figsize=(25,25))
    ax = fig.add_subplot(111)
    sb.heatmap(c_m, annot=False)
    ax.set_title("Confusion Matrix", fontsize=20)
    ax.set_xlabel("Actual class", fontsize=17)
    ax.set_ylabel("Predicted", fontsize=17)
    plt.savefig(os.path.join("./data/", "confusion_matrix.png"))
    # Prints misclassified labels
    misclassified(df_data, pred_labels[test_idxs], labels_not_selected, test_idxs, book_dict)