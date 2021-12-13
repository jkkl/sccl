"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

from numpy.core.fromnumeric import sort
from sklearn import cluster
import torch
import numpy as np
from utils.metric import Confusion
from sklearn.cluster import KMeans
from collections import Counter

def get_kmeans_centers(embedder, train_loader, num_classes, key=None, args=None):
    text_list = []
    info_list = []
    for i, batch in enumerate(train_loader):

        text, label = batch['text'], batch['label']
        if 'info' in batch:
            info = batch['info']
            info_list += info
        corpus_embeddings = embedder.encode(text)
        # print("corpus_embeddings", type(corpus_embeddings), corpus_embeddings.shape)
        if i == 0:
            all_labels = label
            all_embeddings = corpus_embeddings
        else:
            all_labels = torch.cat((all_labels, label), dim=0)
            all_embeddings = np.concatenate((all_embeddings, corpus_embeddings), axis=0)
        text_list += text

    # Perform kmean clustering
    confusion = Confusion(num_classes)
    clustering_model = KMeans(n_clusters=num_classes)
    clustering_model.fit(all_embeddings)
    cluster_assignment = clustering_model.labels_

    true_labels = all_labels
    pred_labels = torch.tensor(cluster_assignment)    
    print("all_embeddings:{}, true_labels:{}, pred_labels:{}".format(all_embeddings.shape, len(true_labels), len(pred_labels)))

    confusion.add(pred_labels, true_labels)
    confusion.optimal_assignment(num_classes)
    print("Iterations:{}, Clustering ACC:{:.3f}, centers:{}".format(clustering_model.n_iter_, confusion.acc(), clustering_model.cluster_centers_.shape))

    save_result(text_list, info_list, cluster_assignment, f'{args.cluster_result_path}/{key}_{args.num_classes}_-1_{clustering_model.inertia_}.txt')    

    
    return clustering_model.cluster_centers_


def save_result(text_list, info_list, cluster_assignment, output_file):
    fr = open(output_file, 'w')
    cluster_assignment_sorted = sort_cluster_assigment_by_count(cluster_assignment)
    cluster_id_text_dict = {}
    if info_list:
        text_info_list = ['\t'.join([text, info]) for text,info in zip(text_list, info_list)]
        for text, cluster_id in zip(text_info_list, cluster_assignment):
            if cluster_id not in cluster_id_text_dict:
                cluster_id_text_dict[cluster_id] = [text]
            else:
                cluster_id_text_dict[cluster_id].append(text)
    else:
        for text, cluster_id in zip(text_list, cluster_assignment):
            if cluster_id not in cluster_id_text_dict:
                cluster_id_text_dict[cluster_id] = [text]
            else:
                cluster_id_text_dict[cluster_id].append(text)

    for cluster_id_count_pair in cluster_assignment_sorted:
        cluster_id = cluster_id_count_pair[0]
        text_list = cluster_id_text_dict[cluster_id]
        for text in text_list:
            out_line = f'{cluster_id}\t{text}\n'
            fr.write(out_line)
    fr.close()


def sort_cluster_assigment_by_count(cluster_assignment):
    cluster_counter = Counter(cluster_assignment)
    cluster_assigment_sorted = sorted(cluster_counter.items(), key=lambda x:x[1], reverse=True)
    return cluster_assigment_sorted


