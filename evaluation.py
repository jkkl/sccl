"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import numpy as np
from utils.kmeans import save_result
from utils.metric import Confusion
from dataloader.dataloader import train_unshuffle_loader
from sklearn import cluster

def prepare_task_input(model, batch, args, is_contrastive=False):
    if is_contrastive:
        text, text1, text2, class_label = batch['text'], batch['text1'], batch['text2'], batch['label'].cuda()
        txts = [text, text1, text2]
        
        feat = []
        for text in txts:
            features = model.tokenizer.batch_encode_plus(text, max_length=args.max_length, return_tensors='pt', padding='longest', truncation=True)
            for k in features.keys():
                features[k] = features[k].cuda()
            feat.append(features)
        return feat, class_label.detach()
    else:
        text, class_label = batch['text'], batch['label'].cuda()
        features = model.tokenizer.batch_encode_plus(text, max_length=args.max_length, return_tensors='pt', padding='longest', truncation=True)
        for k in features.keys():
            features[k] = features[k].cuda()
        return features, class_label.detach()

def evaluate_embedding(model, args, step, intention_key=None, dataloader=None, loss=None):
    confusion, confusion_model = Confusion(args.num_classes), Confusion(args.num_classes)
    model.eval()
    if dataloader == None:
        dataloader = train_unshuffle_loader(args)
    print('---- {} evaluation batches ----'.format(len(dataloader)))
    text_list = []
    info_list = []
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            text, label = batch['text'], batch['label'] 
            if 'info' in batch:
                info_list += batch['info']
            features, _ = prepare_task_input(model, batch, args, is_contrastive=False)
            embeddings = model.get_embeddings(features)
            model_prob = model.get_cluster_prob(embeddings)

            if i == 0:
                all_labels = label
                all_embeddings = embeddings.detach()
                all_prob = model_prob
            else:
                all_labels = torch.cat((all_labels, label), dim=0)
                all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                all_prob = torch.cat((all_prob, model_prob), dim=0)
            text_list += text

    all_pred = all_prob.max(1)[1]
    confusion_model.add(all_pred, all_labels)
    confusion_model.optimal_assignment(args.num_classes)
    acc_model = confusion_model.acc()

    kmeans = cluster.KMeans(n_clusters=args.num_classes, random_state=args.seed)
    embeddings = all_embeddings.cpu().numpy()
    kmeans.fit(embeddings)
    pred_labels = torch.tensor(kmeans.labels_.astype(np.int))
    # clustering accuracy 
    confusion.add(pred_labels, all_labels)
    confusion.optimal_assignment(args.num_classes)
    acc = confusion.acc()
    
    ressave = {"acc":acc, "acc_model":acc_model}
    for key, val in ressave.items():
        args.tensorboard.add_scalar('Test/{}'.format(key), val, step)
    
    print('[Representation] Clustering scores:',confusion.clusterscores()) 
    print('[Representation] ACC: {:.3f}'.format(acc)) 
    print('[Model] Clustering scores:',confusion_model.clusterscores()) 
    print('[Model] ACC: {:.3f}'.format(acc_model))
    save_result(text_list, info_list, kmeans.labels_, f'{args.cluster_result_path}/{intention_key}_{step}_{kmeans.inertia_}.txt')
    return acc, acc_model


def evaluate_embedding_simcse(model, args, step, intention_key=None, dataloader=None, loss=None):
    confusion, confusion_model = Confusion(args.num_classes), Confusion(args.num_classes)
    model.eval()
    if dataloader == None:
        dataloader = train_unshuffle_loader(args)
    print('---- {} evaluation batches ----'.format(len(dataloader)))
    text_list = []
    info_list = []
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            text, label = batch['text'], batch['label'] 
            features, _ = prepare_task_input(model, batch, args, is_contrastive=False)
            embeddings = model.get_embeddings(features)
            model_prob = model.get_cluster_prob(embeddings)

            if i == 0:
                all_labels = label
                all_embeddings = embeddings.detach()
                all_prob = model_prob
            else:
                all_labels = torch.cat((all_labels, label), dim=0)
                all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                all_prob = torch.cat((all_prob, model_prob), dim=0)
            text_list += text

    all_pred = all_prob.max(1)[1]
    confusion_model.add(all_pred, all_labels)
    confusion_model.optimal_assignment(args.num_classes)
    acc_model = confusion_model.acc()

    kmeans = cluster.KMeans(n_clusters=args.num_classes, random_state=args.seed)
    embeddings = all_embeddings.cpu().numpy()
    kmeans.fit(embeddings)
    pred_labels = torch.tensor(kmeans.labels_.astype(np.int))
    # clustering accuracy 
    confusion.add(pred_labels, all_labels)
    confusion.optimal_assignment(args.num_classes)
    acc = confusion.acc()
    
    ressave = {"acc":acc, "acc_model":acc_model}
    for key, val in ressave.items():
        args.tensorboard.add_scalar('Test/{}'.format(key), val, step)
    
    print('[Representation] Clustering scores:',confusion.clusterscores()) 
    print('[Representation] ACC: {:.3f}'.format(acc)) 
    print('[Model] Clustering scores:',confusion_model.clusterscores()) 
    print('[Model] ACC: {:.3f}'.format(acc_model))
    save_result(text_list, info_list, kmeans.labels_, f'saved_result/intention_multi/{intention_key}_{step}_{kmeans.inertia_}.txt')
    return acc, acc_model



# def save_result(text_list, cluster_assignment, output_file):
#     fr = open(output_file, 'w')
#     for text, cluster_id in zip(text_list, cluster_assignment):
#         out_line = f'{text}\t{cluster_id}\n'
#         fr.write(out_line)
#     fr.close()


