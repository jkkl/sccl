"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

from os import write
import sys
sys.path.append( './' )

import torch
import argparse
from sentence_transformers import SentenceTransformer
from models.Transformers import SCCLBert
from learners.cluster import ClusterLearner
from dataloader.dataloader import augment_loader_onlin_log
from training import training
from utils.kmeans import get_kmeans_centers
from utils.logger import setup_path
from utils.randomness import set_global_random_seed
from torch.utils.tensorboard import SummaryWriter
model_writer = SummaryWriter('./model_log/')

MODEL_CLASS = {
    "distil": 'distilbert-base-nli-stsb-mean-tokens', 
    "robertabase": 'roberta-base-nli-stsb-mean-tokens',
    "robertalarge": 'roberta-large-nli-stsb-mean-tokens',
    "msmarco": 'distilroberta-base-msmarco-v2',
    "xlm": "xlm-r-distilroberta-base-paraphrase-v1",
    "bertlarge": 'bert-large-nli-stsb-mean-tokens',
    "bertbase": 'bert-base-nli-stsb-mean-tokens',
    # "chinese": 'distiluse-base-multilingual-cased-v1',
    "chinese": 'saved_models/paraphrase-xlm-r-multilingual-v1'
}


def run(args):
    resPath, tensorboard = setup_path(args)
    args.resPath, args.tensorboard = resPath, tensorboard
    set_global_random_seed(args.seed)

    # dataset loader
    train_loader = augment_loader_onlin_log(args)
    key = 'island'
    sbert = SentenceTransformer(MODEL_CLASS[args.bert])
    for cluster_num in [100, 150, 200, 250, 300, 350, 400, 450, 500, 600,700,1000]:
        args.num_classes = cluster_num
        # model
        torch.cuda.set_device(args.gpuid[0])
        cluster_centers = get_kmeans_centers(sbert, train_loader, args.num_classes, key, args) 
        # model = SCCLBert(sbert, cluster_centers=cluster_centers, alpha=args.alpha)  
        # model = model.cuda()

        # optimizer 
        # optimizer = torch.optim.Adam([
        #     {'params':model.sentbert.parameters()}, 
        #     {'params':model.head.parameters(), 'lr': args.lr*args.lr_scale},
        #     {'params':model.cluster_centers, 'lr': args.lr*args.lr_scale}], lr=args.lr)
        # print(optimizer)
        
        # set up the trainer    
        # learner = ClusterLearner(model, optimizer, args.temperature, args.base_temperature)
        # training(train_loader, learner, args, key)
    return None

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0], help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--print_freq', type=float, default=100, help="")  
    parser.add_argument('--result_path', type=str, default='./results/')
    parser.add_argument('--bert', type=str, default='distill', help="")
    # Dataset
    parser.add_argument('--intention_file_path', type=str, help="")
    parser.add_argument('--dataset', type=str, default='searchsnippets', help="")
    parser.add_argument('--data_path', type=str, default='../datasets/')
    parser.add_argument('--cluster_result_path', type=str, default='saved_result/island')
    parser.add_argument('--dataname', type=str, default='searchsnippets.csv', help="")
    parser.add_argument('--num_classes', type=int, default=8, help="")
    parser.add_argument('--max_length', type=int, default=32)
    # Learning parameters
    parser.add_argument('--lr', type=float, default=1e-5, help="")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--max_iter', type=int, default=3000)
    # contrastive learning
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--temperature', type=float, default=0.5, help="temperature required by contrastive loss")
    parser.add_argument('--base_temperature', type=float, default=0.1, help="temperature required by contrastive loss")
    # Clustering
    parser.add_argument('--use_perturbation', action='store_true', help="")
    parser.add_argument('--alpha', type=float, default=1.0)
    
    args = parser.parse_args(argv)
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None

    return args

if __name__ == '__main__':
    run(get_args(sys.argv[1:]))



    
