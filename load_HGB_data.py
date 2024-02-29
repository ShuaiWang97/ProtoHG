from data_hete import data_HGB
import networkx as nx
import numpy as np
import scipy
import pickle
import scipy.sparse as sp
import torch
import argparse
import random

def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    # seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_hyper_data(prefix ='ACM'):
    set_seed(2048)
    # Chnage from graph structure to hypergraph structure:)
    dl = data_HGB.data_loader('data_hete/HGB_data/'+prefix)
    features = [] # Features in all types here.
    # A loop for each paper, find the corresponding author and subject
    hyperedge={}
    if "wikiart" in prefix :
        # A --- P* --- t
        #       |
        #       T
        # 0. painting  : [N1, 512]
        # 1. Artist    : [N2, 512]
        # 2. Date : [N3, N3]
        # 3. tags      : [N4, N4]
        for i in range(len(dl.nodes['count'])):
            th = dl.nodes['attr'][i]
            if th is not None:
                features.append(th)
        print("dl.nodes[shift] is: ",dl.nodes["shift"])
        #{0: 0, 1: 17785, 2: 17808} for artists
        type_mask = np.zeros(dl.nodes["count"].total(), dtype=int)
        
        for i in range(type_mask.shape[0]):
            for type in range(len(dl.nodes["shift"].keys())):
                if i>dl.nodes["shift"][type]:
                    type_mask[i] = type
        index = range(type_mask.shape[0])
        type_mask = np.vstack((index,type_mask)).transpose()

        # Different types of hyperedge
        if args.art_hyperedge  == "artists":
            for node_idx in np.arange(dl.nodes["shift"][1], dl.nodes["shift"][2]):
                hyperedge[node_idx] = []
                for link_type in [0]: 
                    links = np.array(dl.links["data"][link_type] ,dtype = int)  #get all link tuple for certain type
                    links = links[:,:2]
                    # 0 is painting and 1 is artist
                    link = np.where(links[:,1] == node_idx)[0]
                    hyperedge[node_idx].extend(links[link,0] )
                    
                    # Add paint-date link into hyperedge version
                    for i in links[link, 0]:
                        links = np.array(dl.links["data"][1] ,dtype = int)  #get all link tuple for certain type
                        links = links[:,:2]
                        link = np.where(links[:,0] == i)[0]
                        hyperedge[node_idx].extend(links[link,1] )
        elif args.art_hyperedge == "paintings":
            for node_idx in np.arange(dl.nodes["shift"][1]):
                hyperedge[node_idx] = []
                for link_type in [0, 1]:
                    links = np.array(dl.links["data"][link_type] ,dtype = int)
                    links = links[:,:2]
                    link = np.where(links[:,0] == node_idx)[0]
                    hyperedge[node_idx].extend(links[link,1] )

            ## sampling the hyperedge
            # keys = random.sample(hyperedge.keys(), 1000)
            # hyperedge = {k: hyperedge[k] for k in keys}
        print("len(hyperedge) is: ", len(hyperedge))




    if prefix == "ACM":
        # A --- P* --- S
        #       |
        #       T
        # 0. Paper     : [3025, 1902]
        # 1. Author    : [5959, 1902]
        # 2. Subject: [56, 1902]
        # 3. Term     : None
        for i in range(len(dl.nodes['count'])):
            th = dl.nodes['attr'][i]
            if th is not None:
                features.append(th)
        print("dl.nodes[shift] is: ",dl.nodes["shift"])

        #{0: 0, 1: 3025, 2: 8984, 3: 9040}
        type_mask = np.zeros(dl.nodes["shift"][3])
        #type_mask = np.zeros(dl.nodes["count"].total(), dtype=int)

        for i in range(type_mask.shape[0]):
            for type in range(len(dl.nodes["shift"].keys())):
                if i>=dl.nodes["shift"][type]:
                    type_mask[i] = type
        index = range(type_mask.shape[0])
        type_mask = np.vstack((index,type_mask)).transpose()

        links = np.array(dl.links["data"][4])

        ##Make code here
        for node_idx in np.arange(dl.nodes["shift"][1]):
            hyperedge[node_idx] = []
            for link_type in [0, 1]: # 6 is paper-term
                links = np.array(dl.links["data"][link_type] ,dtype = int)  #get all link tuple for certain type
                links = links[:,:2]
                link = np.where(links[:,:1] == node_idx)[0]
                hyperedge[node_idx].extend(links[link,1])

        print("len(hyperedge) is: ", len(hyperedge))

        
    elif prefix == "DBLP":
        # A* --- P --- T
        #        |
        #        V
        # 1. author: [4057, 334]
        # 2. paper : [14328, 4231]
        # 3. term  : [7723, 50]
        # 4. venue: None
        # link: {"1": "paper-term", "2": "paper-venue", "3": "paper-author"}
        # GCN, GAT, simple-HAN all use feat=2 for DBLP
        for i in range(len(dl.nodes['count'])):
            th = dl.nodes['attr'][i]
            if th is None:
                features.append(np.eye(dl.nodes['count'][i]))
            else:
                features.append(th)
        #features = np.array(features) # Not a array if size is not same
        print("dl.nodes[shift] is: ",dl.nodes["shift"])
        type_mask = np.zeros(dl.nodes["count"].total(), dtype=int)
        for i in range(type_mask.shape[0]):
            for type in range(len(dl.nodes["shift"].keys())):
                if i>dl.nodes["shift"][type]:
                    type_mask[i] = type
        index = range(type_mask.shape[0])
        type_mask = np.vstack((index,type_mask)).transpose()
    
        for node_idx in np.arange(dl.nodes["shift"][1], dl.nodes["shift"][2]):
            hyperedge[node_idx] = [node_idx]
            
            for link_type in [1, 2, 3]: # 2 is the conference features of paper in DBLP
                links = np.array(dl.links["data"][link_type] ,dtype = int)  #get all link tuple for certain type
                links = links[:,:2]
                link = np.where(links[:,:1] == node_idx)[0]
                hyperedge[node_idx].extend(links[link,1])
    elif prefix == "IMDB":    
        # A --- M* --- D
        #       |
        #       K
        # movie    : [4932, 3489]
        # director : [2393, 3341]
        # actor    : [6124, 3341]
        # keywords : None
        for i in range(len(dl.nodes['count'])):
            th = dl.nodes['attr'][i]
            if th is not None:
                features.append(th)
        #features = np.array(features) # Not a array if size is not same
        print("dl.nodes[shift] is: ",dl.nodes["shift"])
        type_mask = np.zeros(dl.nodes["count"].total(), dtype=int)
        for i in range(type_mask.shape[0]):
            for type in range(len(dl.nodes["shift"].keys())):
                if i>dl.nodes["shift"][type]:
                    type_mask[i] = type
        index = range(type_mask.shape[0])
        type_mask = np.vstack((index,type_mask)).transpose()
    
        for node_idx in np.arange(dl.nodes["shift"][1]):
            hyperedge[node_idx] = []
            
            for link_type in [0,2 ]: # 4 is the key words features of movie in IMDB
                links = np.array(dl.links["data"][link_type] ,dtype = int)  #get all link tuple for certain type
                links = links[:,:2]
                link = np.where(links[:,:1] == node_idx)[0]
                hyperedge[node_idx].extend(links[link,1])
    




    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    
    # Get train etxt split
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    print("train_idx[:20] :" , train_idx[:20])
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    labels[test_idx] = dl.labels_test['data'][test_idx]
    print("labels: ",labels)
    

    if prefix != 'IMDB':
        labels = labels.argmax(axis=1)
        # somthing special for IMDB, becasue there are multiple labels here

        index = range(labels.shape[0])
        labels = np.vstack((index,labels)).transpose()


    train_val_test_idx = {}
    train_val_test_idx['train'] = train_idx
    print("train_val_test_idx",train_val_test_idx)
    train_val_test_idx['valid'] = val_idx
    train_val_test_idx['test'] = test_idx
    print("train, valid, test split: ", len(train_idx), len(val_idx), len(test_idx))


    print("labels: ",labels)
    
    return features,\
        hyperedge, \
        labels,\
        train_val_test_idx,\
        type_mask,\
        dl



def main(args):
    save_prefix = 'data_hete/HGB_hyper_data/{}/'.format(args.save)
    features, hyperedge, labels, train_val_test_idx, type_mask, dl = load_hyper_data(args.data)

    with open(save_prefix +"node_types.pickle",'wb') as f:
        pickle.dump(type_mask, f)

    with open(save_prefix +"labels.pickle",'wb') as f:
        pickle.dump(labels, f)

    with open(save_prefix +"hypergraph.pickle",'wb') as f:
        pickle.dump(hyperedge, f)

    with open(save_prefix +"features.pickle",'wb') as f:
        pickle.dump(features, f)

    with open(save_prefix + "splits/0.pickle",'wb') as f:
        pickle.dump(train_val_test_idx, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type = str,
        default="wikiart_artists",
        help = "The name of the dataset to preprocess",
    )

    parser.add_argument(
        "--art_hyperedge",
        type = str,
        default="paintings",
        help = "The name of the dataset to preprocess",
    )

    parser.add_argument(
        "--save",
        type = str,
        default="wikiart_artists",
        help = "The save name of the dataset ",
    )
    args = parser.parse_args()

    args.save = "{}_E_{}".format(args.data, args.art_hyperedge)
    main(args)
