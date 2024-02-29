import os, inspect, pickle
import numpy as np
import pickle


def load(args):
    """
    Use parser to load the dataset and get train test split
    """
    dataset = parser(args.data, args.dataset).parse()

    current = os.path.abspath(inspect.getfile(inspect.currentframe()))
    Dir, _ = os.path.split(current)
    file = os.path.join(Dir, args.data, args.dataset, "splits", str(args.split) + ".pickle")

    if not os.path.isfile(file):
        print("split + ", str(args.split), "does not exist")
    with open(file, 'rb') as H: 
        Splits = pickle.load(H)
        train, valid, test = Splits['train'], Splits["valid"], Splits['test']
    
    return dataset, train, valid, test


class parser(object):
    """
    an object for parsing and laod data
    """
    
    def __init__(self, data, dataset):
        """
        initialises the data directory 

        arguments:
        data: single_modality/multimedia
        dataset: imdb/dblp/acm for coauthorship and painting for multimedia
        """
        
        current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.d = os.path.join(current, data, dataset)
        self.data, self.dataset = data, dataset

    

    def parse(self):
        """
        returns a dataset specific function to parse
        """
        
        name = "_load_data"
        function = getattr(self, name, lambda: {})
        return function()



    def _load_data(self):
        """
        loads the coauthorship hypergraph, features, and labels of cora

        assumes the following files to be present in the dataset directory:
        hypergraph.pickle: nodes of each hyperedge
        features.pickle: bag of word features
        labels.pickle: labels of papers/authors/movies

        n: number of hypernodes
        returns: a dictionary with hypergraph, features, and labels as keys
        """
        
        with open(os.path.join(self.d, 'hypergraph.pickle'), 'rb') as handle:
            hypergraph = pickle.load(handle)
            print("number of hyperedges in {} is : {}".format(self.dataset, len(hypergraph)))

        with open(os.path.join(self.d, 'features.pickle'), 'rb') as handle:
            features = pickle.load(handle)
            length = sum([len(i) for i in features])
            print("Shape of features in {} is : {}".format(self.dataset, length))

        with open(os.path.join(self.d, 'labels.pickle'), 'rb') as handle:
    
            labels = pickle.load(handle)
        
        if labels.shape[1]<3:
            labels = self._1hot(labels)
    
        print("labels is: ",labels[:3])
        print("number of labels in {} is : {}".format(self.dataset, labels.shape))
        
        with open(os.path.join(self.d, 'node_types.pickle'), 'rb') as handle:
            #node_type = self._1hot(pickle.load(handle)) 
            node_type = pickle.load(handle)


        return {'hypergraph': hypergraph, 'features': features, 'labels': labels, "node_type":node_type}



    def _1hot(self, labels):
        """
        converts each positive integer (representing a unique class) into ints one-hot form

        Arguments:
        labels: a list of positive integers with eah integer representing a unique label
        """
        
        sorted_labels = list(labels[labels[:, 0].argsort()][:,1]) # ???
        classes = set(sorted_labels)
        onehot = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        return np.array(list(map(onehot.get, sorted_labels)), dtype=np.int32)