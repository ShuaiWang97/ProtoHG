'''
data: acm_hyper
dataset: ACM/DBLP/ICDM
'''
data = "NC_v0"
dataset = "ACM_hete"

'''
gpu: gpu number to use
cuda: True or False
seed: an integer list
'''
gpu = 0
cuda = True
seeds = [40, 41, 42] 
split = 1


'''
model related parameters
depth: number of hidden layers
dropout: dropout probability for  hidden layer
epochs: number of training epochs
'''
depth = 2
dropout = 0.5
epochs = 500

'''
parameters for optimisation
rate: learning rate
decay: weight decay
'''
lr = 0.01 #0.005 for MAGNN, 0.0001 for ACM dataset(93.00), 0.01 for DBLP(93.87)
decay = 0.0005 #weight decay (L2 penalty) set to 0.001;


# data preprocess
num_hidden = 64
pro_hidden = 16

# prototype
reg_p = 0
reg_d = 0
loss="sim"
n_head=4
n_layer=2


# kn=4
# sample = "hyperedge"

import configargparse, os,sys,inspect
from configargparse import YAMLConfigFileParser



def parse():
	"""
	adds and parses arguments / hyperparameters
	"""
	default = os.path.join(current(), data + ".yml")
	p = configargparse.ArgParser(config_file_parser_class = YAMLConfigFileParser, default_config_files=[default])
	p.add('-c', '--my-config', is_config_file=True, help='config file path')
	p.add('--data', type=str, default=data, help='data name (coauthorship/cocitation)')
	p.add('--dataset', type=str, default=dataset, help='dataset name (e.g.: cora/dblp/acm for coauthorship, cora/citeseer/pubmed for cocitation)')
	p.add('--split', type=int, default=split, help='train-test split used for the dataset')
	p.add('--depth', type=int, default=depth, help='number of hidden layers')
	p.add('--dropout', type=float, default=dropout, help='dropout for hidden layer')
	p.add('--lr', type=float, default=lr, help='learning rate')
	p.add('--decay', type=float, default=decay, help='weight decay')
	p.add('--epochs', type=int, default=epochs, help='number of epochs to train')
	p.add('--gpu', type=int, default=gpu, help='gpu number to use')
	
	p.add('--wandb', default=True, action='store_false', help='use wandb or not')
	p.add('--cuda',  type=bool, default=cuda, help='cuda for gpu')


	p.add('--num_hidden', type=int, default=num_hidden, help='the num_hidden of preprocess heterogenous data')
	p.add('--pro_hidden', type=int, default=pro_hidden, help='the num_hidden prototype input')
	p.add('--reg_p', type=float, default=reg_p, help='rate for similarity reg')
	p.add('--reg_d', type=float, default=reg_d, help='rate for distance reg')

	## Transformer parameter
	p.add('--n_head', type=int, default=n_head, help='number of head for transformer')
	p.add('--n_layer', type=int, default=n_layer, help='number of layer for transformer')
	p.add('--loss', type=str, default=loss, help='type of loss for prototype')
	p.add('--seeds', nargs='+', default=seeds,type=int, help='<Required> Set flag', required=True)
	p.add('--out', type=str, default="output", help='file to store statistics of experiment')
	p.add('-f') # for jupyter default

	print(p.parse_args())
	return p.parse_args()


def current():
	"""
	returns the current directory path
	"""
	current = os.path.abspath(inspect.getfile(inspect.currentframe()))
	head, tail = os.path.split(current)
	return head