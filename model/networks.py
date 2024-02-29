import torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
from torch import linalg as LA

from torch.autograd import Variable
from model import layer 
import ipdb
import copy
import pdb



class Hyper_Atten_Block(nn.Module):
    def __init__(self, input_size, head_num, head_hid, node_type, dropout):
        super().__init__()
        self.dropout = dropout
        self.gat = nn.ModuleList( [layer.HyperGraphAttentionLayerSparse(input_size, head_hid,node_type,  dropout=self.dropout, alpha=0.2, transfer=True,
                                            concat=True) for _ in range(head_num)])

        self.LayerNorm = nn.LayerNorm([input_size])
        self.hm = nn.Sequential(nn.Linear(head_num * head_hid, input_size),
                                nn.LeakyReLU(),
                                nn.Dropout(self.dropout),
                                 )

        # Following feedforward network for transformer
        self.ffn = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.LayerNorm([input_size]),
            nn.Dropout(self.dropout),
        )
        

    def forward(self, X, E, epoch, Y, idx_train):
        
        # residual connection?? add node/edge type in it
        X_res = X
        
        X = torch.concat([gat(X, E)[0] for gat in self.gat], dim=-1)
    
        #head_num * head_hid -> input_size
        #X = self.hm(X)  
        #X = self.LayerNorm(X + X_res)
        
        # #Feedforward
        # X_res = X
        # X = self.ffn(X) 
        # X = self.LayerNorm(X + X_res)
        return X

class HGNN_ATT_MH(nn.Module):
    def __init__(self, features_list, node_type, args, Y, idx_train, head_num=4, layers=3):
        super(HGNN_ATT_MH, self).__init__()
         
        self.dropout, c, num_hidden, self.node_type = args.dropout, args.c, args.num_hidden, node_type
        self.dataset = args.dataset

        head_hid = int(num_hidden / head_num) #diemension for each head


        in_dims = [features.shape[1] for features in features_list]
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # for each head, learn num_hidden->head_hid, then concat each head [head_hid_1, head_hid_2,...]
        self.hatt = nn.ModuleList([Hyper_Atten_Block(num_hidden, head_num, head_hid, node_type, self.dropout) for _ in range(layers)])

        
        self.linear = nn.Linear(num_hidden, args.pro_hidden) #16 for acm,dblp
        self.linear_1 = nn.Linear(args.pro_hidden, c)
        # Map into a 2d feature space
        self.dce = dce_loss(args.loss, args.pro_hidden, c)


    def forward(self, features_list, E, epoch, Y, idx_train):

        if type(features_list) == list:
            H=[]
            for fc, feature in zip(self.fc_list, features_list):
                H.append(fc(feature))
            H = torch.cat(H, dim=0) #(n,d)
        else:
            H = features_list
        
        
        reg=0
        #  E(3025,9040) * H(9040,64) ->(3025, 64) E is binary, E_p is
        if epoch >=0:
            E_num = torch.sum(E,1).float()
            E_p = torch.matmul(E.float(), H)/E_num[:, None] #(3025, 64) / (3025,1 )

            S = torch.matmul(H, E_p.transpose(0, 1).float()) #(N,D) * (D, M) -> (N, M)1GB
            
            reg = torch.norm(input= (S.transpose(0, 1) - E), dim=None,p=2) #5GB memory


        for att in self.hatt:
            H = att(H, E, epoch, Y, idx_train) #no +H for residual connection

        H = F.dropout(H, self.dropout, training=self.training)
        features = torch.squeeze(H)
        H = self.linear(features)
        
        # Learn class center and output
        centers, output = self.dce(H, Y, idx_train, epoch)

        ## uncomment to not use prototype classifier
        # output = self.linear_1(H)


        #return features, centers, F.log_softmax(H, dim=1), 0.0001 * reg
        return H, centers, output, reg

class dce_loss(torch.nn.Module):
    # Calculate which class should be outputed

    def __init__(self, loss, feat_dim, n_classes, init_weight=True):
   
        super(dce_loss, self).__init__()
        self.n_classes=n_classes
        self.feat_dim=feat_dim

        # initialize nodes
        self.centers=nn.Parameter(torch.randn(self.feat_dim, self.n_classes).cuda(), requires_grad=True)
        self.loss = loss
        self.plot = False
        
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)

    def forward(self, X, Y, idx_train, epoch):
        """
        input x.shape (N, D)
        Do cosine similarity
        """
        # if epoch==0:
        #     centers = torch.zeros(self.feat_dim, self.n_classes).cuda()
        #     for i in range(self.n_classes):
        #         if len(Y.shape)>1:
        #             idxs = ((Y[idx_train][:,i] == 1).nonzero(as_tuple=True)[0])
        #         else:
        #             idxs = ((Y[idx_train] == i).nonzero(as_tuple=True)[0])
        #         centers[:, i] =  torch.mean(X[idx_train][idxs], dim =0)
        #     self.centers = torch.nn.Parameter(centers.cuda(), requires_grad=True)


        if self.loss== "dis":
            ## Distance loss
            features_square = torch.sum(torch.pow(X,2),1, keepdim=True) # (50,1) distance from x to 0
            centers_square = torch.sum(torch.pow(self.centers,2),0, keepdim=True) # (1,10) distance from
            # #                            x.shape (N, D) self.centers (D,C)
            features_into_centers = 2 * torch.matmul(X, self.centers) #Cos similarity between nodes and centers 
            # #              (50,1)        (1,10)            (50,10)
            dist = features_square + centers_square - features_into_centers
        elif self.loss=="sim":
            # Similarity loss
            X_norm = X / X.norm(dim=1)[:, None]
            centers_norm = self.centers / self.centers.norm(dim=1)[:, None]
            sim = 2 * torch.matmul(X_norm, centers_norm)
            dist = -sim

        return self.centers, -dist

def regularization(features, centers, labels):
        """
        natural Proto Loss can be added to our CPL framework, to pull the feature vector closer to their corresponding
        prototypes (genuine class representation).
        """
        # features (N, D), centers (D, C),  labels (N)
        distance = (features-torch.t(centers)[labels])


        distance = torch.sum(torch.pow(distance,2), 1, keepdim=True)

        distance = (torch.sum(distance, 0, keepdim=True))/features.shape[0]

        return 0.01*distance

def cos_dis(X):
        """
        cosine distance
        :param X: (N, d)
               XT: (d, N)
        :return: (N, N)
        """
        X = nn.functional.normalize(X)
        XT = X.transpose(0, 1)
        return torch.matmul(X, XT)

def distance(X,Y):
    """
    input X(m,d) and Y(n,d), return (m,n) the distance for each node in X with each node in Y
    (N,M)  (N,M)
    """
    # this has the same affect as taking the dot product of each row with itself
    x2 = torch.sum(X**2, axis=1) # shape of (m)
    y2 = torch.sum(Y**2, axis=1) # shape of (n)

    # we can compute all x_i * y_j and store it in a matrix at xy[i][j] by
    # taking the matrix multiplication between X and X_train transpose
    # if you're stuggling to understand this, draw out the matrices and
    # do the matrix multiplication by hand
    # (m, d) x (d, n) -> (m, n)
    xy = torch.matmul(X, Y.T)

    # each row in xy needs to be added with x2[i]
    # each column of xy needs to be added with y2[j]
    # to get everything to play well, we'll need to reshape
    # x2 from (m) -> (m, 1), numpy will handle the rest of the broadcasting for us
    # see: https://numpy.org/doc/stable/user/basics.broadcasting.html
    x2 = x2.reshape(-1, 1)
    dists = torch.sqrt(x2 - 2*xy + y2) # (m, 1) repeat columnwise + (m, n) + (n) repeat rowwise -> (m, n)
    # print(x2, x2.shape)
    # print(y2, y2.shape)
    # print(2*xy)

    return dists