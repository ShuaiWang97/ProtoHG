import wandb
import torch.nn.functional
import torch.optim as optim
import torch
import umap.plot
import umap
from pynvml.smi import nvidia_smi
import utils
from model import networks
from collections import defaultdict
from data_hete import data
from config import config
import torch
import datetime
import time
import numpy as np
args = config.parse()
timestamp = time.time()
timestamp = datetime.datetime.fromtimestamp(
    int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is: ", device)


def getMemoryUsage():
    nvsmi = nvidia_smi.getInstance()
    usage = nvsmi.DeviceQuery("memory.used")["gpu"][0]["fb_memory_usage"]
    return "%d %s" % (usage["used"], usage["unit"])


def data_process(args, device):
    """
    preprocess the hypergraph data
    """
    dataset, idx_train, idx_val, idx_test = data.load(args)
    print("len(idx_train) is {} len(ids_val) is {} len(idx_test) is {}".format(
        len(idx_train), len(idx_val), len(idx_test)))
    # for ACM, X is paper + author + subject features
    features_list, Y, node_type, E = dataset['features'], dataset[
        'labels'], dataset['node_type'], dataset['hypergraph']

    # k is index node and V are hyperedge nodes
    # for ACM dataset, K should be paper id, E is subject and author
    for k, v in E.items():
        E[k] = list(v)
        if k not in E[k]:
            E[k].append(k)

    print("avg nodes in hyperedge: ", np.mean(
        [len(value) for key, value in E.items()]))
    features_list = [torch.FloatTensor(features) for features in features_list]
    node_num = sum([features.shape[0] for features in features_list])

    edge_feature = utils.build_hyperedge(list(E.values()))
    edge_feature = torch.transpose(edge_feature, 0, 1)
    # Conrtsuct weights oby dimension of global_neighborhood_count and edge_count
    global_neighborhood = defaultdict(list)
    edge_count = np.zeros(node_num)
    global_neighborhood_count = np.zeros(node_num)
    unique_nodes = []
    for edge, nodes in E.items():
        for node in nodes:
            unique_nodes.append(node)
            neighbor_nodes = ([i for i in nodes if i != node])

            edge_count[node] = edge_count[node] + 1
            global_neighborhood[node].extend(neighbor_nodes)
    for k, v in global_neighborhood.items():
        global_neighborhood_count[k] = len(set(v))
    unique_nodes = list(set(unique_nodes))
    input_weight = np.concatenate((np.expand_dims(global_neighborhood_count, axis=1), np.expand_dims(
        edge_count, axis=1)), axis=1)  # (node_num, 2)
    print("average egde_count: ", np.mean(edge_count))

    if "IMDB" and "DBLP" not in args.dataset:
        features_list = [utils.normalize_matrxi(X) for X in features_list]

    args.c = Y.shape[1]
    Y = np.array(Y)
    Y = torch.LongTensor(np.where(Y)[1]).to(device)

    edge_count = torch.Tensor(edge_count).to(device)
    node_type = torch.LongTensor(node_type).to(device)
    edge_feature = edge_feature.to(device)
    # input_weight = torch.FloatTensor(input_weight).to(device)
    features_list = [features.to(device) for features in features_list]

    return features_list, Y, edge_feature, idx_train, idx_val, idx_test, input_weight, E, edge_count, node_type


def vis(features, centers, labels, dataset):
    """
    visualize the representatin space of nodes
    """

    pdb.set_trace()
    all_features = torch.concatenate(
        (features, centers)).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    umap_args_model = {
        "n_neighbors": 100,
        "n_components": 2,  # 5 -> 2 for plotting
        "metric": "cosine",
    }
    umap_model = umap.UMAP(**umap_args_model).fit(all_features)

    # umap_document_vectors = umap_model.embedding_[:-top_num,:]
    # umap_topic_vectors = umap_model.embedding_[-top_num::,:]
    np.save("result/features_ln_{}".format(dataset), umap_model.embedding_)
    np.save("result/labels_{}".format(dataset), labels)

    umap.plot.points(umap_model, labels=labels)

    umap.plot.plt.savefig(
        "result/representation_space_{}.jpg".format(dataset), dpi=300)


def main(args):
    # load data
    utils.set_seed(args.seed)
    features_list, Y, edge_feature, idx_train, idx_val, idx_test, input_weight, E, edge_count, node_type = data_process(
        args, device)

    model = networks.HGNN_ATT_MH(features_list, node_type, args, Y,
                                 idx_train, head_num=args.n_head, layers=args.n_layer).to(device)
    optimiser = optim.Adam(list(model.parameters()),
                           lr=args.lr, weight_decay=args.decay)

    # Train model
    early_stopping = utils.EarlyStopping(
        patience=150, verbose=True, save_path='checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, timestamp))
    loss = torch.nn.CrossEntropyLoss()

    if args.wandb == True:
        print("args.wandb", args.wandb)
        wandb.init(name="Transformer hyper partit chunk:{}, head:{}, layer{}".format(
            1, args.n_head, args.n_layer), project="{}-{}".format(args.data, args.dataset), config=args)

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimiser.zero_grad()

        # node_idx, sub_features_list, sub_edge_feature = sampling(features_list, edge_feature, E, input_weight)
        features, centers, output, reg = model(
            features_list, edge_feature, epoch, Y, idx_train)

        reg_d = networks.regularization(
            features[idx_train], centers, Y[idx_train])
        loss_train = loss(output[idx_train], Y[idx_train])
        loss_train = loss_train + args.reg_p * reg + args.reg_d*reg_d
        f1_micro_train, f1_macro_train = utils.f1_score_(
            labels=Y[idx_train], pred=output[idx_train])

        loss_train.backward()
        optimiser.step()

        # Eval
        model.eval()
        features, centers, output, reg = model(
            features_list, edge_feature, epoch, Y, idx_train)
        # early stopping
        loss_val = loss(output[idx_val], Y[idx_val])
        f1_micro_val, f1_macro_val = utils.f1_score_(
            labels=Y[idx_val], pred=output[idx_val])

        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            print('Early stopping!')
            break

        if epoch % 5 == 0:

            print('Epoch: {:04d}'.format(epoch+1), 'loss_val: {:.4f}'.format(loss_val.item()), 'f1_micro_val: {:.4f}'.format(f1_micro_val.item()),
                  'f1_macro_val: {:.4f}'.format(f1_macro_val.item()), 'time: {:.4f}s'.format(time.time() - t))

            f1_micro_test, f1_macro_test = utils.f1_score_(
                labels=Y[idx_test], pred=output[idx_test])

            if args.wandb == True:
                wandb.log({"loss_train ": loss_train, "f1_macro_train ": f1_macro_train,
                          "f1_micro_train ": f1_micro_train})
                wandb.log(
                    {"loss_val ": loss_val, "f1_macro_val ": f1_macro_val, "f1_micro_val ": f1_micro_val})
                wandb.log({"f1_macro_test ": f1_macro_test,
                          "f1_micro_test ": f1_micro_test})

        if epoch % 5 == 0:
            # print("memory free and total: ",torch.cuda.mem_get_info())
            print("GPU Memory: %s" % getMemoryUsage())

    print('----------------------------------------------------------------')
    print("Training done! ")

    # Load best model and test
    model = torch.load(
        'checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, timestamp))
    model.eval()
    features, centers, output, reg = model(
        features_list, edge_feature, epoch, Y, idx_train)

    # vis(features, centers.T, Y, args.dataset)

    f1_micro_test, f1_macro_test = f1_score_(
        labels=Y[idx_test], pred=output[idx_test])
    print('Test summary: f1_micro_test : {:.4f}'.format(f1_micro_test.item()),
          'f1_macro_test : {:.4f}'.format(f1_macro_test.item()), 'time: {:.4f}s'.format(time.time() - t))

    return [f1_micro_test*100, f1_macro_test*100]


if __name__ == "__main__":
    res = []
    for seed in args.seeds:
        args.seed = seed
        result = main(args)
        res.append(result)
    print(res)
    utils.store_result(
        np.array(res), "result/{}_{}.txt".format(args.out, args.dataset))
