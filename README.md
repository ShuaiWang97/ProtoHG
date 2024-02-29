# ProtoHG
Implementation for paper: Prototype-Enhanced Hypergraph Learning for Heterogeneous Information Networks.  International conference of Multimedia Modelling 2024
https://arxiv.org/pdf/2309.13092.pdf

## Pipeline
<div align=center>
<img src='pipeline.jpg' width='800'>
</div>

## Requirements
The variety and complexity of relations in multimedia data lead to Heterogeneous Information Networks (HINs). Capturing the semantics from such networks requires approaches capable of utilizing the full richness of the HINs. Existing methods for modeling HINs employ techniques originally designed for graph neural networks, and HINs decomposition analysis, like using manually predefined metapaths. In this paper, we introduce a novel prototype-enhanced hypergraph learning approach for node classification in HINs. Using hypergraphs instead of graphs, our method captures higher-order relationships among nodes and extracts semantic information without relying on metapaths. Our method leverages the power of prototypes to improve the robustness of the hypergraph learning process and creates the potential to provide human-interpretable insights into the underlying network structure. Extensive experiments on three real-world HINs demonstrate the effectiveness of our method.




## Requirements
2.1 Main python packages and their version
- Python3
- torch                         1.13.1+cu116
- torch-scatter                 2.1.1+pt113cu117
- torch-sparse                  0.6.17+pt113cu117

## Hardware
- 1 NVIDIA A100

## Datasets
ACM, DBLP, WikiArt

## Structure
```
|- data_hete
    |- ACM_hete
    |- DBLP_hete
    |- wikiart_hete
|- model
    networks.py
    layer.py
main_transformer.py
utils.py
```

## Training

taking ACM dataset as an example:
python main_transformer.py --data HGB_hyper_data --dataset ACM_hete --num_hidden 64 --n_head 4 --n_layer 3 --reg_p 1e-6 --loss sim --lr 0.01 

## Acknowledgement
This repo is based on [HyperSage](https://github.com/worring/HyperMessage) and [HEGEL: Hpyergraph Transformer](https://github.com/hpzhang94/hegel_sum), thanks for their excellent work.


If you find this repo useful, please consider cite: 
```
@misc{wang2023prototypeenhanced,
      title={Prototype-Enhanced Hypergraph Learning for Heterogeneous Information Networks}, 
      author={Shuai Wang and Jiayi Shen and Athanasios Efthymiou and Stevan Rudinac and Monika Kackovic and Nachoem Wijnberg and Marcel Worring},
      year={2023},
      eprint={2309.13092},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


```

 

