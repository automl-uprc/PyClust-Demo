import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
import dgl
import torch
import os
from collections import defaultdict
from dgl.data.utils import save_graphs




def graph_implementation(edge_list_coo,
                         node_emb_path: str):
    graph = dgl.from_scipy(edge_list_coo, eweight_name='weight')
    undir_graph = dgl.add_reverse_edges(graph, copy_edata=True)
    if '_ID' in undir_graph.edata:
        undir_graph.edata.pop('_ID')

    with open(node_emb_path, 'r') as f:
        line = f.readline()
        line = f.readline()
        node_feat_dict = {}
        while line:
            cur_line = line.split(" ")
            node_feat_dict[int(cur_line[0])] = [float(val) for val in cur_line[1:]]
            line = f.readline()

    sorted_node_feat_dict = dict(sorted(node_feat_dict.items(), key=lambda e: e[0]))
    node_feat_array = np.array(list(sorted_node_feat_dict.values()))
    undir_graph.ndata['h'] = torch.FloatTensor(node_feat_array)

    return undir_graph


def convert_to_graph(dataset, edgelist_dir, nodeEmbeddings_dir, dataset_name=""):
    """
    This function converts a dataset into a graph and calculates node embeddings based on weighted  deepwalk.

    :param dataset_name:
    :type dataset_name:
    :param dataset: Either the path to the dataset or the dataset as a np.array.
    :type dataset: str | pd.DataFrame |np.ndarray
    :param edgelist_dir: The directory to save graph's edges.
    :type edgelist_dir: str
    :param nodeEmbeddings_dir: The directory to save graph's node embeddings.
    :type nodeEmbeddings_dir: str
    :return:
    :rtype:
    """

    if type(dataset) is str:
        dataset = pd.read_csv(dataset)
        dataset_name = dataset.split('/')[-1].split('.')[0]
    else:
        dataset = apply_pca(dataset=dataset)
        dataset_name = dataset_name

    # --- Find Edges ---
    # cosine similarity matrix, NxN with values in [0,1]
    cosine_matrix = cosine_similarity(dataset)
    return cosine_matrix
    # Graph's edge list. List of lists.
    path_in = os.path.join(os.getcwd(), edgelist_dir, dataset_name + '.file')
    command, edge_list = edge_list_creation(cosine_matrix=cosine_matrix,
                                            path_to_write=path_in)
    return command, edge_list

    # # there are some cases where the graph created from a dataset
    # # was empty
    # # this check is applied because deepwalk can't
    # # produce node embeddings for an empty graph
    print(dataset_name)
    if command == "Proceed":
        # create node embeddings with deepwalk
        # store them in NodeEmb dir
        path_out = os.path.join(os.getcwd(), nodeEmbeddings_dir, dataset_name + '.file')
        cmd = 'deepwalk --input ' + path_in + ' --format weighted_edgelist --output ' + path_out
        os.system(cmd)

        graph = graph_implementation(edge_list, path_out)
        return graph


def get_label(ranking_path: str, labels_dict):
    ranking = pd.read_csv(ranking_path, index_col=0)

    cvis = ['DB', 'HL', 'HKK', 'Xie', 'Scat', 'SIL', 'CH', 'DU', 'BP', 'MC', 'AVG']

    for cvi in cvis:
        label = ranking.loc[cvi].argmin()
        labels_dict[cvi].append(label)

    return labels_dict


def graph_dataset_creation(graphs, labels):
    graphs_repo = os.path.join(os.getcwd(), 'metadata', 'graphs_dataset_per_cvis')
    if not os.path.exists(graphs_repo):
        os.mkdir(graphs_repo)

    for key, values in labels.items():
        if key == 'mapping_index':  # to drop if it's not working
            continue
        path_to_save = os.path.join(graphs_repo, 'graph_data_' + key + '.dgl')
        graph_labels = {"glabel": torch.tensor(values),
                        "mapping_index": torch.IntTensor(labels['mapping_index'])}  # to modify if it's not working
        save_graphs(path_to_save, graphs, graph_labels)

    return graphs_repo


def dataset2graph(processed_datasets_path: str,
                  edgelist_dir_name: str,
                  nodeEmbeddings_dir_name: str,
                  ranking_path: str):
    """
    This functions creates an iteration over the processed datasets in order to covert them into a graph.

    :param processed_datasets_path: Path of datasets directory. Data should be in .csv format and optionally
            preprocessed.
    :type processed_datasets_path: str
    :param edgelist_dir_name: Name for directory to store Edgelists
    :type edgelist_dir_name: str
    :param nodeEmbeddings_dir_name: Name for directory to store Node Embeddings
    :type nodeEmbeddings_dir_name: str
    :param ranking_path:
    :type ranking_path:
    :return:
    :rtype:
    """

    # create directories to store edgelists and node embeddings
    for dir_ in [edgelist_dir_name, nodeEmbeddings_dir_name]:
        dir_to_create = os.path.join(os.getcwd(), dir_)
        if not os.path.exists(dir_to_create):
            os.mkdir(dir_to_create)

    list_of_datasets = os.listdir(processed_datasets_path)
    graphs = []
    labels_cvi_dict = defaultdict(list)
    mapping_dict = {}  # to delete if it's not working

    # iterate over processed datasets
    for idx, dataset_file in enumerate(list_of_datasets):

        # dataset_name = dataset_file.split('.')[0]+'.file'
        # check_existance = os.path.join(os.getcwd(), nodeEmbeddings_dir_name, dataset_name)
        # if os.path.exists(check_existance):
        #    print(f'{dataset_name} embeddings exist already!')
        #    continue

        dataset_path = os.path.join(processed_datasets_path, dataset_file)

        # convert dataset to graph
        graph = convert_to_graph(dataset_path=dataset_path,
                                 edgelist_dir=edgelist_dir_name,
                                 nodeEmbeddings_dir=nodeEmbeddings_dir_name
                                 )

        if graph is not None:
            graphs.append(graph)
            # get labels of the graph for each cvi
            ranking_dataset_dir = os.path.join(ranking_path, dataset_file)
            labels_cvi_dict = get_label(ranking_dataset_dir, labels_cvi_dict)
            labels_cvi_dict['mapping_index'].append(idx)  # to delete if it's not working
            mapping_dict[idx] = dataset_file  # to delete if it's not working

    graphs_directory = graph_dataset_creation(graphs, labels_cvi_dict)

    print('Graph Representation Step has been Completed')
    return graphs_directory, mapping_dict
