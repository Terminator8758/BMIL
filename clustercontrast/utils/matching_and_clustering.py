import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
import copy 
from scipy.optimize import linear_sum_assignment as linear_assignment
import ot


def cluster_acc(pred_label, gt_label):
    D = max(pred_label.max(), gt_label.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)  # w=[[1 5 0],[4 1 1],[0 2 3]]
    for i in range(pred_label.size):
        w[pred_label[i], gt_label[i]] += 1
    ind = linear_assignment(w.max() - w) # assignment= [[0 1],[1 0],[2 2]]
    ind = np.vstack(ind).T
    acc = sum([w[i, j] for i, j in ind]) * 1.0 / len(pred_label)
    return acc


def subset_clustering(feats, sample_ratio, cluster_algo, k1=30, k2=6, only_cluster_one_subset=False):
    rgb_img_num = len(feats)
    subset_num = int(1 / sample_ratio)
    subset_sz = int(rgb_img_num * sample_ratio)
    shuffled_inds = torch.randperm(rgb_img_num)
    count = 0
    pseudo_labels_rgb = -1 * np.ones((rgb_img_num,), dtype=np.int64)
    index_to_subset_mapper = -1 * torch.ones((rgb_img_num,))
    subset_label_dict = {}
    
    all_subset_inds = []
    for i in range(subset_num):
        subset_inds = shuffled_inds[i * subset_sz: (i + 1) * subset_sz]
        all_subset_inds.append(subset_inds)

    if only_cluster_one_subset:
        subset_num = 1

    for i in range(subset_num):
        subset_inds = shuffled_inds[i * subset_sz: (i + 1) * subset_sz]
        rerank_dist_rgb = compute_jaccard_distance(feats[subset_inds], k1=k1, k2=k2, search_option=3)
        subset_labels = cluster_algo.fit_predict(rerank_dist_rgb)
        val_inds = np.where(subset_labels >= 0)[0]
        subset_labels[val_inds] = subset_labels[val_inds] + count  # accumulate subset label, keep the -1 unchanged

        pseudo_labels_rgb[subset_inds] = subset_labels
        num_clusters = len(set(subset_labels)) - (1 if -1 in subset_labels else 0)
        count += num_clusters
        num_outliers = len(np.where(subset_labels == -1)[0])
        print('   subset {}: generated {} clusters, {} outliers from {} instances'.format(i, num_clusters, num_outliers,
                                                                                          len(subset_inds)))
        subset_labels = torch.from_numpy(subset_labels).long()
        uniq_labels = torch.unique(subset_labels[subset_labels >= 0])
        subset_label_dict[i] = uniq_labels.cuda()  # store the pseudo class of this subset
        index_to_subset_mapper[subset_inds] = i  # store the subset index of each image

    return subset_label_dict, index_to_subset_mapper, pseudo_labels_rgb, all_subset_inds

