#!/usr/bin/env python
# coding: utf-8
# In[1]:
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import scipy.io
import os
import sys
import STitch3D
from STitch3D.utils import align_spots, preprocess, construct_dgl_graph
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# In[2]:
mat = scipy.io.mmread(
    "/data/renlian/snRNAseq_brain/GSE144136_GeneBarcodeMatrix_Annotated.mtx")
meta = pd.read_csv(
    "/data/renlian/snRNAseq_brain/GSE144136_CellNames.csv", index_col=0)
meta.index = meta.x.values
group = [i.split('.')[1].split('_')[0] for i in list(meta.x.values)]
condition = [i.split('.')[1].split('_')[1] for i in list(meta.x.values)]
celltype = [i.split('.')[0] for i in list(meta.x.values)]
meta["group"] = group
meta["condition"] = condition
meta["celltype"] = celltype
genename = pd.read_csv(
    "/data/renlian/snRNAseq_brain/GSE144136_GeneNames.csv", index_col=0)
genename.index = genename.x.values
# adata_ref.X: celltype*gene matrix
adata_ref = ad.AnnData(X=mat.tocsr().T)
adata_ref.obs = meta
adata_ref.var = genename
adata_ref = adata_ref[adata_ref.obs.condition.values.astype(str) == "Control", :]
# In[3]:
anno_df = pd.read_csv(
    '/data/renlian/HumanPilot/10X/barcode_level_layer_map.tsv', sep='\t',
    header=None)

slice_idx = [151673, 151674, 151675, 151676]

adata_st1 = sc.read_visium(
    path="/data/renlian/HumanPilot/10X/%d" % slice_idx[0],
    count_file="%d_filtered_feature_bc_matrix.h5" % slice_idx[0])
anno_df1 = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx[0])]
anno_df1.columns = ["barcode", "slice_id", "layer"]
anno_df1.index = anno_df1['barcode']
adata_st1.obs = adata_st1.obs.join(anno_df1, how="left")
adata_st1 = adata_st1[adata_st1.obs['layer'].notna()]

adata_st2 = sc.read_visium(
    path="/data/renlian/HumanPilot/10X/%d" % slice_idx[1],
    count_file="%d_filtered_feature_bc_matrix.h5" % slice_idx[1])
anno_df2 = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx[1])]
anno_df2.columns = ["barcode", "slice_id", "layer"]
anno_df2.index = anno_df2['barcode']
adata_st2.obs = adata_st2.obs.join(anno_df2, how="left")
adata_st2 = adata_st2[adata_st2.obs['layer'].notna()]

adata_st3 = sc.read_visium(
    path="/data/renlian/HumanPilot/10X/%d" % slice_idx[2],
    count_file="%d_filtered_feature_bc_matrix.h5" % slice_idx[2])
anno_df3 = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx[2])]
anno_df3.columns = ["barcode", "slice_id", "layer"]
anno_df3.index = anno_df3['barcode']
adata_st3.obs = adata_st3.obs.join(anno_df3, how="left")
adata_st3 = adata_st3[adata_st3.obs['layer'].notna()]

adata_st4 = sc.read_visium(
    path="/data/renlian/HumanPilot/10X/%d" % slice_idx[3],
    count_file="%d_filtered_feature_bc_matrix.h5" % slice_idx[3])
anno_df4 = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx[3])]
anno_df4.columns = ["barcode", "slice_id", "layer"]
anno_df4.index = anno_df4['barcode']
adata_st4.obs = adata_st4.obs.join(anno_df4, how="left")
adata_st4 = adata_st4[adata_st4.obs['layer'].notna()]
# In[4]:
adata_st_list_raw = [adata_st1, adata_st2, adata_st3, adata_st4]
adata_st_list = align_spots(adata_st_list_raw, plot=False)
celltype_list_use = [
    'Astros_1', 'Astros_2', 'Astros_3', 'Endo', 'Micro/Macro', 'Oligos_1',
    'Oligos_2', 'Oligos_3', 'Ex_1_L5_6', 'Ex_2_L5', 'Ex_3_L4_5', 'Ex_4_L_6',
    'Ex_5_L5', 'Ex_6_L4_6', 'Ex_7_L4_6', 'Ex_8_L5_6', 'Ex_9_L5_6', 'Ex_10_L2_4']

adata_st, adata_basis, dgl_graph, go_return_path_edges = preprocess(
    adata_st_list, adata_ref, celltype_ref=celltype_list_use, sample_col="group",
    slice_dist_micron=[10., 300., 10.], n_hvg_group=500,
    cache_dir="./data/DLPFC")

adata_st.obs["layer"].to_csv("./results_DLPFC/layer.csv")

target_node_type = "spot"

# In[ ]:
model = STitch3D.model.Model(
    adata_st, adata_basis, graph_encoder_name="GAT",
    dgl_graph=dgl_graph, path_edges=go_return_path_edges,
    target_node_type=target_node_type)
model.train()
save_path = "./results_DLPFC"
result = model.eval(adata_st_list_raw, save=True, output_path=save_path)

# In[ ]:
np.random.seed(1234)
gm = GaussianMixture(n_components=7, covariance_type='tied',
                     init_params='kmeans')
y = gm.fit_predict(model.adata_st.obsm['latent'], y=None)
model.adata_st.obs["GM"] = y
model.adata_st.obs["GM"].to_csv(os.path.join(save_path, "clustering_result.csv"))

# Restoring clustering labels to result
order = [2, 4, 6, 0, 3, 5, 1]  # reordering cluster labels
model.adata_st.obs["Cluster"] = [order[label]
                                 for label in model.adata_st.obs["GM"].values]
for i in range(len(result)):
    result[i].obs["GM"] = model.adata_st.obs.loc[result[i].obs_names, ]["GM"]
    result[i].obs["Cluster"] = \
        model.adata_st.obs.loc[result[i].obs_names, ]["Cluster"]
    print(result)
    print('obs', result.obs)

for i, adata_st_i in enumerate(result):
    print("Slice %d cell-type deconvolution result:" % slice_idx[i])
    sc.pl.spatial(adata_st_i, img_key="lowres",
                  color=list(adata_basis.obs.index), size=1.)

# %%
