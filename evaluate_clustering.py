import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from STitch3D.utils import evaluate, cluster_and_evaluate
given_clustering_result = False
# Read the CSV files
layer = pd.read_csv('./results_DLPFC/layer.csv', index_col=0)
if given_clustering_result:
    clustering_result = pd.read_csv(
        './results_DLPFC/clustering_result.csv', index_col=0)
    

    # Extract the suffix from the index
    clustering_result['slice'] = clustering_result.index.str.extract(
        r'-slice(\d+)$').values
    layer['slice'] = layer.index.str.extract(r'-slice(\d+)$').values
    y_pred = clustering_result['GM']
else:
    representation = pd.read_csv(
        './results_DLPFC/representation.csv').iloc[:, 1:].values
    y_true = layer['layer']
    # Map y_true to numerical labels
    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(y_true)
    (acc, nmi, pur, ari, f1), _ = \
        cluster_and_evaluate(representation, y_true, n_clusters=7)
    print(
          f'Accuracy: {acc:.4f}, NMI: {nmi:.4f}, '
          f'Purity: {pur:.4f}, ARI: {ari:.4f}, F1 Score: {f1:.4f}')
    
    exit(0)
    
    
y_true = layer['layer']
# Map y_true to numerical labels
label_encoder = LabelEncoder()
y_true = label_encoder.fit_transform(y_true)


# Group by the suffix and evaluate each group
results = []
for slice_id in clustering_result['slice'].unique():
    y_pred_slice = y_pred[clustering_result['slice'] == slice_id].values
    y_true_slice = y_true[layer['slice'] == slice_id]

    # Ensure the lengths match
    assert len(y_pred_slice) == len(y_true_slice)

    # Evaluate the clustering result for this slice
    acc, nmi, pur, ari, f1 = evaluate(
        y_true_slice, y_pred_slice, show_details=False)
    results.append((slice_id, acc, nmi, pur, ari, f1))

# Print the results for each slice
for slice_id, acc, nmi, pur, ari, f1 in results:
    print(f'Slice {slice_id}: '
          f'Accuracy: {acc:.4f}, NMI: {nmi:.4f}, '
          f'Purity: {pur:.4f}, ARI: {ari:.4f}, F1 Score: {f1:.4f}')
