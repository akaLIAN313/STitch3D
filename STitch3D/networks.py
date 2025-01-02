import dgl.function as fn
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Literal
from STitch3D.utils import cluster_and_compuse_kl


class DenseLayer(nn.Module):

    def __init__(self,
                 c_in,  # dimensionality of input features
                 c_out,  # dimensionality of output features
                 zero_init=False,  # initialize weights as zeros; use Xavier uniform init if zero_init=False
                 ):

        super().__init__()

        self.linear = nn.Linear(c_in, c_out)

        # Initialization
        if zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(
                self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)),
                np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)

    def forward(self,
                node_feats,  # input node features
                ):

        node_feats = self.linear(node_feats)

        return node_feats


class GATSingleHead(nn.Module):

    def __init__(self,
                 c_in,  # dimensionality of input features
                 c_out,  # dimensionality of output features
                 temp=1,  # temperature parameter
                 ):

        super().__init__()

        self.linear = nn.Linear(c_in, c_out)
        self.v0 = nn.Parameter(torch.Tensor(c_out, 1))
        self.v1 = nn.Parameter(torch.Tensor(c_out, 1))
        self.temp = temp

        # Initialization
        nn.init.uniform_(
            self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)),
            np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)
        nn.init.uniform_(
            self.v0.data, -np.sqrt(6 / (c_out + 1)),
            np.sqrt(6 / (c_out + 1)))
        nn.init.uniform_(
            self.v1.data, -np.sqrt(6 / (c_out + 1)),
            np.sqrt(6 / (c_out + 1)))

    def forward(self,
                node_feats,  # input node features
                adj_matrix,  # adjacency matrix including self-connections
                ):

        # Apply linear layer and sort nodes by head
        node_feats = self.linear(node_feats)
        f1 = torch.matmul(node_feats, self.v0)
        f2 = torch.matmul(node_feats, self.v1)
        attn_logits = adj_matrix * (f1 + f2.T)
        unnormalized_attentions = (F.sigmoid(attn_logits) - 0.5).to_sparse()
        attn_probs = torch.sparse.softmax(
            unnormalized_attentions / self.temp, dim=1)
        attn_probs = attn_probs.to_dense()
        node_feats = torch.matmul(attn_probs, node_feats)

        return node_feats


class GATMultiHead(nn.Module):

    def __init__(self,
                 c_in,  # dimensionality of input features
                 c_out,  # dimensionality of output features
                 n_heads=1,  # number of attention heads
                 concat_heads=True,  # concatenate attention heads or not
                 ):

        super().__init__()

        self.n_heads = n_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % n_heads == 0, "The number of output features should be divisible by the number of heads."
            c_out = c_out // n_heads

        self.block = nn.ModuleList()
        for i_block in range(self.n_heads):
            self.block.append(GATSingleHead(c_in=c_in, c_out=c_out))

    def forward(self, param_list
                # node_feats,  # input node features
                # adj_matrix,  # adjacency matrix including self-connections
                ):
        node_feats, adj_matrix = param_list[0], param_list[1]

        res = []
        for i_block in range(self.n_heads):
            res.append(self.block[i_block](node_feats, adj_matrix))

        if self.concat_heads:
            node_feats = torch.cat(res, dim=1)
        else:
            node_feats = torch.mean(torch.stack(res, dim=0), dim=0)

        return node_feats


class DeconvNet(nn.Module):
    def __init__(self,
                 hidden_dims,  # dimensionality of hidden layers
                 n_celltypes,  # number of cell types
                 n_spot_clusters,  # number of spot clusters
                 n_slices,  # number of slices
                 graph_encoder_name: Literal['GAT', 'EMP'] = 'EMP',  # graph encoder
                 G: dgl.DGLHeteroGraph = None, # 
                 path_edges=None,  # path edges
                 target_node_type=None,  # target node type
                 target_node_emb=None,  # target node embedding
                 n_heads=None,  # number of attention heads
                 slice_emb_dim=None,  # dimensionality of slice id embedding
                 coef_fe=None,
                 device='cpu'
                 ):
        super().__init__()
        self.n_spot_clusters = n_spot_clusters
        # define layers
        # autoencoder
        self.graph_encoder_name = graph_encoder_name
        if graph_encoder_name == 'GAT':
            print("encoder: GAT")
            self.encoder = nn.Sequential(
                GATMultiHead(
                    hidden_dims[0], hidden_dims[1],
                    n_heads=n_heads, concat_heads=True
                ),
                nn.ReLU(),
                DenseLayer(hidden_dims[1], hidden_dims[2])
            )
            self.decoder = nn.Sequential(
                GATMultiHead(
                    hidden_dims[2]+slice_emb_dim, hidden_dims[1],
                    n_heads=n_heads, concat_heads=True
                ),
                nn.ReLU(),
                DenseLayer(hidden_dims[1], hidden_dims[0])
            )
        elif graph_encoder_name == 'EMP':
            print("encoder: EMP")
            self.encoder = HGIN(
                G, path_edges,  
                target_node_emb, target_node_type, 
                hidden_dims[0], hidden_dims[1], hidden_dims[2],
                num_layers=1, num_mlp_layers=2,
                dropout=0.1, device=device
            )
            self.decoder = HGIN(
                G, path_edges,  
                target_node_emb, target_node_type, 
                hidden_dims[2], hidden_dims[1], hidden_dims[0],
                num_layers=1, num_mlp_layers=2,
                dropout=0.1, device=device
            )

        # deconvolution layers
        self.deconv_alpha_layer = DenseLayer(
            hidden_dims[2] + slice_emb_dim, 1, zero_init=True)
        self.deconv_beta_layer = DenseLayer(
            hidden_dims[2], n_celltypes, zero_init=True)

        self.gamma = nn.Parameter(torch.Tensor(n_slices, 4558).zero_())

        self.slice_emb = nn.Embedding(n_slices, slice_emb_dim)

        self.coef_fe = coef_fe

    def forward(self,
            G: dgl.DGLHeteroGraph=None,
            adj_matrix=None,  # adjacency matrix including self-connections
            node_feats=None,  # input node features
            count_matrix=None,  # gene expression counts
            library_size=None,  # library size (based on Y)
            slice_label=None,  # slice label
            basis=None,  # basis matrix
        ):
        slice_label_emb = self.slice_emb(slice_label)
        # autoencoder
        if self.graph_encoder_name == 'GAT':
            Z = self.encoder([node_feats, adj_matrix])
            node_feats_recon = self.decoder(
                [torch.cat((Z, slice_label_emb), axis=1), adj_matrix])
        elif self.graph_encoder_name == 'EMP':
            Z = self.encoder(G, node_feats)
            node_feats_recon = self.decoder(G, Z)
        else:
            raise ValueError(
                'Unknown graph encoder name:',
                self.encoder.__class__.__name__)

        # deconvolutioner
        

        beta, alpha = self.deconvolutioner(Z, slice_label_emb)

        # KL loss

        # kl_loss = cluster_and_compuse_kl(Z, self.n_spot_clusters)

        # reconstruction loss of node features
        self.features_loss = torch.mean(
            torch.sqrt(
                torch.sum(
                    torch.pow(node_feats - node_feats_recon, 2),
                    axis=1)))

        # deconvolution loss
        log_lam = torch.log(torch.matmul(
            beta, basis) + 1e-6) + alpha + self.gamma[slice_label]
        lam = torch.exp(log_lam)
        self.decon_loss = -torch.mean(
            torch.sum(
                count_matrix * (torch.log(library_size + 1e-6) + log_lam) -
                library_size * lam, axis=1))

        # Total loss
        loss = self.decon_loss + self.coef_fe * self.features_loss \
            # + self.coef_fe*kl_loss

        return loss

    def evaluate(self, adj_matrix, node_feats, slice_label,
                 G:dgl.DGLHeteroGraph=None):
        slice_label_emb = self.slice_emb(slice_label)
        # encoder
        if self.graph_encoder_name == 'GAT':
            Z = self.encoder(adj_matrix, node_feats)
        elif self.graph_encoder_name == 'EMP':
            Z = self.encoder(G, node_feats)

        # deconvolutioner
        beta, alpha = self.deconvolutioner(Z, slice_label_emb)

        return Z, beta, alpha, self.gamma

    def deconvolutioner(self, Z, slice_label_emb):
        beta = self.deconv_beta_layer(F.elu(Z))
        beta = F.softmax(beta, dim=1)
        H = F.elu(torch.cat((Z, slice_label_emb), axis=1))
        alpha = self.deconv_alpha_layer(H)
        return beta, alpha


class DeconvNet_NB(nn.Module):

    def __init__(self,
                 hidden_dims,  # dimensionality of hidden layers
                 n_celltypes,  # number of cell types
                 n_slices,  # number of slices
                 n_heads,  # number of attention heads
                 slice_emb_dim,  # dimensionality of slice id embedding
                 coef_fe,
                 ):

        super().__init__()

        # define layers
        # encoder layers
        self.encoder_layer1 = GATMultiHead(
            hidden_dims[0],
            hidden_dims[1],
            n_heads=n_heads, concat_heads=True)
        self.encoder_layer2 = DenseLayer(hidden_dims[1], hidden_dims[2])
        # decoder layers
        self.decoder_layer1 = GATMultiHead(
            hidden_dims[2] + slice_emb_dim, hidden_dims[1],
            n_heads=n_heads, concat_heads=True)
        self.decoder_layer2 = DenseLayer(hidden_dims[1], hidden_dims[0])
        # deconvolution layers
        self.deconv_alpha_layer = DenseLayer(
            hidden_dims[2] + slice_emb_dim, 1, zero_init=True)
        self.deconv_beta_layer = DenseLayer(
            hidden_dims[2], n_celltypes, zero_init=True)

        self.gamma = nn.Parameter(torch.Tensor(n_slices, hidden_dims[0]).zero_())
        self.logtheta = nn.Parameter(5. * torch.ones(n_slices, hidden_dims[0]))

        self.slice_emb = nn.Embedding(n_slices, slice_emb_dim)

        self.coef_fe = coef_fe

    def forward(self,
                adj_matrix,  # adjacency matrix including self-connections
                node_feats,  # input node features
                count_matrix,  # gene expression counts
                library_size,  # library size (based on Y)
                slice_label,  # slice label
                basis,  # basis matrix
                ):
        # encoder
        Z = self.encoder(adj_matrix, node_feats)

        # deconvolutioner
        slice_label_emb = self.slice_emb(slice_label)
        beta, alpha = self.deconvolutioner(Z, slice_label_emb)

        # decoder
        node_feats_recon = self.decoder(adj_matrix, Z, slice_label_emb)

        # reconstruction loss of node features
        self.features_loss = torch.mean(
            torch.sqrt(
                torch.sum(
                    torch.pow(node_feats - node_feats_recon, 2),
                    axis=1)))

        # deconvolution loss
        log_lam = torch.log(torch.matmul(
            beta, basis) + 1e-6) + alpha + self.gamma[slice_label]
        lam = torch.exp(log_lam)
        theta = torch.exp(self.logtheta)
        self.decon_loss = - torch.mean(torch.sum(torch.lgamma(count_matrix + theta[slice_label] + 1e-6) -
                                                 torch.lgamma(theta[slice_label] + 1e-6) +
                                                 theta[slice_label] * torch.log(theta[slice_label] + 1e-6) -
                                                 theta[slice_label] * torch.log(theta[slice_label] + library_size * lam + 1e-6) +
                                                 count_matrix * torch.log(library_size * lam + 1e-6) -
                                                 count_matrix * torch.log(theta[slice_label] + library_size * lam + 1e-6), axis=1))

        # Total loss
        loss = self.decon_loss + self.coef_fe * self.features_loss

        return loss

    def evaluate(self, adj_matrix, node_feats, slice_label):
        slice_label_emb = self.slice_emb(slice_label)
        # encoder
        Z = self.encoder(adj_matrix, node_feats)

        # deconvolutioner
        beta, alpha = self.deconvolutioner(Z, slice_label_emb)

        return Z, beta, alpha, self.gamma

    def encoder(self, adj_matrix, node_feats):
        H = node_feats
        H = F.elu(self.encoder_layer1(H, adj_matrix))
        Z = self.encoder_layer2(H)
        return Z

    def decoder(self, adj_matrix, Z, slice_label_emb):
        H = torch.cat((Z, slice_label_emb), axis=1)
        H = F.elu(self.decoder_layer1(H, adj_matrix))
        X_recon = self.decoder_layer2(H)
        return X_recon

    def deconvolutioner(self, Z, slice_label_emb):
        beta = self.deconv_beta_layer(F.elu(Z))
        beta = F.softmax(beta, dim=1)
        H = F.elu(torch.cat((Z, slice_label_emb), axis=1))
        alpha = self.deconv_alpha_layer(H)
        return beta, alpha


class MLP(nn.Module):
    def __init__(self, num_layers, in_size, hidden_size, out_size):
        """
            num_layers: number of layers in MLP, if num_layers=1, then MLP is a Linear Model.
            in_size: dimensionality of input features
            hidden_size: dimensionality of hidden units at ALL layers
            out_size: number of classes for prediction
        """
        super(MLP, self).__init__()

        self.if_linear = True  # default is True, which return a linear model.
        self.num_layers = num_layers

        if self.num_layers == 1:
            # a linear model
            self.Linear = nn.Linear(in_size, out_size)
        else:
            # MLP
            self.if_linear = False
            self.Linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.Linears.append(nn.Linear(in_size, hidden_size))
            for layer in range(self.num_layers-2):
                self.Linears.append(nn.Linear(hidden_size, hidden_size))
            self.Linears.append(nn.Linear(hidden_size, out_size))

            # Batch Norms
            for layer in range(self.num_layers-1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_size)))

    def forward(self, inputs):
        if self.if_linear:
            return self.Linear(inputs)
        else:
            h = inputs
            for layer in range(self.num_layers-1):
                h = F.relu(self.batch_norms[layer](self.Linears[layer](h)))
            return self.Linears[-1](h)


class HGINLayer(nn.Module):
    def __init__(self, paths, in_size, out_size, num_mlp_layers,
                 target_node_type, device):
        super(HGINLayer, self).__init__()
        self.paths = list(tuple(path) for path in paths)
        self.go_path = list(path[0] for path in self.paths)
        self.return_path = list(path[1] for path in self.paths)
        self.num_mlp_layers = num_mlp_layers
        self.target_node_type = target_node_type
        self.device = device
        # construct parameters.

        # create a project layer for target-node-type
        self.project = nn.Linear(in_size, out_size)
        # create MLP for each meta-path's Go-Relation.
        self.MLP = nn.ModuleDict({
            name: MLP(num_mlp_layers, in_size, out_size, out_size).to(self.device)
            for name in self.go_path
        })
        # one-more MLP for Return-Message-Passing.
        self.MLP['Return'] = MLP(
            num_mlp_layers, out_size, out_size, out_size).to(self.device)
        # eps
        eps_dict = {name: nn.Parameter(torch.zeros(1))
                    for name in self.go_path}
        eps_dict['Return'] = nn.Parameter(torch.zeros(1))
        self.eps = nn.ParameterDict(eps_dict)

    def forward(self, G: dgl.DGLHeteroGraph, feat_dict: dict):
        """
            G: is a Heterogenous Graph in DGL.
            feat_dict: is a dictionary of node features for each type.
        """
        with G.local_scope():
            # set features for all nodes.
            for ntype in G.ntypes:
                G.nodes[ntype].data['h'] = feat_dict[ntype].to(G.device)

            # GNN for GOPATH.
            for path in self.go_path:
                srctype, etype, dsttype = G.to_canonical_etype(path)
                G.send_and_recv(G[etype].edges(), fn.copy_u('h', 'm'),
                                fn.sum('m', 'a'), etype=etype)
                G = G.to(self.device)
                G.apply_nodes(
                    lambda nodes: {'h': self.MLP[etype](
                    (1+self.eps[etype])*nodes.data['h'] + nodes.data['a'])}, 
                    ntype=dsttype)

            # GNN for RETUANPATH
            funcs = {}
            for path in self.return_path:
                srctype, etype, dsttype = G.to_canonical_etype(path)
                funcs[etype] = (fn.copy_u('h', 'm'), fn.sum('m', 'a'))
                # _G = dgl.edge_subgraph(G, )
            # for path in self.return_path:
                # _G = dgl.edge_subgraph(G,)
                # G.multi_send_and_recv(funcs, "sum")
                # srctype, etype, dsttype = G.to_canonical_etype(path)
            G.multi_update_all(funcs, "sum")
            G.apply_nodes(lambda nodes: {'h': self.project(
                nodes.data['h'])}, ntype=self.target_node_type)
            G.apply_nodes(
                lambda nodes: {'h': self.MLP['Return'](
                (1+self.eps['Return'])*nodes.data['h'] + nodes.data['a'])},
                ntype=self.target_node_type)
            return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HGIN(nn.Module):
    def __init__(
            self, G, paths, features, target_node_type, in_size, hidden_size,
            out_size, num_layers, num_mlp_layers, dropout, device):
        """
            features: input features, which is a dictionary of node features for each type
        """
        super(HGIN, self).__init__()
        self.device = device
        self.dropout = dropout
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {}
        self.target_node_type = target_node_type
        for ntype in G.ntypes:
            if ntype == target_node_type:
                embed_dict[ntype] = features
            else:
                # embed_dict[ntype] = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size)).to(self.device)
                embed_dict[ntype] = torch.Tensor(
                    G.number_of_nodes(ntype), in_size).to(self.device)
                # initialization ramdomly
                nn.init.xavier_uniform_(embed_dict[ntype])

        self.embed = embed_dict

        # Create layers.
        self.layers = nn.ModuleList()
        self.layers.append(
            HGINLayer(
                paths, in_size, hidden_size, num_mlp_layers,
                target_node_type, device).to(self.device))
        for i in range(1, num_layers):
            self.layers.append(HGINLayer(paths, hidden_size,
                               hidden_size, num_mlp_layers, target_node_type))

        self.predict = nn.Linear(hidden_size, out_size)

    def forward(self, G, target_node_emb):
        self.embed[self.target_node_type] = target_node_emb
        h_dict = {ntype: F.dropout(embed, self.dropout, training=self.training)
                  for ntype, embed in self.embed.items()}
        for gnn in self.layers:
            h_dict = gnn(G, h_dict)
            h_dict = {
                ntype: F.leaky_relu(
                    F.dropout(embed, self.dropout, training=self.training))
                for ntype, embed in h_dict.items()}
            # h_dict = {ntype: F.leaky_relu(embed) for ntype, embed in h_dict.items()}
        embeds = h_dict[self.target_node_type]

        # return embeds, self.predict(h_dict[self.target_node_type])
        return self.predict(h_dict[self.target_node_type])

