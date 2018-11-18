"""
Models for prosestats
"""

import time
import logging

import pdb
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torchcrf import CRF

from .schema import SentenceGraph
from .torch_utils import length_mask, length_mask_2d, pack_sequence, to_one_hot
from .torch_utils import position_matrix, build_span_mask, project_on_spans, project_on_spans_2d, collapse_edges, collapse_edge_features
from . import torch_utils as tu

logger = logging.getLogger(__name__)

try:
    from allennlp.modules.augmented_lstm import AugmentedLstm
except ImportError:
    logger.error("Please install allennlp to use alstm")

def _mask(ls):
    (batch_len,), max_len = ls.size(), ls.data.max()
    ret = torch.zeros(batch_len, max_len).byte()
    for i, l in enumerate(ls.data):
        ret[i, :l] = 1
    return Variable(ret)

def _handle_crf(y_node, ls, dim):
    y_node = [to_one_hot(torch.LongTensor(ys), dim) for ys in y_node]
    y_node, ls_ = pack_sequence(y_node)
    # convert to one-hot
    assert (ls.data == ls_).all()
    return Variable(y_node)

def _check_logistic(config):
    assert config["update_L"] is False
    assert config["update_L_"] is False
    assert config["node_model"] == "none"
    assert config["edge_model"] == "none"
    return True

class Model(nn.Module):
    @classmethod
    def make_config(cls):
        return {
            # Architecture options.
            "embed_layer": "conv",
            "n_layers": 2,
            "decode_layer": "crf",
            "node_model":"simple",
            "edge_model":"simple",
            "span_agg":"sum",
            "path_agg":"max",
            # hyperparams
            "hidden_dim": 50,
            "dropout": 0.5,
            "rdropout": 0.1,
            "update_L": False,
            "update_L_": True,
            "use_gold": True, # If true, use the gold labels in training.

            # Should be set by main
            "embed_dim": 50,
            "n_features_fixed": 2,
            "n_features_learn": 4,
            "n_features_exact": 1,
            "feat_dim": None,

            # Should never be changed.
            "output_node_dim": 1 + len(SentenceGraph.NODE_LABELS),
            "output_arc_dim": 1 + len(SentenceGraph.EDGE_LABELS),
            "node_weights": [1.0 for _ in range(1 + len(SentenceGraph.NODE_LABELS))],
            "edge_weights": [1.0 for _ in range(1 + len(SentenceGraph.EDGE_LABELS))],
            }

    def __init__(self, config, L):
        super().__init__()
        # Setting up the ebemdding.
        if "vocab_dim" not in config:
            config["vocab_dim"] = L.shape[0]
        assert (config["vocab_dim"], config["embed_dim"]) == L.shape

        # 1. Set up featurization
        self.L = nn.Embedding(config["vocab_dim"], config["embed_dim"])
        self.L.weight.data = torch.from_numpy(L)
        self.L.requires_grad = config["update_L"]

        self.L_ = nn.Embedding(config["feat_dim"], config["embed_dim"])
        nn.init.normal_(self.L_.weight)
        self.L_.requires_grad = config["update_L_"]

        input_feature_dim = \
                config["n_features_fixed"] * config["embed_dim"] +\
                config["n_features_learn"] * config["embed_dim"] +\
                config["n_features_exact"]

        # 2. Embed layer
        if config["embed_layer"] == "conv":
            self.embed_C0 = nn.Conv1d(input_feature_dim, config['hidden_dim'], 3, stride=1, padding=1)
            self.embed_Cn = torch.nn.ModuleList([nn.Conv1d(config['hidden_dim'], config['hidden_dim'], 3, stride=1, padding=1) for _ in range(config["n_layers"])])
        elif config["embed_layer"] == "none":
            assert _check_logistic(config)
        elif config["embed_layer"] == "lstm":
            # (//2 because bidirectional)
            self.embed_h0 = Variable(torch.Tensor(2*config["n_layers"], config["hidden_dim"]//2))
            self.embed_c0 = Variable(torch.Tensor(2*config["n_layers"], config["hidden_dim"]//2))
            self.embed_W = nn.LSTM(input_feature_dim, config["hidden_dim"]//2,
                              num_layers=config["n_layers"],
                              dropout=config["rdropout"],
                              bidirectional=True,
                              batch_first=True,
                             )
            for param in self.embed_W.parameters():
                if len(param.size()) > 1:
                    nn.init.orthogonal_(param)
        elif config["embed_layer"] == "alstm":
            # (//2 because bidirectional)
            self.embed_h0 = Variable(torch.Tensor(2*config["n_layers"], config["hidden_dim"]//2))
            self.embed_c0 = Variable(torch.Tensor(2*config["n_layers"], config["hidden_dim"]//2))

            self.embed_Wn = torch.nn.ModuleList([AugmentedLstm(input_feature_dim, config["hidden_dim"]//2,
                              go_forward=(i % 2 == 0),
                              recurrent_dropout_probability=config["dropout"],
                              use_input_projection_bias=False,
                              bidirectional=True,
                              batch_first=True,
                             ) for i in range(2 * config["n_layers"])])
        else:
            raise ValueError("Invalid embedding layer {}".format(config["embed_layer"]))

        # 3. Node model
        if config["node_model"] == "simple":
            self.node_W = nn.Linear(config["hidden_dim"], config["hidden_dim"])
            nn.init.xavier_normal_(self.node_W.weight)
            self.node_U = nn.Linear(config["hidden_dim"], config["output_node_dim"])
            nn.init.xavier_normal_(self.node_U.weight)
        elif config["node_model"] == "none":
            assert _check_logistic(config)
            self.node_U = nn.Linear(input_feature_dim, config["output_node_dim"])
            nn.init.xavier_normal_(self.node_U.weight)
        else:
            raise ValueError("Invalid node model {}".format(config["node_model"]))


        # 4. Decode layer
        if config["decode_layer"] == "simple":
            pass
        elif config["decode_layer"] == "crf":
            self.crf = CRF(config["output_node_dim"])
            nn.init.orthogonal_(self.crf.transitions)
        else:
            raise ValueError("Invalid decode layer {}".format(config["decode_layer"]))

        # 5. Edge features
        self.edge_feature_dim = 1 + 1 # Position embeddings and path length embeddings.
        if config["path_agg"] == "max" or config["path_agg"] == "sum":
            # Include edge embeddings.
            self.edge_feature_dim += config["hidden_dim"]

        # 6. Edge model
        if config["edge_model"] == "simple":
            self.edge_W = nn.Linear(3*(config["hidden_dim"]+config["output_node_dim"]) + self.edge_feature_dim, config["hidden_dim"])
            nn.init.xavier_normal_(self.edge_W.weight)
            self.edge_U = nn.Linear(config["hidden_dim"], config["output_arc_dim"])
            nn.init.xavier_normal_(self.edge_U.weight)
        elif config["edge_model"] == "none":
            assert _check_logistic(config)
            self.edge_U = nn.Linear(3*(input_feature_dim+config["output_node_dim"]) + self.edge_feature_dim, config["output_arc_dim"])
            nn.init.xavier_normal_(self.edge_U.weight)
        else:
            raise ValueError("Invalid edge model {}".format(config["edge_model"]))

        # 7. Objectives
        self.node_objective = nn.CrossEntropyLoss(torch.FloatTensor(config["node_weights"]))
        self.edge_objective = nn.CrossEntropyLoss(torch.FloatTensor(config["edge_weights"]))

        self.config = config

    def _dropout(self, x):
        return F.dropout(x, self.config["dropout"])

    def _embed_sentence(self, x, ls):
        _, _, n_features = x.size()
        # Project onto L
        total_features = self.config["n_features_fixed"] + self.config["n_features_learn"] + self.config["n_features_exact"]
        assert n_features >= total_features
        if n_features != total_features:
            logger.warning(
                "Using only %d + %d + %d features when the data has %d",
                self.config["n_features_fixed"],
                self.config["n_features_learn"],
                self.config["n_features_exact"],
                n_features)

        # Do token dropout.
        x = self._dropout(x)
        x_ = []
        ix = 0
        x_ += [self.L(x[:,:,i]) for i in range(self.config['n_features_fixed'])]
        ix += self.config['n_features_fixed']
        x_ += [self.L_(x[:,:,i]) for i in range(ix, ix + self.config['n_features_learn'])]
        ix += self.config['n_features_learn']

        x_ = torch.cat(x_, 2)

        if self.config['n_features_exact'] > 0:
            s = slice(ix, ix+self.config["n_features_exact"])
            x = torch.cat([x_, x[:,:,s].float()], -1)
        else:
            x = x_
        return x

    def _embed_layer(self, x, ls):
        batch_len, _, _ = x.size()
        if self.config["embed_layer"] == "alstm":
            x = nn.utils.rnn.pack_padded_sequence(x, list(ls.data), batch_first=True)
            ret = []

            h0 = self.embed_h0.unsqueeze(1).repeat(1, batch_len, 1)
            c0 = self.embed_c0.unsqueeze(1).repeat(1, batch_len, 1)

            for i, W in enumerate(self.embed_Wn):
                y, _ = W(x, (h0[i], c0[i]))
                ret.append(y)

            x = torch.cat([
                nn.utils.rnn.pad_packed_sequence(ret[-2], batch_first=True)[0],
                nn.utils.rnn.pad_packed_sequence(ret[-1], batch_first=True)[0],
                ], -1)
        elif self.config["embed_layer"] == "lstm":
            x = nn.utils.rnn.pack_padded_sequence(x, list(ls.data), batch_first=True)
            h0 = self.embed_h0.unsqueeze(1).repeat(1, batch_len, 1)
            c0 = self.embed_c0.unsqueeze(1).repeat(1, batch_len, 1)
            x, _ = self.embed_W(x, (h0, c0))
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        elif self.config["embed_layer"] == "conv":
            x = F.relu(self.embed_C0(x.transpose(1,2)).transpose(1,2))
            for C in self.embed_Cn:
                x = F.relu(C(x.transpose(1,2)).transpose(1,2))
        elif self.config["embed_layer"] == "none":
            pass
        return x

    def _node_model(self, x, ls):
        if self.config["node_model"] == "simple":
            x = F.relu(self.node_W(self._dropout(x)))
            # These are logits, don't softmax
            y_node = self.node_U(self._dropout(x))
        elif self.config["node_model"] == "none":
            y_node = self.node_U(x)
        return y_node

    def _decode_layer(self, y_node, ls):
        if self.config["decode_layer"] == "simple":
            pass
        elif self.config["decode_layer"] == "crf":
            y_node = self.crf.decode(y_node.t(), mask=_mask(ls).t())
            # _handle_crf converts the sparse list representation back
            # into a one-hot version for downstream consumption.
            y_node = _handle_crf(y_node, ls, self.config["output_node_dim"])
        return y_node

    def _path_agg(self, x, q):
        _, seq_len, _ = x.size()
        if self.config["path_agg"] == "max":
            z = torch.stack([
                torch.stack([
                    self.L_(q[:, i, j, :]).max(1)[0]
                    for j in range(seq_len)], 1)
                for i in range(seq_len)], 1)
        elif self.config["path_agg"] == "sum":
            z = torch.stack([
                torch.stack([
                    self.L_(q[:, i, j, :]).sum(1)
                    for j in range(seq_len)], 1)
                for i in range(seq_len)], 1)
        else:
            z = None
        return z

    def _edge_features(self, x, y_node, q, pls):
        """
        Constructs raw edge features (that are then projected onto spans).
        """
        batch_len, seq_len, _ = x.size()
        z_size = (batch_len, seq_len, seq_len, -1)

        x_ = torch.cat([x, y_node], -1)

        # broadcast to larger dimensions and then cat.
        ## need to handle flagging here?
        position = Variable(position_matrix(seq_len).unsqueeze(0).expand(*z_size[:-1]).unsqueeze(-1))
        z = torch.cat([
            x_.unsqueeze(1).expand(*z_size) + x_.unsqueeze(2).expand(*z_size),
            x_.unsqueeze(1).expand(*z_size),
            x_.unsqueeze(2).expand(*z_size),
            position,
            pls.unsqueeze(-1).float(),
            ], 3)
        if self.config["path_agg"] in ["max", "sum"]:
            z = torch.cat([z, self._path_agg(x, q)], 3)

        return z

    def _project_on_spans(self, z, y_node, ls, y_gold=None):
        if y_gold is not None:
            span_mask, ls_ = build_span_mask(y_gold.data, ls.data)
        else:
            _, y_node_ = y_node.max(-1)
            span_mask, ls_ = build_span_mask(y_node_.data, ls.data)
        span_mask, ls_ = Variable(span_mask), Variable(ls_)

        # If ls_ is 0 then well, just return 0 vectors...
        if ls_.max().data[0] == 0:
            return None, ls_

        z_ = project_on_spans_2d(z, span_mask, self.config["span_agg"])

        return z_, ls_

    def _edge_model(self, z_):
        if self.config["edge_model"] == "simple":
            z_ = self.edge_U(self._dropout(F.relu(self.edge_W(self._dropout(z_)))))
        if self.config["edge_model"] == "none":
            z_ = self.edge_U(z_)
        return z_

    def forward(self, x, q, ls, pls, y_gold=None):
        """
        Predicts on input graph @x, and lengths @ls.
        @pls are path lenghts.
        """
        batch_len, _, _ = x.size()

        # 1. Featurize sentence
        x = self._embed_sentence(x, ls)
        # 2. Embed layer
        x = self._embed_layer(x, ls)

        # 3. Node model
        y_node = self._node_model(x, ls)
        # 4. Decode layer
        y_node = self._decode_layer(y_node, ls)

        # 5. Edge features
        z = self._edge_features(x, y_node, q, pls)
        z_, ls_ = self._project_on_spans(z, y_node, ls, y_gold=y_gold)

        # 6. Edge model
        if z_ is not None:
            y_arc = self._edge_model(z_)
        else:
            y_arc = torch.zeros(batch_len)

        return y_node, y_arc, ls_

    def loss(self, xs, ys, qs, zs, ls, ls_, pls, use_cuda=False):
        batch_size = xs.size()[0]
        yhs, zhs, _ = self.forward(xs, qs, ls, pls, ys)

        if self.config["decode_layer"] == "simple":
            ys_ = Variable(length_mask(ys.data,  ls.data, use_cuda).view(-1))
            yhs_ = length_mask(yhs, ls.data, use_cuda).view(-1, self.config["output_node_dim"])
            loss = self.node_objective(yhs_, ys_)
        elif self.config["decode_layer"] == "crf":
            # 1. Featurize sentence
            xs = self._embed_sentence(xs, ls)
            # 2. Embed layer
            xs = self._embed_layer(xs, ls)
            # 3. Node model
            y_node = self._node_model(xs, ls)

            # convert y_node and tags into a format that CRF will like.
            emissions = y_node.transpose(0,1).contiguous()
            tags = ys.t().contiguous()
            mask = _mask(ls).t().contiguous()

            # Convert log likelihood into negative log likelihood
            loss = -self.crf(emissions, tags, mask)/batch_size

        zs_ = Variable(length_mask_2d(zs.data,  ls_.data, use_cuda).view(-1))
        zhs_ = length_mask_2d(zhs, ls_.data, use_cuda).view(-1, self.config["output_arc_dim"])
        loss += self.edge_objective(zhs_, zs_)

        return loss
