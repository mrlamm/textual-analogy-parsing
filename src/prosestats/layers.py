"""
Specific layer modules for models
"""

import pdb
import numpy as np
import torch
from torchcrf import CRF
from torch.autograd import Variable
from torch import nn
import torch
from .torch_utils import to_scalar, log_sum_exp, NINF, to_one_hot_logits

def constv(value, size):
    return torch.ones(*size) * value

def _pad_tags(tags, lengths):
    """
    Pads tags to make sure that they all start with START and end with END.
    """
    batch_len, _ = tags.size()

    tags = torch.cat([torch.LongTensor([self.START] * batch_len), tags, torch.LongTensor([self.END] * batch_len)])
    tags[list(range(batch_len)), lengths] = self.END
    return tags

class GlobalLinearCRFLayer(nn.Module):
    """
    Creates a globally normalized linear-CRF model.
    Assumes input as a sequence of features (x_t).
    """

    @classmethod
    def make_config(cls):
        return {
            "tagset_size": 5,
            "hidden_dim": 100,
        }

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.START = config["tagset_size"]
        self.END = config["tagset_size"] + 1
        self.tagset_size = config["tagset_size"] + 2
        #self._tagset_size = config["tagset_size"] + 2

        self.unary_potentials = nn.Linear(config["hidden_dim"], self.tagset_size) # for START and END + 2
        # Transitioning to (i) from (j)
        self.binary_potentials = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size)) # for START and END
        # Nothing goes to START
        self.binary_potentials.data[self.START, :] = NINF
        # End goes nowhere
        self.binary_potentials.data[:, self.END] = NINF

    def _score(self, feats, tags, lengths):
        """
        Returns an unnormalized p(y|x).
        """
        batch_len, seq_len, _ = feats.size()

        batchv = list(range(batch_len))

        tags = torch.cat([constv(self.START, (batch_len,1)).long(), tags.data], 1)

        score = Variable(torch.zeros(batch_len))
        for i, feat in enumerate(feats.transpose(0,1)):
            unary = ((i  < lengths).float() * feat[batchv, tags[:,i+1]])
            binary = ((i  < lengths).float() * self.binary_potentials[tags[:,i+1], tags[:,i]] +\
                      (i == lengths).float() * self.binary_potentials[[self.END] * batch_len, tags[:,i]])
            score += unary + binary
        binary = ((seq_len == lengths).float() * self.binary_potentials[[self.END] * batch_len, tags[:,-1]])
        score += binary

        return score

    def _partition_function(self, feats, lengths):
        """
        Returns the normalization constant  Z(x).
        Computes forward scores (in log space):
            alpha initialized to -inf
            alpha(0) = 0
            alpha(l) = phi(l) + max_l' phi(l, l') * alpha_(l')
        """
        batch_len, seq_len, _ = feats.size()

        alphas = torch.Tensor(batch_len, self.tagset_size).fill_(NINF)
        alphas[:,self.START] = 0. # Start with all mass on 0.
        alphas = Variable(alphas)

        # Expand for all the batches.
        T = self.binary_potentials.unsqueeze(0).expand(batch_len, self.tagset_size, self.tagset_size)
        for i, feat in enumerate(feats.transpose(0,1)):
            # Make a copy along one dimension so we can add by matrix multiply.
            alphas_ = alphas.unsqueeze(1).expand(batch_len, self.tagset_size, self.tagset_size)
            # Note the log_sum_exp in the 2nd case is deferred till the end.
            alphas = ((i  < lengths).float() * (feat + log_sum_exp(T + alphas_, 2)).t() +\
                      (i == lengths).float() * (alphas + T[:, self.END]).t() +\
                      (i  > lengths).float() * (alphas).t()).t()
        # to terminal
        alphas = ((seq_len == lengths).float() * (alphas + T[:, self.END]).t() +\
                  (seq_len  > lengths).float() * (alphas).t()).t()
        return log_sum_exp(alphas,1)

    def _viterbi_decode(self, feats, lengths):
        """
        Returns the max probability sequence.
        """
        batch_len, seq_len, _ = feats.size()

        alphas = torch.Tensor(batch_len, self.tagset_size).fill_(NINF)
        alphas[:, self.START] = 0.
        alphas = Variable(alphas)
        backpointers = []

        # Expand for all batches
        T = self.binary_potentials.unsqueeze(0).expand(batch_len, self.tagset_size, self.tagset_size)
        I = Variable(torch.LongTensor(list(range(self.tagset_size))).unsqueeze(0).expand(batch_len, self.tagset_size))
        for i, feat in enumerate(feats.transpose(0,1)):
            alphas_ = alphas.unsqueeze(1).expand(batch_len, self.tagset_size, self.tagset_size)
            alphas_, bkptrs = torch.max(T + alphas_, 2)
            # Note the log_sum_exp in the 2nd case is deferred till the end.
            alphas = ((i  < lengths).float() * (alphas_ + feat).t() +\
                      (i == lengths).float() * (alphas + T[:, self.END]).t() +\
                      (i  > lengths).float() * (alphas).t()).t()

            # Backpointer
            bkptrs = ((i  < lengths).long() * bkptrs.t() +\
                      (i >= lengths).long() * I.t()).t()
            backpointers.append(bkptrs.data.numpy())

        # to terminal
        alphas = ((seq_len == lengths).float() * (alphas + T[:, self.END]).t() +\
                  (seq_len  > lengths).float() * (alphas).t()).t()
        scores, best_tags = torch.max(alphas, 1)

        best_tags = to_scalar(best_tags)
        best_paths = [best_tags]

        for bptrs in reversed(backpointers):
            best_tags = [bptrs_i[best_tag_i] for best_tag_i, bptrs_i in zip(best_tags, bptrs)]
            best_paths.append(best_tags)
        best_paths.pop(-1)
        best_paths.reverse()
        best_paths = torch.from_numpy(np.array(best_paths).T)

        return scores, best_paths

    def forward(self, sentence_feats, lengths):
        """
        Assumes input features of size sentence_feats x sentence_feats.
        """
        feats = self.unary_potentials(sentence_feats)
        #p_tags = self._compute_token_probs(feats, lengths)
        _, tag_seq = self._viterbi_decode(feats, lengths)
        return Variable(to_one_hot_logits(tag_seq, self.config["tagset_size"]))

    def loss(self, sentence_feats, tags, lengths):
        """
        Returns a CRF-loss function that is the negative log likelihood.
        """
        feats = self.unary_potentials(sentence_feats)

        Z = self._partition_function(feats, lengths)
        p_yx = self._score(feats, tags, lengths)

        return Z - p_yx

def test_crf_layer_batch():
    config = GlobalLinearCRFLayer.make_config()
    config["tagset_size"] = 4
    config["hidden_dim"] = 10
    layer = GlobalLinearCRFLayer(config)

    xs, ys = Variable(torch.randn(4,6,10)), Variable(torch.LongTensor([
        [0, 1, 0, 2, 3, 0],
        [0, 1, 0, 2, 3, 0],
        [0, 1, 0, 2, 3, 0],
        [0, 1, 0, 2, 3, 0],
        ]))
    ls = Variable(torch.LongTensor([6, 6, 6, 6]))

    l = layer.loss(xs, ys, ls)
    ys_ = layer(xs, ls)
    l.sum().backward()

    assert (l.data.numpy() > 0.).all()
    #assert (scores.data.numpy() > 0.).all()
    #assert ys_.size() == ys.size()
    #assert ys_.max() < config["tagset_size"]

    ls = Variable(torch.LongTensor([6, 4, 3, 5]))
    ys_ = layer(xs, ls)
    l = layer.loss(xs, ys, ls)

    assert (l.data.numpy() > 0.).all()
    #assert (scores.data.numpy() > 0.).all()
    #assert ys_.size() == ys.size()
    #assert (ys_.max() < config["tagset_size"]


def test_crf():
    torch.manual_seed(43)
    seq_length, batch_size, num_tags = 3, 2, 5
    emissions = Variable(torch.randn(seq_length, batch_size, num_tags), requires_grad=True)
    tags = Variable(torch.LongTensor([[0, 1], [2, 4], [3, 1]]))  # (seq_length, batch_size)
    mask = Variable(torch.ByteTensor([[1, 1], [1, 1], [1, 0]]))

    model = CRF(num_tags)
    for p in model.parameters():
        nn.init.uniform(p, -1, 1)

    loss = model(emissions, tags)
    preds = model.decode(emissions)

    assert np.allclose(to_scalar(loss), -11.229, 1e-3)
    assert preds == [[2,4,3], [4,3,1]]

    loss = model(emissions, tags, mask=mask)
    preds = model.decode(emissions, mask=mask)

    assert np.allclose(to_scalar(loss), -10.073, 1e-3)
    assert preds == [[2,4,3], [4,3]]

    loss.backward()
    #for p in model.parameters():
    #    print(p.grad)
