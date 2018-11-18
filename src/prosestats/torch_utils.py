"""
Useful routines for torch.
"""

import pdb
import logging
from tqdm import tqdm, trange
from itertools import product

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm

from .schema import SentenceGraph
from .evaluation import Scorer
from .util import get_chunks, invert_index

logger = logging.getLogger(__name__)

INF = 1e5
NINF = -1 * INF

def get_span_lengths(ys, ls):
    return [len(get_chunks(y[:l])) for y, l in zip(ys, ls)]

def build_span_mask(ys, ls):
    """
    Converts a set of predictions: [batch_len x seq_len] into a set of slices
    [batch_len x slices], where each row has 1s where mask is true.
    Decisions are based on contiguity of y.
    """
    #TODO: potentially return a smaller matrix that depends on span length.
    batch_len, seq_len = ys.size()
    ls_ = torch.LongTensor(get_span_lengths(ys, ls))
    seq_len_ = ls_.max() # max length
    ret = torch.zeros(batch_len, seq_len_, seq_len).byte()

    for batch_ix, (y, l) in enumerate(zip(ys,ls)):
        chunks = get_chunks(y[:l])
        for ix, ((start, end), _) in enumerate(chunks):
            ret[batch_ix, ix, start:end] = 1
        ls_[batch_ix] = len(chunks)
    return ret, ls_

def test_build_span_mask():
    y = torch.LongTensor([
        [0,1,1,0,1,2,0],
        [1,0,1,0,1,0,2]
        ])
    z = torch.ByteTensor([[
        [0,1,1,0,0,0,0],
        [0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0],
        ],[
        [1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1],
        ]
        ])

    z_, ls = build_span_mask(y, [7, 7])

    assert (z_ == z).all()
    assert ls[0] == 3

# TODO: Can make atleast sum more efficient by doing bmm. 
def project_on_spans(x, span_mask, method="sum"):
    """
    Projects x which is [batch_len, seq_len, features] to [batch_len,
    seq_len_, features] by collapsing spans that are appropriately masked.
    """
    batch_len, seq_len, dim = x.size()
    _, seq_len_, _ = span_mask.size()
    assert span_mask.size() == (batch_len, seq_len_, seq_len)

    if method == "sum":
        return torch.matmul(span_mask.float(), x)
    elif method == "max":
        x = x.unsqueeze(1).expand(batch_len, seq_len_, seq_len, dim)
        span_mask = span_mask.unsqueeze(-1).expand(batch_len, seq_len_, seq_len, dim).float()

        return (x * span_mask).max(2)[0]
    else:
        raise ValueError("Invalid method: {}".format(method))

def project_on_spans_2d(z, span_mask, method="sum"):
    """
    Projects x which is [batch_len, seq_len, features] to [batch_len,
    seq_len_, features] by collapsing spans that are appropriately masked.
    First go to batch x seq_ x seq
    """
    batch_len, seq_len, seq_len, dim = z.size()
    _, seq_len_, _ = span_mask.size()
    assert span_mask.size() == (batch_len, seq_len_, seq_len)

    if method == "sum":
        z = torch.matmul(span_mask.float().unsqueeze(1), z)
        z = torch.matmul(span_mask.float().unsqueeze(1),z.transpose(1,2)).transpose(1,2)
    elif method == "max":
        z = z.unsqueeze(1).expand(batch_len, seq_len_, seq_len, seq_len, dim)
        span_mask_ = span_mask.unsqueeze(3).unsqueeze(-1).expand(batch_len, seq_len_, seq_len, seq_len, dim).float()

        z = (z * span_mask_).max(2)[0]
        assert z.size() == (batch_len, seq_len_, seq_len, dim)

        z = z.unsqueeze(3).expand(batch_len, seq_len_, seq_len, seq_len_, dim)
        span_mask_ = span_mask.transpose(1,2).unsqueeze(1).unsqueeze(-1).expand(batch_len, seq_len_, seq_len, seq_len_, dim).float()

        z = (z * span_mask_).max(2)[0]
    else:
        raise ValueError("Invalid method: {}".format(method))

    return z


def test_project_on_spans():
    span_mask = torch.FloatTensor([[
        [0,1,1,0,0,0,0],
        [0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0],
        ]])

    x = torch.arange(1, 8).unsqueeze(0).unsqueeze(-1)
    y = torch.FloatTensor([[
        2+3,
        5,
        6,
        ]])
    y_ = project_on_spans(x, span_mask)
    assert (y.unsqueeze(-1) == y_).all()

    y = torch.FloatTensor([[
        3,
        5,
        6,
        ]])
    y_ = project_on_spans(x, span_mask, "max")
    assert (y.unsqueeze(-1) == y_).all()

def test_project_on_spans_2d():
    span_mask = torch.FloatTensor([[
        [0,1,1,0,0,0,0],
        [0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0],
        ]])

    x = torch.arange(1, 8).unsqueeze(0).unsqueeze(0).expand(1, 7, 7).unsqueeze(-1)
    y = torch.FloatTensor([
        [10, 10, 12],
        [5, 5, 6],
        [5, 5, 6],
        ])
    y_ = project_on_spans_2d(x, span_mask)
    assert (y.unsqueeze(-1) == y_).all()

    y = torch.FloatTensor([
        [3, 5, 6],
        [3, 5, 6],
        [3, 5, 6],
        ])
    y_ = project_on_spans_2d(x, span_mask, "max")
    assert (y.unsqueeze(-1) == y_).all()

def collapse_edges(ys, zs, ls):
    """
    Uses the sequence ys to collapse zs.
    """
    batch_len, seq_len = ys.size()
    assert zs.size() == (batch_len, seq_len, seq_len)

    ls_ = torch.zeros(batch_len).long()
    ret = torch.zeros(batch_len, seq_len, seq_len).long()
    for batch_ix, (y, l) in enumerate(zip(ys,ls)):
        chunks = get_chunks(y[:l])
        for i, ((start, _), _) in enumerate(chunks):
            for j, ((start_, _), _) in enumerate(chunks):
                ret[batch_ix, i, j] = zs[batch_ix, start, start_]
        ls_[batch_ix] = len(chunks)
    return ret, ls_


def collapse_edge_features(ys, qs, ls):
    """
    Uses the sequence ys to collapse z zs
    """
    #print(ys.size())
    batch_len, seq_len = ys.size()
    assert len(qs) == batch_len
    assert len(qs[0]) == seq_len
    assert len(qs[0][0]) == seq_len
    try:
        dim = len(q[0][0][0])
    except:
        dim = 1
	
    ls_ = torch.zeros(batch_len).long()
    ret = torch.zeros(batch_len, seq_len, seq_len, dim)
    for batch_ix, (y, l) in enumerate(zip(ys,ls)):
         chunks = get_chunks(y[:l])
         for chunk_ix, ((start, end), _) in enumerate(chunks):
             for chunk_ix_, ((start_, end_), _) in enumerate(chunks):
                 ret[batch_ix, chunk_ix, chunk_ix_, :] = torch.mean(torch.cat([qs[batch_ix][i][j] for (i,j) in product(range(start,end),range(start_,end_))],0), 0)
         ls_[batch_ix] = len(chunks)
    #print(torch.max(ret))
    return ret, ls_

def pack_sequence(batch, pad=0):
    """
    Assumes @batch is a list of B Tensors, each Tensor is Lx*

    Returns list of BxTx*, list[int]
    """
    lengths = [len(ex) for ex in batch]
    B, T = len(batch), max(lengths)
    shape = (B, T, *batch[0].size()[1:])

    ret = batch[0].new(*shape)
    ret.fill_(pad)
    ret = ret.view(B, T, -1)
    for i, ex in enumerate(batch):
        ret[i, :lengths[i], :] = ex.view(lengths[i], -1)
    ret = ret.view(*shape)

    return ret, torch.LongTensor(lengths)

def pack_sequence_2d(batch, pad=0):
    """
    Assumes @batch is a list of B Tensors, each Tensor is LxLx*

    Returns list of BxTxTx*, list[int]
    """
    lengths = [len(ex) for ex in batch]
    B, T = len(batch), max(lengths)
    shape = (B, T, T, *batch[0].size()[2:])

    ret = batch[0].new(*shape)
    ret.fill_(pad)
    ret = ret.view(B, T, T, -1)
    # Copy over data.
    for i, ex in enumerate(batch):
        ret[i, :lengths[i], :lengths[i], :] = ex.view(lengths[i], lengths[i], -1)
    ret = ret.view(*shape)

    return ret, torch.LongTensor(lengths)

def pack_sequence_3d(batch, pad=0):
    """
    Assumes @batch is a list of B LxLxd lists

    Returns a similar list of B TxTxd lists
    """
    lengths = [len(ex) for ex in batch]
    B, T = len(batch), max(lengths)
    T_ = max(max(max(len(end) for end in start) for start in ex) for ex in batch)
    shape = (B, T, T, T_)

    lengths_ = torch.LongTensor(B, T, T)

    ret = torch.LongTensor(*shape).zero_()
    ret = ret.view(B, T, T, T_)
    for i, ex in enumerate(batch):
        for start in range(lengths[i]):
            for end in range(lengths[i]):
                l = len(batch[i][start][end])
                if l > 0:
                    ret[i, start, end, :l] = torch.from_numpy(ex[start][end])
                lengths_[i, start, end] = len(ex[start][end])
    ret = ret.view(*shape)

    return ret, torch.LongTensor(lengths), lengths_


def create_batch(batch, pad=0):
    xs, ys, qs, zs = zip(*batch)
    ls, ixs = zip(*sorted([(-len(x), i) for i, x in enumerate(xs)]))
    xs, ys, qs, zs = [xs[i] for i in ixs], [ys[i] for i in ixs], [qs[i] for i in ixs], [zs[i] for i in ixs]

    xs, ls = pack_sequence(list(xs), pad)
    ys, ls_ = pack_sequence(list(ys), pad)
    assert (ls == ls_).all()
    zs, ls_ = pack_sequence_2d(list(zs), pad)
    assert (ls == ls_).all()
    qs, ls_, pls = pack_sequence_3d(list(qs), None)
    assert (ls == ls_).all()

    return [ixs, xs.contiguous(), ys.contiguous(), zs.contiguous(), qs, ls.contiguous(), pls.contiguous()]

def length_mask(x, lengths, use_cuda=False):
    """
    Assumes @x is B, T, * and returns B*L, *
    """
    shape = x.size()

    mask = torch.ByteTensor(*shape)
    mask.fill_(0)
    mask = mask.view(*shape[:2], -1)
    for i, _ in enumerate(x):
        mask[i, :lengths[i], :] = 1

    if use_cuda:
        mask=mask.cuda()

    return x[mask.squeeze(-1)].view(-1, *shape[2:])

def length_mask_2d(x, lengths, use_cuda=False):
    """
    Assumes @x is B, T, T, * and returns B*L*L, *
    """
    shape = x.size()

    mask = torch.ByteTensor(*shape)
    mask.fill_(0)
    mask = mask.view(*shape[:3], -1)
    for i, l in enumerate(lengths):
        mask[i, :l, :l, :] = 1

    if use_cuda:
        mask=mask.cuda()

    return x[mask.squeeze(-1)].view(-1, *shape[3:])

def position_matrix(seq_len):
    """
    Returns a seq_len x seq_len matrix where position (i,j) is i - j
    """
    x = torch.arange(0, seq_len).unsqueeze(-1).expand(seq_len, seq_len)
    return x.t() - x

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()

def log_sum_exp(mat, axis=0):
    max_score, _ = mat.max(axis)

    if mat.dim() > 1:
        max_score_broadcast = max_score.unsqueeze(axis).expand_as(mat)
    else:
        max_score_broadcast = max_score

    return max_score + \
        torch.log(torch.sum(torch.exp(mat - max_score_broadcast), axis))

def softmax(x, axis=-1):
    """
    Computes softmax over the elements of x.
    """
    return torch.exp(x - log_sum_exp(x, axis).unsqueeze(axis).expand(*x.size()))

def to_one_hot(x, dim):
    """
    Converts a LongTensor @x from an integer format into one with float format and one-hot.
    """
    assert x.type() == 'torch.LongTensor'
    ret = torch.zeros(*x.size(), dim)
    ret.scatter_(-1, x.view(-1,1), 1.)
    return ret

def to_one_hot_logits(x, dim):
    """
    Converts a LongTensor @x from an integer format into one with float format and one-hot.
    """
    assert x.type() == 'torch.LongTensor'
    ret = NINF * torch.ones(*x.size(), dim)
    ret.scatter_(-1, x.unsqueeze(-1), 0.)
    return ret

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        xs, ys, qs, zs = self.data[i]
        xs = torch.from_numpy(xs)
        ys = torch.from_numpy(ys)
        zs = torch.from_numpy(zs)

        return xs, ys, qs, zs

def run_epoch(model, dataset, optimizer=None, train=False, use_cuda=False, collapse_spans=True):
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss, count = 0., 0.
    scorer = Scorer()

    for _, xs, ys, zs, qs, ls, pls in dataset:
        if train and optimizer:
            optimizer.zero_grad()
        if use_cuda:
            xs, ys, zs, ls = xs.cuda(), ys.cuda(), zs.cuda(), ls.cuda()

        if collapse_spans:
            zs, ls_ = collapse_edges(ys, zs, ls) # span lengths.
        else:
            ls_ = ls

        xs, ys, zs, ls, ls_, pls = Variable(xs), Variable(ys), Variable(zs), Variable(ls), Variable(ls_), Variable(pls)

        yhs, zhs, lhs = model(xs, qs, ls, pls)

        ## haven't changed anything here yet.
        loss = model.loss(xs, ys, qs, zs, ls, ls_, pls)
        if train and optimizer:
            loss.backward()
            #clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()

        batch_size = xs.size()[0]
        epoch_loss += (loss.data[0] * batch_size - count * epoch_loss)/(count+batch_size)
        count += batch_size

        if hasattr(dataset, "set_postfix"):
            dataset.set_postfix(loss=epoch_loss)

        for i, (l, l_, lh) in enumerate(zip(ls.data, ls_.data, lhs.data)):
            # l_ is gold chunks and lh is guess chunks 
            scorer.update_with_tokens(ys[i,:l].data.numpy(), zs[i,:l_,:l_].data.numpy(), yhs[i,:l].data.numpy(), zhs[i,:lh,:lh].data.numpy() if lh > 0 else [], collapse_spans=collapse_spans)

    return epoch_loss, scorer

def train_model(model, train_dataset, dev_dataset=None, n_epochs=15, use_cuda=False, learning_rate=1.0, **kwargs):
    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=create_batch, shuffle=True)
    dev_loader = dev_dataset and DataLoader(dev_dataset, batch_size=16, collate_fn=create_batch)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

    if use_cuda:
        model.cuda()

    epoch_it = trange(n_epochs, desc="Epochs")
    best_epoch, best_dev_loss = 0, np.inf
    best_state_dict = model.state_dict()
    for epoch in epoch_it:
        loss, train_scorer = run_epoch(model, tqdm(train_loader, desc="Train batch"), optimizer, train=True, use_cuda=use_cuda)
        logger.info("Train %d loss: %.2f", epoch, loss)
        logger.info("Train %d %s", epoch, train_scorer)

        #if epoch > n_epochs // 2:
        #    model.config["use_gold"] = False
        model.state_dict()

        if dev_loader:
            loss, dev_scorer = run_epoch(model, tqdm(dev_loader, desc="Dev batch"), train=False, use_cuda=use_cuda)
            logger.info("Dev %d Loss: %.2f", epoch, loss)
            logger.info("Dev %d %s", epoch, dev_scorer)
            epoch_it.set_postfix(
                loss="{:.3f}".format(loss),
                node_f1="{:3f}".format(dev_scorer.state["span_node"].f1()),
                edge_f1="{:3f}".format(dev_scorer.state["span_edge"].f1()),
                )
            if loss < best_dev_loss:
                logger.info("Updating best epoch to be %d", epoch)
                best_epoch = epoch
                best_dev_loss = loss
                best_state_dict = model.state_dict()
        else:
            dev_scorer = Scorer()
            best_epoch = epoch
            best_dev_loss = loss
            best_state_dict = model.state_dict()

    if best_epoch < n_epochs-1:
        logger.info("Restoring params to best epoch (%d), with loss=%.2f", best_epoch, best_dev_loss)
        model.load_state_dict(best_state_dict)
    else:
        logger.info("Last epoch had best lost; consider increasing # of epochs.")

    return model, best_epoch, train_scorer.summary(), dev_scorer.summary()

def run_model(model, dataset, use_cuda=False, collapse_spans=False, **kwargs):
    loader = DataLoader(dataset, batch_size=16, collate_fn=create_batch)

    if use_cuda:
        model.cuda()

    output = []
    for ixs, xs, _, _, qs, ls, pls in tqdm(loader, desc="run batch"):
        if use_cuda:
            xs = xs.cuda()
        yhs, zhs, lhs = model(Variable(xs), Variable(qs), Variable(ls), Variable(pls))
        yhs, zhs = softmax(yhs), softmax(zhs)

        output_ = []
        # Reverse ixs:
        for i, (l, lh) in enumerate(zip(ls, lhs.data)):
            output_.append((yhs[i, :l].data.numpy(), zhs[i, :lh, :lh].data.numpy() if lh > 0 else []))
        ixs_ = invert_index(ixs)
        for ix in ixs_:
            output.append(output_[ix])
    return output

## ML: not sure if need to change this ... is this being used by anything?
def run_example(model, x, q):
    l, _ = x.shape
    # TODO: support qs, pls
    xs, qs, ls = np.array([x]), np.array([q]), np.array([l])
    xs, qs, ls = torch.from_numpy(xs), torch.from_numpy(qs), torch.from_numpy(ls)

    yhs, zhs, lhs = model(Variable(xs), Variable(qs), Variable(ls))

    for i, (l, lh) in enumerate(zip(ls, lhs.data)):
        return (yhs[i, :l].data.numpy(), zhs[i, :lh, :lh].data.numpy() if lh > 0 else [])
