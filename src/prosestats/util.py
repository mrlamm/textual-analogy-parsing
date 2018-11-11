"""
Simple common utilities
"""
import pdb
import logging
from collections import defaultdict, Counter
import numpy as np
import tqdm

logger = logging.getLogger(__name__)

def cargmax(cntr):
    if not isinstance(cntr, Counter):
        cntr = Counter(cntr)
    return (cntr and cntr.most_common(1)[0][0]) or None

def cmax(cntr):
    if not isinstance(cntr, Counter):
        cntr = Counter(cntr)
    return (cntr and cntr.most_common(1)[0][1]) or 0.

def cnormalize(cntr):
    z = sum(cntr.values())
    for k, v in cntr.items():
        cntr[k] = v/z
    return cntr

def vnormalize(lst):
    z = sum(lst)
    return [v/z for v in lst]

def invert_index(ixs):
    """
    Given a list of elements with indexes [2,3,1,0], invert it so that
    indexing on these elements restores the order [3, 2, 0, 1].
    abcd -> cdba -> abcd
    """
    _, ret = zip(*sorted((i,j) for j, i in enumerate(ixs)))
    return list(ret)

def test_invert_index():
    ixs = [2, 3, 1, 0]
    assert invert_index(ixs) == [3, 2, 0, 1]

def print_sentence(output, sentence, labels, predictions):
    spacings = [max(len(sentence[i]), len(labels[i]), len(predictions[i])) for i in range(len(sentence))]
    # Compute the word spacing
    output.write("x : ")
    for token, spacing in zip(sentence, spacings):
        output.write(token)
        output.write(" " * (spacing - len(token) + 1))
    output.write("\n")

    output.write("y*: ")
    for token, spacing in zip(labels, spacings):
        output.write(token)
        output.write(" " * (spacing - len(token) + 1))
    output.write("\n")

    output.write("y': ")
    for token, spacing in zip(predictions, spacings):
        output.write(token)
        output.write(" " * (spacing - len(token) + 1))
    output.write("\n")

def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = isinstance(data, list) and isinstance(data[0], list) or isinstance(data[0], np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)

def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if isinstance(data, np.ndarray) else [data[i] for i in minibatch_idx]

def minibatches(data, batch_size, shuffle=True):
    batches = [np.array(col) for col in zip(*data)]
    return get_minibatches(batches, batch_size, shuffle)

def window_iterator(seq, n=1, beg="<s>", end="</s>"):
    """
    Iterates through seq by returning windows of length 2n+1
    """
    for i in range(len(seq)):
        l = max(0, i-n)
        r = min(len(seq), i+n+1)
        ret = seq[l:r]
        if i < n:
            ret = [beg,] * (n-i) + ret
        if i+n+1 > len(seq):
            ret = ret + [end,] * (i+n+1 - len(seq))
        yield ret

def test_window_iterator():
    assert list(window_iterator(list("abcd"), n=0)) == [["a",], ["b",], ["c",], ["d"]]
    assert list(window_iterator(list("abcd"), n=1)) == [["<s>","a","b"], ["a","b","c",], ["b","c","d",], ["c", "d", "</s>",]]

def one_hot(n, y):
    """
    Create a one-hot @n-dimensional vector with a 1 in position @i
    """
    if isinstance(y, int):
        ret = np.zeros(n)
        ret[y] = 1.0
        return ret
    elif isinstance(y, list):
        ret = np.zeros((len(y), n))
        ret[np.arange(len(y)),y] = 1.0
        return ret
    else:
        raise ValueError("Expected an int or list got: " + y)

def to_table(data, row_labels, column_labels, number_format="%d"):
    """Pretty print tables.
    Assumes @data is a 2D array and uses @row_labels and @column_labels
    to display table.
    """
    # Convert data to strings
    data = [[number_format%v for v in row] for row in data]
    cell_width = max(
        max(map(lambda x: len(str(x)), row_labels)),
        max(map(lambda x: len(str(x)), column_labels)),
        max(max(map(len, row)) for row in data))
    def c(s):
        """adjust cell output"""
        return str(s) + " " * (cell_width - len(str(s)))
    ret = ""
    ret += "\t".join(map(c, column_labels)) + "\n"
    for l, row in zip(row_labels, data):
        ret += "\t".join(map(c, [l] + row)) + "\n"
    return ret

class PRF1(object):
    def __init__(self):
        self.value = np.array([0., 0., 0.])
        self.n = 0.

    def update(self, prf1):
        self.value += (prf1 - self.value)/(self.n+1)
        self.n += 1

    def clear(self):
        self.value[:] = 0.
        self.n = 0.

    def prec(self):
        return self.value[0]

    def rec(self):
        return self.value[1]

    def f1(self):
        return self.value[2]

    def as_table(self):
        """Print tables"""
        # Header
        data = [[self.prec(), self.rec(), self.f1()]]
        return to_table(data, [""], ["", "p", "r", "f1"], number_format="%0.3f")


class ConfusionMatrix(object):
    """
    A confusion matrix stores counts of (true, guessed) labels, used to
    compute several evaluation metrics like accuracy, precision, recall
    and F1.
    """

    def __init__(self, labels, default_label=None, ignore_labels=None):
        self.labels = list(labels)
        self.default_label = default_label
        self.ignore_labels = ignore_labels or []
        self._counts = defaultdict(Counter)
        self._stale = True

        self._summary_stats = None

    def update(self, gold, guess, count=1):
        """Update counts"""
        self._counts[gold][guess] += count
        self._stale = True

    def clear(self):
        self._counts.clear()

    def as_table(self):
        """Print tables"""
        # Header
        data = [[self._counts[l][l_] for l_ in self.labels] for l in self.labels]
        return to_table(data, self.labels, ["go\\gu"] + self.labels)

    def _update_summary(self):
        """Summarize counts"""
        keys = self.labels
        data = []
        macro = np.array([0., 0., 0., 0.])
        micro = np.array([0., 0., 0., 0.])
        for l in keys:
            if l in self.ignore_labels:
                data.append([0., 0., 0., 0.])
                continue

            tp = self._counts[l][l]
            fp = sum(self._counts[l_][l] for l_ in keys if l_ != l)
            tn = sum(self._counts[l_][l__] for l_ in keys if l_ != l for l__ in keys if l__ != l)
            fn = sum(self._counts[l][l_] for l_ in keys if l_ != l)

            acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
            prec = (tp)/(tp + fp) if tp > 0  else 0
            rec = (tp)/(tp + fn) if tp > 0  else 0
            f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0
            data.append([acc, prec, rec, f1])

            # update micro/macro averages
            if l != self.default_label: # Count count for everything that is not the default label!
                micro += np.array([tp, fp, tn, fn])
                macro += np.array([acc, prec, rec, f1])

        # micro average
        tp, fp, tn, fn = micro
        acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
        prec = (tp)/(tp + fp) if tp > 0  else 0
        rec = (tp)/(tp + fn) if tp > 0  else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0
        data.append([acc, prec, rec, f1])
        # Macro average
        data.append(macro / (len(keys) - len(self.ignore_labels)))

        self._stale = False
        self._summary_stats = (data, self.labels + ["micro","macro"], ["acc", "prec", "rec", "f1"])

    def summary(self):
        # Macro and micro average.
        if self._stale:
            self._update_summary()
        data, labels, metrics = self._summary_stats
        return to_table(data, labels, ["label",] + metrics, number_format="%0.3f")

    def _get(self, metric, tag="micro"):
        if self._stale:
            self._update_summary()

        data, labels, metrics = self._summary_stats
        assert tag in labels
        assert metric in metrics
        return data[labels.index(tag)][metrics.index(metric)]

    def acc(self, tag="micro"):
        return self._get("acc", tag)
    def prec(self, tag="micro"):
        return self._get("prec", tag)
    def rec(self, tag="micro"):
        return self._get("rec", tag)
    def f1(self, tag="micro"):
        return self._get("f1", tag)

def get_chunks(seq, default=0):
    """Breaks input of 0 0 0 1 1 0 2 ->   (1, 4, 5), (2, 6, 7) (if default is 4)"""
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = ((chunk_start, i), chunk_type,)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None
        # End of a chunk + start of a chunk!
        elif tok != default:
            if chunk_type is None:
                chunk_type, chunk_start = tok, i
            elif tok != chunk_type:
                chunk = ((chunk_start, i), chunk_type,)
                chunks.append(chunk)
                chunk_type, chunk_start = tok, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = ((chunk_start, len(seq)), chunk_type,)
        chunks.append(chunk)
    return chunks

def test_get_chunks():
    assert get_chunks([0, 0, 0, 1, 1, 0, 2, 3, 0, 4], 0) == [((3,5),1), ((6, 7),2), ((7, 8),3), ((9,10),4)]

def build_dict(words, max_words=None, offset=0):
    cnt = Counter(words)
    if max_words:
        words = cnt.most_common(max_words)
    else:
        words = cnt.most_common()
    return {word: offset+i for i, (word, _) in enumerate(words)}

def test_build_dict():
    words = "a a b a b b a c b a c a".split()
    tok2id = build_dict(words, max_words=2, offset=1)
    assert tok2id.get('a') == 1
    assert tok2id.get('b') == 2
    assert tok2id.get('c') is None

def featurize_windows(data, start, end, window_size = 1):
    """Uses the input sequences in @data to construct new windowed data points.
    """
    ret = []
    for sentence, labels in data:
        sentence_ = []
        for window in window_iterator(sentence, window_size, beg=start, end=end):
            sentence_.append(sum(window, []))
        ret.append((sentence_, labels))
    return ret

def pad_sequences(data, max_length, zero_vector, zero_label):
    """Ensures each input-output seqeunce pair in @data is of length
    @max_length by padding it with zeros and truncating the rest of the
    sequence.

    Args:
        data: is a list of (sentence, labels) tuples. @sentence is a list
            containing the words in the sentence and @label is a list of
            output labels. Each word is itself a list of
            @n_features features. For example, the sentence "Chris
            Manning is amazing" and labels "PER PER O O" would become
            ([[1,9], [2,9], [3,8], [4,8]], [1, 1, 4, 4]). Here "Chris"
            the word has been featurized as "[1, 9]", and "[1, 1, 4, 4]"
            is the list of labels.
        max_length: the desired length for all input/output sequences.
    Returns:
        a new list of data points of the structure (sentence', labels', mask).
        Each of sentence', labels' and mask are of length @max_length.
        See the example above for more details.
    """
    ret = []

    for sentence, labels in data:
        ### YOUR CODE HERE ###
        sentence_ = sentence[:max_length] + [zero_vector] * max(0, max_length - len(sentence))
        labels_ = labels[:max_length] + [zero_label] * max(0, max_length - len(labels))
        lengths_ = min(max_length, len(sentence))
        masks_ = [True] * min(len(sentence), max_length) + [False] * max(0, max_length - len(sentence))
        ret.append((sentence_, labels_, lengths_, masks_))
        ### END YOUR CODE ###
    return ret

class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.tqdm.write(msg)
        self.flush()

    def flush(self):
        super().flush()

def invert_dict(tok2id):
    """
    Inverts a dictionary from tokens to ids.
    """
    ret = [None for _ in range(max(tok2id.values())+1)]
    for k, v in tok2id.items():
        ret[v] = k
    return ret

def test_invert_dict():
    id2tok = "a b c d e f g".split()
    tok2id = {t: i for i, t in enumerate(id2tok)}

    id2tok_ = invert_dict(tok2id)

    assert id2tok_ == id2tok

class TokenSentence(object):
    def __init__(self, tokens):
        self.tokens = tokens

        self.text = ""
        self.indices = []
        offset = 0
        for token in tokens:
            self.indices.append((offset, offset+len(token)))
            self.text += token + " "
            offset += len(token) + 1
        self.text = self.text[:-1]

    def __str__(self):
        return self.text

    def to_tokens(self, span):
        start, end = None, None
        for i, (start_, end_) in enumerate(self.indices):
            if span[0] == start_:
                start = i
            if span[1] == end_:
                end = i+1
                break
        if start is None or end is None:
            return None
        return (start, end)

def trynumber(x):
    if x.lower() == "true":
        return True
    elif x.lower() == "false":
        return False
    try:
        return int(x)
    except ValueError:
        pass
    try:
        return float(x)
    except ValueError:
        pass
    return x

def dictstr(x):
    """
    A converter from string to dictionary, for use in argparse.
    """
    if "=" in x:
        k, v = x.split("=")
        # Try to parse v.
        return (k, trynumber(v))
    else:
        return (x, True)
