import logging

import tensorflow as tf

from .util import window_iterator
from .Model import SequenceModel
from .Model import Config as _Config

logger = logging.getLogger(__name__)

def make_windowed_data(data, start, end, window_size = 1):
    """Uses the input sequences in @data to construct new windowed data points.
    Args:
        data: is a list of (sentence, labels) tuples. @sentence is a list
            containing the words in the sentence and @label is a list of
            output labels. Each word is itself a list of
            @n_features features. For example, the sentence "Chris
            Manning is amazing" and labels "PER PER O O" would become
            ([[1,9], [2,9], [3,8], [4,8]], [1, 1, 4, 4]). Here "Chris"
            the word has been featurized as "[1, 9]", and "[1, 1, 4, 4]"
            is the list of labels.
        start: the featurized `start' token to be used for windows at the very
            beginning of the sentence.
        end: the featurized `end' token to be used for windows at the very
            end of the sentence.
        window_size: the length of the window to construct.
    Returns:
        a new list of data points, corresponding to each window in the
        sentence. Each data point consists of a list of
        @n_window_features features (corresponding to words from the
        window) to be used in the sentence and its NER label.
        If start=[5,8] and end=[6,8], the above example should return
        the list
        [([5, 8, 1, 9, 2, 9], 1),
         ([1, 9, 2, 9, 3, 8], 1),
         ...
         ]
    """
    windowed_data = []
    for sentence, labels in data:
        for i, window in enumerate(window_iterator(sentence, window_size, beg=start, end=end)):
            window = sum(window, [])
            label = labels[i]
            windowed_data.append((window, label))
    return windowed_data

def test_make_windowed_data():
    sentences = [[[1,1], [2,0], [3,3]]]
    sentence_labels = [[1, 2, 3]]
    data = zip(sentences, sentence_labels)
    w_data = make_windowed_data(data, start=[5,0], end=[6,0], window_size=1)

    assert len(w_data) == sum(len(sentence) for sentence in sentences)

    assert w_data == [
        ([5,0] + [1,1] + [2,0], 1,),
        ([1,1] + [2,0] + [3,3], 2,),
        ([2,0] + [3,3] + [6,0], 3,),
        ]

class WindowModel(SequenceModel):
    """
    Implements a feedforward neural network with an embedding layer and
    single hidden layer.
    This network will predict what label (e.g. PER) should be given to a
    given token (e.g. Manning) by  using a featurized window around the token.
    """

    class Config(_Config):
        """Holds model hyperparams and data information.

        The config class is used to store various hyperparameters and dataset
        information parameters. Model objects are passed a Config() object at
        instantiation.
        """
        TOTAL_LABELS = ("n", "iv", "du", "dv", "dvc", "dl", "sm", "dr", "il", "im")
        SHORT_LABELS = ("n", "iv", "du", "dv", "dvc", "dl",)
        LABELS = SHORT_LABELS
        n_classes = len(LABELS) # Number of classes to predict as output.

        n_word_features = 2 # Currently only 2 features per word; word and casing.
        window_size = 1 # Size of the symmetric window.
        n_window_features = (2*window_size + 1) * n_word_features # The total number of features used for each window.

        dropout = 0.5
        embed_size = 50
        hidden_size = 200
        batch_size = 64
        n_epochs = 10
        lr = 0.001

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        """
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.n_window_features), name="x")
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None), name="y")
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name="p_drop")

    def create_feed_dict(self, inputs_batch, labels_batch=None, **kwargs):
        """Creates the feed_dict for the model.
        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {
            self.input_placeholder: inputs_batch,
            self.dropout_placeholder: kwargs.get("dropout", 1.)
            }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates an embedding tensor and initializes it with
              self.helper.embeddings.
            - Uses the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, n_window_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, n_window_features * embedding_size).
        Returns:
            embeddings: tf.Tensor of shape (None, n_window_features*embed_size)
        """
        L = tf.Variable(self.helper.embeddings, name="L")
        x = self.input_placeholder
        embeddings = tf.reshape(tf.nn.embedding_lookup(L, x), [-1, self.config.n_window_features * self.config.embed_size])
        return embeddings

    def add_prediction_op(self):
        """Adds the 1-hidden-layer NN:
            h = Relu(xW + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_dropU + b2
        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """

        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder
        W = tf.get_variable("W", dtype=tf.float32, shape=(self.config.n_window_features * self.config.embed_size, self.config.hidden_size),
                            initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", dtype=tf.float32, shape=(self.config.hidden_size,),
                             initializer=tf.contrib.layers.xavier_initializer())
        U = tf.get_variable("U", dtype=tf.float32, shape=(self.config.hidden_size,self.config.n_classes),
                            initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2", dtype=tf.float32, shape=(self.config.n_classes,),
                             initializer=tf.contrib.layers.xavier_initializer())

        h = tf.nn.relu(tf.matmul(x, W) + b1, name="h")
        h_drop = tf.nn.dropout(h, dropout_rate, name="h_drop")
        pred = tf.matmul(h_drop, U) + b2
        return pred

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        y = self.labels_placeholder
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y))
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.
        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)
        return train_op

    def preprocess_sequence_data(self, examples):
        return make_windowed_data(examples, start=self.helper.START, end=self.helper.END, window_size=self.config.window_size)

    def consolidate_predictions(self, examples_raw, _, preds):
        """Batch the predictions into groups of sentence length.
        """
        ret = []
        #pdb.set_trace()
        i = 0
        for sentence, labels in examples_raw:
            labels_ = preds[i:i+len(sentence)]
            i += len(sentence)
            ret.append([sentence, labels, labels_])
        return ret

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=1), feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def __init__(self, helper, config, model_output):
        super(WindowModel, self).__init__(helper, config, model_output)

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None

        self.build()
