import logging

import tensorflow as tf

from .Model import SequenceModel
from .Model import Config as _Config
from .util import featurize_windows, pad_sequences

logger = logging.getLogger(__name__)

LSTMCell = tf.contrib.rnn.LSTMCell
stack_bidirectional_dynamic_rnn = tf.contrib.rnn.stack_bidirectional_dynamic_rnn
static_bidirectional_rnn = tf.contrib.rnn.static_bidirectional_rnn

class LSTMModel(SequenceModel):
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
        n_features = n_window_features

        max_length=100

        dropout = 0.5
        embed_size = 50
        hidden_size = 100
        batch_size = 64
        n_epochs = 10
        lr = 0.01

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        """
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_length, self.config.n_features), name="x")
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_length), name="y")
        self.length_placeholder = tf.placeholder(tf.int32, shape=(None), name="length")
        self.mask_placeholder = tf.placeholder(tf.bool, shape=(None, self.config.max_length), name="mask")
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name="p_drop")

    def create_feed_dict(self, inputs_batch, length_batch, mask_batch, labels_batch=None, **kwargs):
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
            self.length_placeholder: length_batch,
            self.mask_placeholder: mask_batch,
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
        embeddings = tf.reshape(tf.nn.embedding_lookup(L, x), [-1, self.config.max_length, self.config.n_features * self.config.embed_size])
        return embeddings

    def add_prediction_op(self):
        """Adds the 1-hidden-layer NN:
            h = Relu(xW + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_dropU + b2
        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """
        xs = self.add_embedding()
        dropout_rate = self.dropout_placeholder
        lengths = self.length_placeholder

        # NOTE: Try stacked LSTMs
        fw_cell = LSTMCell(self.config.hidden_size)
        bw_cell = LSTMCell(self.config.hidden_size)

        zs, state_fw, state_bw = static_bidirectional_rnn(
            fw_cell, bw_cell,
            tf.unstack(xs, axis=1),
            dtype="float32",
            sequence_length=lengths)

        U = tf.get_variable("U", dtype=tf.float32, shape=(2*self.config.hidden_size, self.config.n_classes),
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", dtype=tf.float32, shape=(self.config.n_classes,),)
        preds_ = []
        # Now use the predictions to produce output labels.
        for time_step in range(self.config.max_length):
            z = zs[time_step]
            output = tf.nn.dropout(z, dropout_rate, name="h_drop")
            pred = tf.matmul(z, U) + b
            preds_.append(pred)

        # Make sure to reshape @preds here.
        ### YOUR CODE HERE ###
        preds = tf.stack(preds_)
        preds = tf.transpose(preds, perm=[1, 0, 2])
        ### END YOUR CODE

        assert preds.get_shape().as_list() == [None, self.config.max_length, self.config.n_classes], "predictions are not of the right shape. Expected {}, got {}".format([None, self.max_length, self.config.n_classes], preds.get_shape().as_list())
        return preds

        return pred

    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.
        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        y = self.labels_placeholder
        elems = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds, labels=y)
        elems = tf.boolean_mask(elems, self.mask_placeholder)
        loss = tf.reduce_mean(elems)
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
        examples = featurize_windows(examples, self.helper.START, self.helper.END)
        zero_vector, zero_label = [0] * self.config.n_features, 0
        return pad_sequences(examples, self.config.max_length, zero_vector, zero_label)

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        assert len(examples_raw) == len(examples)
        assert len(examples_raw) == len(preds)

        ret = []
        for i, (sentence, labels) in enumerate(examples_raw):
            _, _, _, mask = examples[i]
            sentence = sentence[:self.config.max_length]
            labels = labels[:self.config.max_length]
            labels_ = [l for l, m in zip(preds[i], mask) if m] # only select elements of mask.
            assert len(labels_) == len(labels) and len(sentence) == len(labels)
            ret.append([sentence, labels, labels_])
        return ret

    def predict_on_batch(self, sess, inputs_batch, length_batch, mask_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch=inputs_batch, length_batch=length_batch, mask_batch=mask_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=2), feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch, length_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch, length_batch=length_batch, mask_batch=mask_batch,
                                     labels_batch=labels_batch, dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def __init__(self, helper, config, model_output):
        super(LSTMModel, self).__init__(helper, config, model_output)

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.length_placeholder = None
        self.mask_placeholder = None
        self.dropout_placeholder = None

        self.build()
