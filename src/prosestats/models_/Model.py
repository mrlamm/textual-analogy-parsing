import json
import logging
from types import MethodType

from tqdm import tqdm

from .util import ConfusionMatrix, minibatches, get_chunks

logger = logging.getLogger(__name__)

class Model(object):
    """
    A helper abstraction to handle many of the common operations for
    tensorflow.
    """
    def __init__(self, helper, config, model_output):
        self.helper = helper
        self.config = config
        self.model_output = model_output

        self.pred = None
        self.loss = None
        self.train_op = None

    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.

        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used as
        inputs by the rest of the model building and will be fed data during
        training.

        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, inputs_batch, labels_batch=None, **kwargs):
        """Creates the feed_dict for one step of training.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If labels_batch is None, then no labels are added to feed_dict.

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Args:
            loss: Loss tensor (a scalar).
        Returns:
            train_op: The Op for training.
        """

        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def preprocess_data(self, examples):
        """
        Preprocesses input data into a format that can be passed into
        create_feed_dict
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def consolidate_predictions(self, data_raw, data, preds):
        """
        Convert predictions according to the batching process back into
        the original format.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def evaluate(self, sess, examples, examples_raw):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs.
            examples_raw: A list of the original input/output sequence pairs.
        Returns:
            The F1 score for predicting tokens as named entities.
        """
        raise NotImplementedError("Each model must implement this")

    def run_epoch(self, sess, train_examples, dev_set, train_examples_raw, dev_set_raw):
        it = tqdm(minibatches(train_examples, self.config.batch_size), desc="training")
        for batch in it:
            loss = self.train_on_batch(sess, *batch)
            it.set_postfix(train_loss=loss)

        logger.info("Evaluating on training data")
        token_cm, entity_scores = self.evaluate(sess, train_examples, train_examples_raw)
        if token_cm is not None:
            logger.info("Token-level confusion matrix:\n" + token_cm.as_table())
            logger.info("Token-level scores:\n" + token_cm.summary())
        logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        logger.info("Evaluating on development data")
        token_cm, entity_scores = self.evaluate(sess, dev_set, dev_set_raw)
        if token_cm is not None:
            logger.info("Token-level confusion matrix:\n" + token_cm.as_table())
            logger.info("Token-level scores:\n" + token_cm.summary())
        logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        f1 = entity_scores[-1]
        return f1

    def output(self, sess, inputs_raw, inputs=None):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        if inputs is None:
            inputs = self.preprocess_data(self.helper.vectorize(inputs_raw))

        preds = []
        for batch in tqdm(minibatches(inputs, self.config.batch_size, shuffle=False), desc="minibatch"):
            # Ignore predict
            batch = batch[:1] + batch[2:]
            preds_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)
        return self.consolidate_predictions(inputs_raw, inputs, preds)

    def fit(self, sess, saver, train_examples_raw, dev_set_raw, n_epochs=10):
        best_score = 0.

        train_examples = self.preprocess_data(train_examples_raw)
        dev_set = self.preprocess_data(dev_set_raw)

        for epoch in range(n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            score = self.run_epoch(sess, train_examples, dev_set, train_examples_raw, dev_set_raw)
            if score > best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s", self.model_output)
                    saver.save(sess, self.model_output)
            print("")
        return best_score

class SequenceModel(Model):
    """
    A sequence model expects a sequence of tokens, and predicts
    a sequence of labels (of the same length).
    """
    def __init__(self, helper, config, model_output):
        Model.__init__(self, helper, config, model_output)

    def preprocess_data(self, examples):
        """Preprocess sequence data for the model.

        Args:
            examples: A list of vectorized input/output sequences.
        Returns:
            A new list of vectorized input/output pairs appropriate for the model.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def consolidate_predictions(self, data_raw, data, preds):
        """
        Convert a sequence of predictions according to the batching
        process back into the original sequence.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def evaluate(self, sess, examples, examples_raw):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs.
            examples_raw: A list of the original input/output sequence pairs.
        Returns:
            The F1 score for predicting tokens as named entities.
        """
        if self.config.LABELS is not NotImplemented:
            token_cm = ConfusionMatrix(labels=self.config.LABELS)
        else:
            token_cm = None

        correct_preds, total_correct, total_preds = 0., 0., 0.
        for _, labels, labels_  in self.output(sess, examples_raw, examples):
            for l, l_ in zip(labels, labels_):
                if token_cm is not None: token_cm.update(l, l_)
            gold = set(get_chunks(labels, 0))
            pred = set(get_chunks(labels_, 0))
            correct_preds += len(gold.intersection(pred))
            total_preds += len(pred)
            total_correct += len(gold)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

        entity_scores = (p, r, f1)
        if token_cm is not None:
            logger.info("Token-level confusion matrix:\n" + token_cm.as_table())
            logger.info("Token-level scores:\n" + token_cm.summary())
        logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        return f1

class SequenceArcModel(Model):
    """
    A sequence arc model expects a sequence of tokens, and predicts
    a sequence of labels (of the same length).
    """
    def __init__(self, helper, config, model_output):
        Model.__init__(self, helper, config, model_output)

    def preprocess_data(self, examples):
        """Preprocess sequence data for the model.

        Args:
            examples: A list of vectorized input/output sequences.
        Returns:
            A new list of vectorized input/output pairs appropriate for the model.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def consolidate_predictions(self, data_raw, data, preds):
        """
        Convert a sequence of predictions according to the batching
        process back into the original sequence.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def evaluate(self, sess, examples, examples_raw):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs.
            examples_raw: A list of the original input/output sequence pairs.
        Returns:
            The F1 score for predicting tokens as named entities.
        """
        if self.config.LABELS is not NotImplemented:
            token_cm = ConfusionMatrix(labels=self.config.LABELS)
        else:
            token_cm = None

        correct_preds, total_correct, total_preds = 0., 0., 0.
        for _, labels, labels_  in self.output(sess, examples_raw, examples):
            for l, l_ in zip(labels, labels_):
                if token_cm is not None: token_cm.update(l, l_)
            gold = set(get_chunks(labels, 0))
            pred = set(get_chunks(labels_, 0))
            correct_preds += len(gold.intersection(pred))
            total_preds += len(pred)
            total_correct += len(gold)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

        entity_scores = (p, r, f1)
        if token_cm is not None:
            logger.info("Token-level confusion matrix:\n" + token_cm.as_table())
            logger.info("Token-level scores:\n" + token_cm.summary())
        logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        return f1
