import tensorflow as tf
import numpy as np
from collections import Counter


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of editorial split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    token_dict = {}
    token_dict['.'] = "||Period||"
    token_dict[','] = "||Comma||"
    token_dict['"'] = "||Quotation_Mark||"
    token_dict[';'] = "||Semicolon||"
    token_dict['!'] = "||Exclamation_Mark||"
    token_dict['?'] = "||Question_Mark||"
    token_dict['('] = "||Left_Parentheses||"
    token_dict[')'] = "||Right_Parentheses||"
    token_dict['--'] = "||Dash||"
    token_dict['\n'] = "||Return||"

    return token_dict

def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32,[None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32,  name='learning_rate')

    return inputs, targets, learning_rate


def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    #drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=0.5)
    cell = tf.contrib.rnn.MultiRNNCell([lstm] * 3)
    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, name="initial_state")

    return cell, initial_state

def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)

    return embed

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, name="final_state")

    return outputs, final_state


def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    embedding = get_embed(input_data, vocab_size, rnn_size)
    outputs, final_state = build_rnn(cell, embedding)
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)

    return logits, final_state


def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    slice_size = batch_size * seq_length
    n_batches = int(len(int_text) / slice_size)
    result = []

    for i in range(n_batches):
        start, stop = i * slice_size, (i + 1) * slice_size
        x = int_text[start: stop]
        x = np.pad(x, (0, slice_size - len(x)), 'constant')
        x = x.reshape([-1, seq_length])

        y = int_text[start + 1: stop + 1]
        y = np.pad(y, (0, slice_size - len(y)), 'constant')
        y = y.reshape([-1, seq_length])
        result.append([x, y])

    return np.asarray(result)
