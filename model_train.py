import tensorflow as tf
import data_preprocess
import generator_model as gm
from tensorflow.contrib import seq2seq
data_dir = './data/news_opinion.txt'
text = data_preprocess.load_data(data_dir)

data_preprocess.preprocess_and_save_data(data_dir, gm.token_lookup, gm.create_lookup_tables)
int_text, vocab_to_int, int_to_vocab, token_dict = data_preprocess.load_preprocess()

# Number of Epochs
num_epochs = 100
# Batch Size
batch_size = 256
# RNN Size
rnn_size = 1024
# Embedding Dimension Size
embed_dim = None
# Sequence Length
seq_length = 20
# Learning Rate
learning_rate = 0.001
# Show stats for every n number of batches
show_every_n_batches = 20

save_dir = './save'

train_graph = tf.Graph()

with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = gm.get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = gm.get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = gm.build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

batches = gm.get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)

    # Save parameters for checkpoint
    data_preprocess.save_params((seq_length, save_dir))