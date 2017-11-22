import data_preprocess
import numpy as np
import tensorflow as tf

_, vocab_to_int, int_to_vocab, token_dict = data_preprocess.load_preprocess()
seq_length, load_dir = data_preprocess.load_params()


def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    inputTensor = loaded_graph.get_tensor_by_name("input:0")
    initialStateTensor = loaded_graph.get_tensor_by_name("initial_state:0")
    finalStateTensor = loaded_graph.get_tensor_by_name("final_state:0")
    probsTensor = loaded_graph.get_tensor_by_name("probs:0")

    return inputTensor, initialStateTensor, finalStateTensor, probsTensor


def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    int_word = np.argmax(probabilities)

    return int_to_vocab[int_word]


gen_length = 200
prime_word = 'trump'
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word]
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})

        pred_word = pick_word(probabilities[dyn_seq_length - 1], int_to_vocab)

        gen_sentences.append(pred_word)

    # Remove tokens
    editorial = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        editorial = editorial.replace(' ' + token.lower(), key)
    editorial = editorial.replace('\n ', '\n')
    editorial = editorial.replace('( ', '(')

    print(editorial)