import tensorflow as tf

def sample_gumbel(shape, eps=1e-20):
    """ Sample from Gumbel(0,1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
    """ Sample from the Gumbel-Softmax distribution and optionally discretize.
    Inputs:
        - logits: [batch_size, n_classes] unnormalized log-probs
        - temperature: non-negative scalar
        - hard: if True, take argmax to discretize
    Returns:
        - [batch_size, n_classes] sample from the distribution
        if hard, it will be one-hot, otherwise sums to 1
    """

    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y
