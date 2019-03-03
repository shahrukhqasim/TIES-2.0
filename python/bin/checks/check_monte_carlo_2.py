import tensorflow as tf
import numpy as np


v = 3000
probabs = np.ones(shape=(v,))/v
x = tf.distributions.Categorical(probs=probabs).sample(sample_shape=(5000))


with tf.Session() as sess:
    v = sess.run(x)
    print(v)
    print(np.min(v), np.max(v))


