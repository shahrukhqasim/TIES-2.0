import tensorflow as tf
import numpy as np

num_batch = 10
max_vertices = 80

samples_per_vertex=5

placeholder_num_vertices = tf.placeholder(tf.float32, shape=(num_batch,))

x = tf.ones(shape=(num_batch, max_vertices)) / placeholder_num_vertices[..., tf.newaxis]
mask = tf.sequence_mask(placeholder_num_vertices, maxlen=max_vertices)  # [batch, num_vertices, 1] It will broadcast on the last dimension!
x = x * tf.cast(mask, tf.float32)



x = tf.distributions.Categorical(probs=x).sample(sample_shape=(max_vertices, samples_per_vertex))
x = tf.transpose(x, perm=[2,0,1])
print(x.shape)
# 0/0


with tf.Session() as sess:
    v = np.zeros(shape=(num_batch,))+30 + np.random.randint(0, 20, num_batch)
    print(v)

    m = sess.run(x, feed_dict={placeholder_num_vertices:v})

    # print(m)

    print(np.min(np.min(m, axis=1), axis=1))
    print(np.max(np.max(m, axis=1), axis=1))


    0/0

