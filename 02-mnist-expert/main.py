loss = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

session.run(tf.initialize_all_variables())

for i in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    if not i % 100:
        print(accuracy.eval({x: mnist.test.images[:500], y_: mnist.test.labels[:500], keep_prob: 1.0}))
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
