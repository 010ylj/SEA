multi_grad=[]
for i in range(x_test.shape[0]):

    images = x_test[[i]]

    images = tf.Variable(images, dtype=float)
    with tf.GradientTape(watch_accessed_variables=True) as tape:  # 计算梯度

        pred = model_(tf.convert_to_tensor(images), training=False)  # 因为你要把x转换成一个numpy数组，而np.array是不可微的。

        class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
        loss = pred[0][class_idxs_sorted[0]]

    grads = tape.gradient(loss, images, unconnected_gradients=tf.UnconnectedGradients.ZERO)

    dgrad_abs = tf.math.abs(grads)

    multigrad= dgrad_abs.numpy() * images.numpy()

    multi_grad.append(multigrad.sum())


