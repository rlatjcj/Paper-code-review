import tensorflow as tf

def crossentropy(args):
    def _loss(y_true, y_pred):
        if args.classes == 1:
            return tf.keras.losses.binary_crossentropy(y_true, y_pred)
        else:
            return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return _loss


def supervised_contrastive(args):
    def _loss(labels, logits):
        labels = tf.reshape(labels, (-1, 1))
        # indicator for yi=yj
        mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)

        # (zi dot zj) / temperature
        anchor_dot_contrast = tf.math.divide(
            tf.linalg.matmul(logits, tf.transpose(logits)), 
            tf.constant(args.temperature, dtype=tf.float32))

        # for numerical stability
        logits_max = tf.math.reduce_max(anchor_dot_contrast, axis=-1, keepdims=True)
        anchor_dot_contrast = anchor_dot_contrast - logits_max

        # tile mask for 2N images
        mask = tf.tile(mask, (2, 2))

        # indicator for i \neq j
        logits_mask = tf.ones_like(mask)-tf.eye(args.batch_size*2)
        mask *= logits_mask

        # compute log_prob
        # log(\exp(z_i \cdot z_j / temperature) / (\sum^{2N}_{k=1} \exp{z_i \cdot z_k / temperature}))
        # = (z_i \cdot z_j / temperature) - log(\sum^{2N}_{k=1} \exp{z_i \cdot z_k / temperature})
        # apply indicator for i \neq k in denominator
        exp_logits = tf.math.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - tf.math.log(tf.math.reduce_sum(exp_logits, axis=-1, keepdims=True))
        mean_log_prob = tf.reduce_sum(mask * log_prob, axis=-1) / tf.reduce_sum(mask, axis=-1)
        loss = -tf.reduce_mean(tf.reshape(mean_log_prob, (2, args.batch_size)), axis=0)
        return loss
    return _loss
