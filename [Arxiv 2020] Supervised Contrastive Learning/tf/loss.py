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
        temperature = tf.constant(args.temperature, dtype=tf.float32)
        loss = []
        for i in range(2*args.batch_size):
            loss_per_anchor = 0.
            for j in range(2*args.batch_size):
                if i == j:
                    continue
                if labels[i] != labels[j]:
                    continue

                denominator = 0.
                for k in range(2*args.batch_size):
                    if i == k:
                        continue

                    denominator += tf.math.exp(
                        tf.linalg.matmul(
                            tf.expand_dims(logits[i], axis=0), 
                            tf.transpose(tf.expand_dims(logits[k], axis=0))) / temperature)

                loss_per_anchor += tf.math.log(
                    tf.math.exp(
                        tf.linalg.matmul(
                            tf.expand_dims(logits[i], axis=0), 
                            tf.transpose(tf.expand_dims(logits[j], axis=0))) / temperature
                    ) / denominator)
            
            loss.append(-loss_per_anchor)
        return loss
    return _loss
