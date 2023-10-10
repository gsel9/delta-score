import tensorflow as tf 


class DeltaLossMulticlass(tf.keras.losses.Loss):

    def __init__(self, name="delta"):
        super(DeltaLossMulticlass, self).__init__(name=name)

    def call(self, y_true, p_pred, sample_weight=None):

        p_pred = tf.squeeze(tf.clip_by_value(p_pred, clip_value_min=1e-6, clip_value_max=1 - 1e-6))
        y_true = tf.squeeze(tf.cast(y_true, p_pred.dtype))

        delta = tf.reduce_max((1 - y_true) * p_pred, axis=1) - tf.reduce_sum(y_true * p_pred, axis=1)
        
        if sample_weight is None: 
            return delta

        return sample_weight * delta


class LMDNNLossMulticlass(tf.keras.losses.Loss):
    
    def __init__(self, C=1.0, name="lmdnn"):
        super(LMDNNLossMulticlass, self).__init__(name=name)
        
        self.C = C 

    def _delta(self, y_true, y_pred):
        
        y_pred = tf.squeeze(tf.clip_by_value(y_pred, clip_value_min=1e-6, clip_value_max=1 - 1e-6))
        y_true = tf.squeeze(tf.cast(y_true, y_pred.dtype))

        return tf.reduce_max((1 - y_true) * y_pred, axis=1) - tf.reduce_sum(y_true * y_pred, axis=1)

    def _cce(self, y_true, y_pred, sample_weight=None):

        if sample_weight is None:
            return -1.0 * tf.reduce_sum(tf.math.log(y_pred) * tf.squeeze(y_true), axis=1)

        return -1.0 * sample_weight * tf.reduce_sum(tf.math.log(y_pred) * tf.squeeze(y_true), axis=1)
        
    def call(self, y_true, y_pred, sample_weight=None):

        y_pred = tf.squeeze(tf.clip_by_value(y_pred, clip_value_min=1e-6, clip_value_max=1 - 1e-6))
        y_true = tf.squeeze(tf.cast(y_true, y_pred.dtype))
    
        margin_reg = tf.square(1 + self._delta(y_true, y_pred))
        
        return self._cce(y_true, y_pred, sample_weight=sample_weight) + self.C * margin_reg
