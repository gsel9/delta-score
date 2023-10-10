import tensorflow as tf 


class DeltaLoss(tf.keras.losses.Loss):

    def __init__(self, name="delta"):
        super(DeltaLoss, self).__init__(name=name)

    def call(self, y_true, y_pred, sample_weight=None):

        y_pred = tf.squeeze(tf.clip_by_value(y_pred, clip_value_min=1e-6, clip_value_max=1 - 1e-6))
        y_true = tf.squeeze(tf.cast(y_true, y_pred.dtype))
        
        delta = y_true * (1 - 2 * y_pred) + (1 - y_true) * (2 * y_pred - 1)

        if sample_weight is None: 
            return delta

        return sample_weight * delta 


class LMDNNLoss(tf.keras.losses.Loss):
    
    def __init__(self, C=1.0, name="lmdnn"):
        super(LMDNNLoss, self).__init__(name=name)
        
        self.C = C 

    def _bce(self, y_true, y_pred, sample_weight=None):

        bce = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
        
        if sample_weight is None:
            return -1.0 * bce 
            
        return -1.0 * sample_weight * bce 
        
    def _delta(self, y_true, y_pred):
        
        return tf.squeeze(y_true * (1 - 2 * y_pred) + (1 - y_true) * (2 * y_pred - 1))
        
    def call(self, y_true, y_pred, sample_weight=None):

        y_pred = tf.squeeze(tf.clip_by_value(y_pred, clip_value_min=1e-6, clip_value_max=1 - 1e-6))
        y_true = tf.squeeze(tf.cast(y_true, y_pred.dtype))
    
        margin_reg = tf.square(1 + self._delta(y_true, y_pred))

        return self._bce(y_true, y_pred, sample_weight=sample_weight) + self.C * margin_reg


class LDAMLoss(tf.keras.losses.Loss):
    
    def __init__(self, C=1.0, max_m=0.5, s=1, eps=1e-16, name="ldam"):
        super(LDAMLoss, self).__init__(name=name)
        
        self.C = C
        self.s = s 
        self.eps = eps
        self.max_m = max_m

    def _bce(self, y_true, y_pred, sample_weight=None):

        bce = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
        
        if sample_weight is None:
            return -1.0 * bce 
            
        return -1.0 * sample_weight * bce

    def call(self, y_true, y_pred, sample_weight=None):
        # Amounts to a shifted sigmoid. 

        y_pred = tf.squeeze(tf.clip_by_value(y_pred, clip_value_min=1e-6, clip_value_max=1 - 1e-6))
        y_true = tf.squeeze(tf.cast(y_true, y_pred.dtype))

        margin = self.C / tf.square(tf.square(tf.reduce_sum(y_true)))
        
        # Convert both class probabilities to logits 
        logits = tf.math.log(y_pred / (1 - y_pred))

        y_pred = 1 / (1 + tf.exp(-1 * (logits - margin)))

        return self._bce(y_true, y_pred, sample_weight=sample_weight)
      