from __future__ import division
import keras.backend as K
import theano.tensor as T
from keras.layers import Layer, InputSpec
from keras import initializations, regularizers, constraints
import theano
import numpy as np
floatX = theano.config.floatX


class LearningPrior(Layer):
    def __init__(self, nb_gaussian, init='normal', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, **kwargs):
        self.nb_gaussian = nb_gaussian
        self.init = initializations.get(init, dim_ordering='th')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        super(LearningPrior, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_shape = (self.nb_gaussian*4, )
        self.W = self.init(self.W_shape, name='{}_W'.format(self.name))

        self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint

    def get_output_shape_for(self, input_shape):
        self.b_s = input_shape[0]
        self.height = input_shape[2]
        self.width = input_shape[3]

        return self.b_s, self.nb_gaussian, self.height, self.width

    def call(self, x, mask=None):
        mu_x = self.W[:self.nb_gaussian]
        mu_y = self.W[self.nb_gaussian:self.nb_gaussian*2]
        sigma_x = self.W[self.nb_gaussian*2:self.nb_gaussian*3]
        sigma_y = self.W[self.nb_gaussian*3:]

        self.b_s = x.shape[0]
        self.height = x.shape[2]
        self.width = x.shape[3]

        e = self.height / self.width
        e1 = (1 - e) / 2
        e2 = e1 + e

        mu_x = K.clip(mu_x, 0.25, 0.75)
        mu_y = K.clip(mu_y, 0.35, 0.65)

        sigma_x = K.clip(sigma_x, 0.1, 0.9)
        sigma_y = K.clip(sigma_y, 0.2, 0.8)

        x_t = T.dot(T.ones((self.height, 1)), self._linspace(0, 1.0, self.width).dimshuffle('x', 0))
        y_t = T.dot(self._linspace(e1, e2, self.height).dimshuffle(0, 'x'), T.ones((1, self.width)))

        x_t = K.repeat_elements(K.expand_dims(x_t, dim=-1), self.nb_gaussian, axis=-1)
        y_t = K.repeat_elements(K.expand_dims(y_t, dim=-1), self.nb_gaussian, axis=-1)

        gaussian = 1 / (2 * np.pi * sigma_x * sigma_y + K.epsilon()) * \
                   T.exp(-((x_t - mu_x) ** 2 / (2 * sigma_x ** 2 + K.epsilon()) +
                           (y_t - mu_y) ** 2 / (2 * sigma_y ** 2 + K.epsilon())))

        gaussian = K.permute_dimensions(gaussian, (2, 0, 1))
        max_gauss = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(gaussian, axis=1), axis=1)), self.height, axis=-1)), self.width, axis=-1)
        gaussian = gaussian / max_gauss

        output = K.repeat_elements(K.expand_dims(gaussian, dim=0), self.b_s, axis=0)

        return output

    @staticmethod
    def _linspace(start, stop, num):
        # produces results identical to:
        # np.linspace(start, stop, num)
        start = T.cast(start, floatX)
        stop = T.cast(stop, floatX)
        num = T.cast(num, floatX)
        step = (stop - start) / (num - 1)
        return T.arange(num, dtype=floatX) * step + start

    def get_config(self):
        config = {'nb_gaussian': self.nb_gaussian,
                  'init': self.init.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  }
        base_config = super(LearningPrior, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))