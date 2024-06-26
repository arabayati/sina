class SmoothnessLossLayer(tf.keras.layers.Layer):
    def __init__(self, lambda_smooth=0.0005, **kwargs):
        quantile_threshold=0.3
        super(SmoothnessLossLayer, self).__init__(**kwargs)
        self.lambda_smooth = lambda_smooth
        self.quantile_threshold = quantile_threshold

    def call(self, inputs):
        # Exclude the last element (bias term) for smoothing calculation
        inputs_to_consider = inputs[:, :-1]

        # Calculate differences
        differences = inputs_to_consider[:, 1:] - inputs_to_consider[:, :-1]

        # Calculate the quantile threshold value
        quantile_values = tfp.stats.percentile(differences, 100 * self.quantile_threshold, axis=-1, interpolation='linear')

        # Mask differences based on quantile threshold
        masked_differences = tf.where(differences < quantile_values[..., tf.newaxis], differences, 0.0)

        # Using L1 norm for smoothness on masked differences
        smoothness_loss = tf.reduce_mean(tf.abs(masked_differences))

        self.add_loss(self.lambda_smooth * smoothness_loss)
        return inputs  # Pass through layer.




class AddBiasLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AddBiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(
            name='bias',
            shape=input_shape[1:],
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return  0*inputs + self.bias


# surrogate model
EffectiveSize = 365
class Manual_LSTM(tf.keras.Model):
    def __init__(self, reg_l2=0.001, EffectiveSize = 365, lambda_smooth = 1):  # default value for reg_l2 is set to 0.01
        super(Manual_LSTM, self).__init__()
        self.EffectiveSize = EffectiveSize


        self.lstm1 = tf.keras.layers.LSTM(units=32, return_sequences=False, input_shape=(365, 24))
        self.dropout = tf.keras.layers.Dropout(0.4)


        self.pre_denseT = tf.keras.layers.Dense(units=128)
        self.pre_denseP = tf.keras.layers.Dense(units=128)
        self.pre_densePET = tf.keras.layers.Dense(units=128)
        self.pre_denseAET = tf.keras.layers.Dense(units=128)


        self.denseT = tf.keras.layers.Dense(units=EffectiveSize +1      ,activation='elu', kernel_regularizer=regularizers.l1(reg_l2))
        self.denseP = tf.keras.layers.Dense(units=EffectiveSize +1     ,activation='elu', kernel_regularizer=regularizers.l1(reg_l2))
        self.densePET = tf.keras.layers.Dense(units=EffectiveSize +1    ,activation='elu', kernel_regularizer=regularizers.l1(reg_l2))
        self.denseAET = tf.keras.layers.Dense(units=EffectiveSize +1   ,activation='elu', kernel_regularizer=regularizers.l1(reg_l2))


        # Smoothness Loss Layer for each output
        self.smooth_loss_dT = SmoothnessLossLayer(lambda_smooth)
        self.smooth_loss_dP = SmoothnessLossLayer(lambda_smooth)
        self.smooth_loss_dPET = SmoothnessLossLayer(lambda_smooth)
        self.smooth_loss_dAET = SmoothnessLossLayer(lambda_smooth)


    def call(self, input_tensor, return_components=False):

        T = input_tensor[:, -self.EffectiveSize:, 0]
        P = input_tensor[:, -self.EffectiveSize:, 1]
        PET = input_tensor[:, -self.EffectiveSize:, 2]
        AET = input_tensor[:, -self.EffectiveSize:, 3]

        T = tf.concat([T, tf.ones_like(T[:, :1])], axis=1)
        P = tf.concat([P, tf.ones_like(P[:, :1])], axis=1)
        PET = tf.concat([PET, tf.ones_like(PET[:, :1])], axis=1)
        AET = tf.concat([AET, tf.ones_like(AET[:, :1])], axis=1)



        x = self.lstm1(input_tensor)
        x = self.dropout(x)

        # Apply the new pre-dense layers
        pre_dT = self.pre_denseT(x)
        pre_dP = self.pre_denseP(x)
        pre_dPET = self.pre_densePET(x)
        pre_dAET = self.pre_denseAET(x)

        dT = self.denseT(pre_dT)
        dP = self.denseP(pre_dP)
        dPET = self.densePET(pre_dPET)
        dAET = self.denseAET(pre_dAET)


        # Adding Smoothness Loss
        dT = self.smooth_loss_dT(dT)
        dP = self.smooth_loss_dP(dP)
        dPET = self.smooth_loss_dPET(dPET)
        dAET = self.smooth_loss_dAET(dAET)


        dQ_T = tf.math.reduce_sum(tf.math.multiply(dT, T), axis=1, keepdims=True)
        dQ_P = tf.math.reduce_sum(tf.math.multiply(dP, P), axis=1, keepdims=True)
        dQ_PET = tf.math.reduce_sum(tf.math.multiply(dPET, PET), axis=1, keepdims=True)
        dQ_AET = tf.math.reduce_sum(tf.math.multiply(dAET, AET), axis=1, keepdims=True)



        output = dQ_T + dQ_P + dQ_PET + dQ_AET
        if return_components:
            return dT[:, :-1], dP[:, :-1], dPET[:, :-1], dAET[:, :-1], dQ_T, dQ_P, dQ_PET, dQ_AET   # [:, :-1] is for not returning the bias terms

            # return tf.math.multiply(dT, T), tf.math.multiply(dP, P), tf.math.multiply(dPET, PET), tf.math.multiply(dAET, AET)
        else:
            return output

surrogate_model = Manual_LSTM(reg_l2= 0.01, EffectiveSize = 365, lambda_smooth = 2.5)  ########################## main model setup



surrogate_model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=LearningRate))


input =  np.random.rand(1,TIME_STEP,Num_Input_features)    # one batch, and exacyly the shape of the input
Model_Activated = surrogate_model(input)
