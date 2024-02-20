import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, BatchNormalization, ReLU,  MaxPooling2D, Concatenate, UpSampling2D, Reshape, Add, AveragePooling2D

from loss import *


class UNet(tf.keras.Model):
    def __init__(
        self,
        img_size,
        config,
        in_ch=1,
        time_dim=1,
    ):  
        super().__init__()
        num_layer = 32 
        self.config = config
        self.time_dim = time_dim

        # Define input layers
        x_input = Input(shape=(img_size, img_size, in_ch), dtype=tf.float64)
        t_input = Input(shape=(time_dim, ), dtype=tf.float64)
        
        v = Dense(time_dim, input_shape=(time_dim, ), activation='linear')(t_input)
        
        # Encoder
        t = Dense(in_ch, input_shape=(time_dim, ))(v)
        t = Reshape((1, 1, in_ch))(t)
                
        x1 = Add()([x_input, t])
        x1 = Conv2D(filters=num_layer, kernel_size=3, padding='same', activation='relu')(x1)
        x = AveragePooling2D(pool_size=(2, 2))(x1)
        
        
        t = Dense(num_layer, input_shape=(time_dim, ))(v)
        t = Reshape((1, 1, num_layer))(t)
        
        x2 = Add()([x, t])
        x2 = Conv2D(filters=num_layer*2, kernel_size=3, padding='same', activation='relu')(x2)
        x = AveragePooling2D(pool_size=(2, 2))(x2)

        
        t = Dense(num_layer*2, input_shape=(time_dim, ))(v)
        t = Reshape((1, 1, num_layer*2))(t)
        
        x3 = Add()([x, t])
        x3 = Conv2D(filters=num_layer*4, kernel_size=3, padding='same', activation='relu')(x3)
        x = AveragePooling2D(pool_size=(2, 2))(x3)

        
        # Bottleneck
        t = Dense(num_layer*4, input_shape=(time_dim, ))(v)
        t = Reshape((1, 1, num_layer*4))(t)
        
        x = Add()([x, t])
        x = Conv2D(filters=num_layer*8, kernel_size=3, padding='same', activation='relu')(x)

        # Decoder
        x = UpSampling2D()(x)
        x = Concatenate(axis=-1)([x, x3])
        
        t = Dense(num_layer*4+num_layer*8, input_shape=(time_dim, ))(v)
        t = Reshape((1, 1, num_layer*4+num_layer*8))(t)
        
        x = Add()([x, t])
        x = Conv2D(filters=num_layer*4, kernel_size=3, padding='same', activation='relu')(x)
        
        
        x = UpSampling2D()(x)
        x = Concatenate(axis=-1)([x, x2])

        t = Dense(num_layer*4+num_layer*2, input_shape=(time_dim, ))(v)
        t = Reshape((1, 1, num_layer*2+num_layer*4))(t)
        
        x = Add()([x, t])
        x = Conv2D(filters=num_layer*2, kernel_size=3, padding='same', activation='relu')(x)
        

        x = UpSampling2D()(x)
        x = Concatenate(axis=-1)([x, x1])
        
        t = Dense(num_layer*2+num_layer, input_shape=(time_dim, ))(v)
        t = Reshape((1, 1, num_layer*2+num_layer))(t)
        
        x = Add()([x, t])
        x = Conv2D(filters=num_layer, kernel_size=3, padding='same', activation='relu')(x)
        
        x_out = Conv2D(1, kernel_size=1, padding='same', dtype=tf.float64)(x)
        
        # Create the model
        self.model = tf.keras.models.Model(inputs=[x_input, t_input], outputs=x_out)


    def call(self, input, training=False):
        x_input, t_input = input
        return self.model([x_input, t_input])


    def get_hidden_output(self, layer_name):
        intermediate_layer_model = tf.keras.models.Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.call([[self.x_input, self.t_input]], training=False)
        return intermediate_output


    @tf.function
    def train_step(self, images):
        t = tf.random.uniform(shape=(self.config.data.batch_size, ), minval=0, maxval=self.config.model.trajectory_steps, dtype=tf.int64)
        with tf.GradientTape() as tape:
            x_0 = tf.cast(images, dtype=tf.float64)
            betas = np.linspace(self.config.model.beta_start, self.config.model.beta_end, self.config.model.trajectory_steps, dtype=np.float64) # 0から1までの間をtrajectory_steps個に分割
            b = tf.constant(betas, dtype=tf.float64)
            e = tf.random.normal(shape=x_0.shape, dtype=tf.float64)

            at = tf.math.cumprod(1 - b, axis=0)
            at = tf.gather(at, t, axis=0)
            at = tf.reshape(at, [-1, 1, 1, 1])

            x_t = tf.math.sqrt(at) * x_0 + tf.math.sqrt(1 - at) * e
            
            t_float = tf.cast(t, dtype=tf.float64)
            t_float = tf.convert_to_tensor(t_float, dtype=tf.float64)
            
            output = self.call([x_t, t_float], training=True)
            loss = get_loss(output, e)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(e, output)
        return {"loss": loss}
