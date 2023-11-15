import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, LeakyReLU, Input, Conv3DTranspose
from tensorflow.keras.models import Model
import numpy as np

# Define the generator model
def build_generator(latent_dim):
    # latent_dim is the size of the latent space (a vector of random numbers) the generator uses to input data
    model = tf.keras.Sequential([
        # Means it is a densly connected neural network layer
        # This layer is the first layer of the model and receives the latent vector as input.
        Dense(128 * 16 * 16 * 16, input_dim=latent_dim),
        # Reshape layer reshapes the original layer which is the Dense layer
        # Shapes it into a specific shape that is suitable for the convolutional layers to follow. Here
        # it reshapes to a 4D tensor with the shape `16, 16, 16, 128` 
        Reshape((16, 16, 16, 128)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv3DTranspose(64, (4, 4, 4), strides=(2, 2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv3DTranspose(1, (4, 4, 4), strides=(2, 2, 2), padding='same', activation='tanh')
    ])
    return model

# Define the discriminator model
def build_discriminator(input_shape):
    model = tf.keras.Sequential([
        Conv3D(64, (4, 4, 4), strides=(2, 2, 2), padding='same', input_shape=input_shape),
        LeakyReLU(alpha=0.2),
        Conv3D(128, (4, 4, 4), strides=(2, 2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        tf.keras.layers.Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

# Define the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    return model

# Define the main training loop
def train_gan(gan, generator, discriminator, dataset, latent_dim, epochs, batch_size):
    gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
    for epoch in range(epochs):
        for _ in range(dataset.shape[0] // batch_size):
            noise = np.random.normal(0, 1, [batch_size, latent_dim])
            generated_images = generator.predict(noise)
            image_batch = dataset[np.random.randint(0, dataset.shape[0], size=batch_size)]
            X = np.concatenate([image_batch, generated_images])
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X, y_dis)
            noise = np.random.normal(0, 1, [batch_size, latent_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, y_gen)
        print(f"Epoch {epoch} | Discriminator Loss: {d_loss} | Generator Loss: {g_loss}")

# Example usage
latent_dim = 100
input_shape = (16, 16, 16, 1)
generator = build_generator(latent_dim)
discriminator = build_discriminator(input_shape)
gan = build_gan(generator, discriminator)

# Load your 3D dataset (you need to provide this)
# For simplicity, I'm assuming a random array of 3D images with shape (num_samples, 16, 16, 16, 1)
# Replace this with your actual data
dataset = np.random.randn(1000, 16, 16, 16, 1)

# Train the GAN
train_gan(gan, generator, discriminator, dataset, latent_dim, epochs=100, batch_size=32)
