from glob import glob
import numpy as np
import tensorflow as tf

def generate_images_iid(img_size, sigma, signal, num, config):
    rng = np.random.default_rng(config.data.seed)
    images = []
    for _ in range(num):
        image = rng.normal(0, sigma, (img_size, img_size))
        abnormal_size = int(img_size/3)
        abnormal_x = rng.integers(0, img_size - abnormal_size)
        abnormal_y = rng.integers(0, img_size - abnormal_size)
        image[abnormal_x:abnormal_x+abnormal_size, abnormal_y:abnormal_y+abnormal_size] += signal
        images.append(image)
    images = np.array(images)
    images = images.reshape(num, img_size, img_size, 1)
    return images

class DatasetMaker(tf.keras.utils.Sequence):
    def __init__(self, root, category, config, is_train=True, batch_size=32):
        self.image_transform = tf.keras.Sequential([
            tf.keras.layers.Resizing(config.data.image_size, config.data.image_size),
        ])
        
        self.config = config
        if ('iid' in category) or ('corr' in category):
            if 'iid' in category:
                self.images = generate_images_iid(config.data.image_size, 1, 0, 2000, config)
                self.image_files = ['good' for i in range(len(self.images))]
                
                estimate_var_imgs = generate_images_iid(config.data.image_size, 1, 0, 1000, config)

                self.images = tf.convert_to_tensor(self.images, dtype=tf.float64)
                self.estimate_var_imgs = tf.convert_to_tensor(estimate_var_imgs, dtype=tf.float64)
                
                self.mean = tf.reduce_mean(self.estimate_var_imgs)
                self.std = tf.cast(np.std(self.estimate_var_imgs, ddof=1), dtype=tf.float64)

            elif 'corr' in category:
                rng = np.random.default_rng(config.data.seed)
                image_size = config.data.image_size
                cov = [[] for _ in range(image_size*image_size)]
                for i in range(image_size*image_size):
                    for j in range(image_size*image_size):
                        cov[i].append(np.abs(i - j))
                cov = np.array(cov)
                cov = np.power(0.5, cov)
                self.images = rng.multivariate_normal(
                    np.zeros(image_size * image_size), cov, 2000
                )
                self.images = self.images.reshape(2000, image_size, image_size, 1)
                self.images = tf.convert_to_tensor(self.images, dtype=tf.float64)
                self.image_files = ['good' for i in range(len(self.images))]
                self.estimate_var_imgs = rng.multivariate_normal(
                    np.zeros(image_size * image_size), cov, 1000
                )
                self.estimate_var_imgs = self.estimate_var_imgs.reshape(1000, image_size, image_size, 1)
                self.estimate_var_imgs = tf.convert_to_tensor(self.estimate_var_imgs, dtype=tf.float64)
                self.mean = tf.reduce_mean(self.estimate_var_imgs)
                self.std = tf.cast(np.std(self.estimate_var_imgs, ddof=1), dtype=tf.float64)
        
        self.is_train = is_train
        self.images = (self.images - self.mean) / self.std

    def __getitem__(self, index):
        image = self.images[index]

        image_file = self.image_files[index]
        if self.is_train:
            label = "good" 
        else:
            if "good" in image_file:
                label = "good"
            else:
                label = "bad"

        return tf.convert_to_tensor(image), tf.convert_to_tensor(label)

    def __len__(self):
        return len(self.image_files)