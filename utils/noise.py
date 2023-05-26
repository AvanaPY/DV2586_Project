import numpy as np

def create_categorised_noise(noise_img_dim : int, n_classes : int):
    n_classes -= 1
    noise_imgs = np.zeros(shape=(n_classes, noise_img_dim, noise_img_dim, n_classes))
    noise_y = np.arange(n_classes) + 1
    for i in range(n_classes):
        noise_imgs[i,:,:,i] = np.random.uniform(low=0, high=1,size=(noise_img_dim, noise_img_dim))
    return noise_imgs, noise_y
