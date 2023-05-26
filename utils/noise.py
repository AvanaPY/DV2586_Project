import numpy as np

def create_categorised_noise(noise_img_dim : int, n_classes : int):
    """
        Generates one noise image per class. 
    """
    n_classes -= 1
    noise_imgs = np.zeros(shape=(n_classes, n_classes, noise_img_dim))
    noise_y = np.arange(n_classes) + 1
    for i in range(n_classes):
        noise_imgs[i,i,:] = np.random.uniform(low=-1, high=1,size=noise_img_dim)
    return noise_imgs, noise_y

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    x, y = create_categorised_noise(7, 27)
    print(x.shape)
    print(x)
    print(y)