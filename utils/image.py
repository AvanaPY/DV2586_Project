import os
from PIL import Image

def images_to_gif(folder : str, file_name : str):
    gif_file = os.path.join(folder, file_name)
    
    images = os.listdir(folder)
    images = filter(lambda x : 'png' in x, images)
    images = sorted(images, key=lambda x : int(x.split('_')[1].split('.')[0]))
    frames = [Image.open(os.path.join(folder, frame)) for frame in images]
    frames[0].save(gif_file, format='GIF', append_images=frames[1:], save_all=True, loop=0)

if __name__ == '__main__':
    images_to_gif('/home/emkl/Documents/Projects/MNIST_AutoEncoder/checkpoints/run46/imgs', 'training.gif')