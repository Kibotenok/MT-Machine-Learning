#  Script to work with image import and save
#  Created by Anropov N.A. 03.08.2019
import os
from tensorflow import image, io
import matplotlib.pyplot as plt
from PIL.Image import fromarray


def image_import(all_paths, img_size, ch=3):
    """Function to convert two images into tensors for putting it into the network
       Arguments: all_path - list of paths to images,
                  img_size - image size
                  ch - kind of image color channels
       Return: list of tensors"""

    images = [image.resize(image.decode_image(io.read_file(path)), [img_size, img_size, ch]) for path in all_paths]
    images /= 255.0
    print('Images are ready')

    return images


def save_image(img, path, save_format='png', name='result'):
    """Function to save result to necessary path
       Arguments: img - result image
                  path - path to save image
                  save_format - format of saving image
                  name - image name
       Return: nothing"""

    enc_img = fromarray(img, 'RGB')
    enc_img.save(os.path.join(path, name+'.'+save_format))
    print('Image was successfully saved to %s with name %s' % path, name+'.'+save_format)


def show_image(img):
    """Function to show image and information about it
       Arguments: img - tensor
       Return: nothing"""

    plt.imshow(img)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(img.shape)
    plt.show()
