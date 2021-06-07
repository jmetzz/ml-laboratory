"""This was developed by Lukas Biewald.
The original can be found at https://github.com/lukas/ml-class
"""
import cv2
import numpy
from PIL import Image
from scipy.signal import convolve2d


def draw_image(channel, img_file, background_color, pixel_size=10):
    image = Image.open(img_file)
    red, green, blue = image.split()
    if channel == 'grey':
        image = Image.open(img_file).convert('LA')
    elif channel == 'r':
        image = red
    elif channel == 'g':
        image = green
    elif channel == 'b':
        image = blue

    image = image.resize((image.size[0] // pixel_size, image.size[1] // pixel_size), Image.NEAREST)
    image = image.resize((image.size[0] * pixel_size, image.size[1] * pixel_size), Image.NEAREST)
    image = image.convert('RGB')
    pixel = image.load()

    for i in range(0, image.size[0], pixel_size):
        for j in range(0, image.size[1], pixel_size):
            for r in range(pixel_size):
                pixel[i + r, j] = background_color
                pixel[i, j + r] = background_color

    return image


def draw_image_conv(img_file, kernel, x, y, background_color, pixel_size=10):
    image = Image.open(img_file)
    image = image.resize((image.size[0] // pixel_size, image.size[1] // pixel_size), Image.NEAREST)

    new_image = convolve2d(numpy.asarray(image)[:, :, 0], kernel)
    new_image = new_image.clip(0.0, 255.0)
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            if i > y or (i == y and j >= x):
                new_image[i, j] = 0

    image = Image.fromarray(new_image)
    image = image.convert('RGB')
    image = image.resize((image.size[0] * pixel_size, image.size[1] * pixel_size), Image.NEAREST)
    pixel = image.load()

    for i in range(0, image.size[0], pixel_size):
        for j in range(0, image.size[1], pixel_size):
            for r in range(pixel_size):
                pixel[i + r, j] = background_color
                pixel[i, j + r] = background_color

    return image


def show_image(img_file, x, y, w, h, background_color, kernel, color=True, pixel_size=10):
    if color:
        image_r = draw_image('r', img_file, background_color, pixel_size)
        image_g = draw_image('g', img_file, background_color, pixel_size)
        image_b = draw_image('b', img_file, background_color, pixel_size)
        opencv_image_r = cv2.cvtColor(numpy.array(image_r), cv2.COLOR_RGB2BGR)
        opencv_image_g = cv2.cvtColor(numpy.array(image_g), cv2.COLOR_RGB2BGR)
        opencv_image_b = cv2.cvtColor(numpy.array(image_b), cv2.COLOR_RGB2BGR)

        conv_image = draw_image_conv(img_file, kernel, x, y, background_color)
        opencv_conv_image = cv2.cvtColor(numpy.array(conv_image), cv2.COLOR_RGB2BGR)

        cv2.rectangle(opencv_image_r,
                      (x * pixel_size, y * pixel_size),
                      ((x + w) * pixel_size, (y + h) * pixel_size),
                      (0, 0, 255))
        cv2.rectangle(opencv_image_g,
                      (x * pixel_size, y * pixel_size),
                      ((x + w) * pixel_size, (y + h) * pixel_size),
                      (0, 0, 255))
        cv2.rectangle(opencv_image_b,
                      (x * pixel_size, y * pixel_size),
                      ((x + w) * pixel_size, (y + h) * pixel_size),
                      (0, 0, 255))
        cv2.imshow('image red', opencv_image_r)
        cv2.imshow('image green', opencv_image_g)
        cv2.imshow('image blue', opencv_image_b)
        cv2.imshow('image out', opencv_conv_image)
    else:
        image = draw_image('grey', img_file, background_color, pixel_size)
        opencv_image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)

        conv_image = draw_image_conv(img_file, kernel, x, y, background_color)
        opencv_conv_image = cv2.cvtColor(numpy.array(conv_image), cv2.COLOR_RGB2BGR)

        cv2.rectangle(opencv_image, (x * pixel_size, y * pixel_size), ((x + w) * pixel_size, (y + h) * pixel_size),
                      (0, 0, 255))
        cv2.imshow('image', opencv_image)
        cv2.imshow('image conv', opencv_conv_image)


kernel = [[0, 0, 0],
          [0, 0.5, 0],
          [0, 0, 0]]

background_color = (1,) * 3
img_file = 'puppy.jpg'

w = 3
h = 1

show_image(img_file, 100, 100, w, h, background_color, kernel, pixel_size=10, color=True)
while True:
    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
        break
