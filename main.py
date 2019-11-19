import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

import pry;


def to_blocks(image_height, image_width):
    blocks = []
    for i in range(image_height // BLOCK_HEIGHT):
        for j in range(image_width // BLOCK_WIDTH):
            block = []
            for y in range(BLOCK_HEIGHT):
                for x in range(BLOCK_WIDTH):
                    for color in range(3):
                        block.append(ARRAY_IMAGE[i * BLOCK_HEIGHT + y, j * BLOCK_WIDTH + x, color])
            blocks.append(block)
    return np.array(blocks)


def to_array(blocks, image_height, image_width):
    array = []
    blocks_in_line = image_width // BLOCK_WIDTH
    for i in range(image_height // BLOCK_HEIGHT):
        for y in range(BLOCK_HEIGHT):
            line = []
            for j in range(blocks_in_line):
                for x in range(BLOCK_WIDTH):
                    pixel = []
                    for color in range(3):
                        pixel.append(blocks[i * blocks_in_line + j, (y * BLOCK_WIDTH * 3) + (x * 3) + color])
                    line.append(pixel)
            array.append(line)
    return np.array(array)


def show(array_image):
    read_image = 1 * (array_image + 1) / 2
    plt.axis('off')
    plt.imshow(read_image)
    plt.show()


def read_image():
    image = mpimg.imread("simple.png")
    image = (2.0 * image / 1.0) - 1.0
    return np.array(image)


def image_size():
    return np.size(ARRAY_IMAGE, 0), np.size(ARRAY_IMAGE, 1)


def set_layers():
    first_layer = np.random.rand(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE) * 2 - 1
    temp = np.copy(first_layer)
    second_layer = temp.transpose()
    return first_layer, second_layer


def alpha(y):
    elements_sum = sum(np.matmul(element, element) for element in y)
    return MAX_ALPHA if elements_sum == 0 else (1 / elements_sum)

def study():
    current_error = ERROR_MAX + 1
    epoch = 0

    first_layer, second_layer = set_layers()

    while current_error > ERROR_MAX:
        current_error = 0
        epoch += 1
        for block in blocks():
            y = block @ first_layer
            x1 = y @ second_layer
            delta = x1 - block
            first_layer -= alpha(y) * np.matmul(np.matmul(block.transpose(), delta), second_layer.transpose())
            second_layer -= alpha(y) * np.matmul(y.transpose(), delta)
            error = (delta * delta).sum()
            current_error += error
        print('Epoch ', epoch, '   ', 'errors ', current_error)
    return first_layer, second_layer


def blocks():
    image_height, image_width = image_size()
    return to_blocks(image_height, image_width).reshape(number_of_blocks(), 1, INPUT_LAYER_SIZE)


def number_of_blocks():
    image_height, image_width = image_size()
    return int((image_height * image_width) / (BLOCK_HEIGHT * BLOCK_WIDTH))


def compres():
    image_height, image_width = image_size()
    first_layer, second_layer = study()

    result = []
    for block in blocks():
        result.append(block.dot(first_layer).dot(second_layer))
    result = np.array(result)

    show(ARRAY_IMAGE)
    show(to_array(result.reshape(number_of_blocks(), INPUT_LAYER_SIZE), image_height, image_width))


BLOCK_HEIGHT = 4
BLOCK_WIDTH = 4
HIDDEN_LAYER_SIZE = 16
INPUT_LAYER_SIZE = BLOCK_HEIGHT * BLOCK_HEIGHT * 3
ERROR_MAX = 10000.0
ARRAY_IMAGE = read_image()
MAX_ALPHA = 0.0007

compres()
