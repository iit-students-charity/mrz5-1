import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
from skimage.util.shape import view_as_blocks
import pry;


class Studying:
    def __init__(self):
        self.image_initialize()
        self.block_initialize()
        self.error_max = 3000.0
        self.error_current = self.error_max + 1
        self.alpha = 0.0007
        self.epoch = 0

    def study(self):
        self.set_neural_numbers_on_second_layer()
        self.set_neural_layers(self.pixels_number())
        block_form = self.image.to_blocks(self.block.width, self.block.height)

    def image_initialize(self):
        self.image = Image('simple.png')
        self.image_array = np.array(self.image)

    def block_initialize(self):
        block_width, block_height = self.block_data_input()
        self.block = Block(block_width, block_height)

    def block_data_input(self):
        print("Enter block's width: ")
        block_width = int(input())

        print("Enter block's height: ")
        block_height= int(input())

        return block_width, block_height

    def pixels_number(self):
        return self.image.width * self.image.height * 3

    def set_neural_numbers_on_second_layer(self):
        print('Enter number of neurons on the second layer: ')
        self.neurons_number = int(input())

    def set_neural_layers(self, pixels_number):
        self.first_layer = np.random.rand(self.number_of_blocks(), self.neurons_number) * 2 - 1
        self.second_layer = np.copy(self.first_layer).transpose()

    def number_of_blocks(self):
        return int((self.image.width * self.image.height) / (self.block.width * self.block.height))


class Image:
    def __init__(self, image_link):
        self.image_link = image_link
        self.image_initialize()

    def image_initialize(self):
        self.image = imread(self.image_link)
        self.image_array = np.array(self.image)

        self.set_image_size()

        return (2.0 * self.image / 1.0) - 1.0

    def set_image_size(self):
        self.width = np.size(self.image_array, 1)
        self.height = np.size(self.image_array, 0)

    def to_blocks(self, block_width, block_height):
        blocks = []
        for py in range(self.width // block_width):
            x_blocks = []
            for px in range(self.height // block_height):
                tmp = []
                for y in range(block_height):
                    for x in range(block_width):
                        for k in range(3):
                            tmp.append(self.image_array[y][x][k])
                x_blocks.append(tmp)
            blocks.append(x_blocks)
        return np.array(blocks)


class Block:
    def __init__(self, width, height):
        self.width = width
        self.height = height



Studying().study()
