import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
from skimage.util.shape import view_as_blocks
import pry;


class Studying:
    def __init__(self):
        self.image_initialize()
        self.block_initialize()
        self.error_max = 10000
        self.current_error = self.error_max + 1
        self.max_alpha = 0.01
        self.epoch = 0
        self.y_values = []

    def study(self):
        self.set_neural_numbers_on_second_layer()
        self.set_neural_layers(self.pixels_number())
        block_form = self.image.to_blocks(self.block.width, self.block.height)
        while self.current_error > 10000:
            self.current_error = 0
            self.epoch += 1

            for block in range(np.size(block_form, 0)):
                y = np.matmul(block_form[block], self.first_layer)
                block_on_second_layer = np.matmul(y, self.second_layer)
                delta = np.subtract(block_on_second_layer, block_form[block])
                self.modify_layer_weight(delta[block])

                self.learn_neurons_on_first_layer(block_form[block], delta)
                self.adjustment_weight_on_second_layer(y, delta)

            print('Epoch ', self.epoch, '   ', 'error ', self.current_error)

        self.show_image(block_form)

    def show_image(self, blocks):
        self.show_image_from_blocks(self.image.restore_image(blocks, self.block.width, self.block.height))


    def show_image_from_blocks(self, blocks):
            plt.imshow(blocks)
            plt.axis('off')
            plt.show()

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
        self.error_max = 0.1 * self.neurons_number

    def set_neural_layers(self, pixels_number):
        self.first_layer = np.random.rand(self.block.width * self.block.height * 3, self.neurons_number) * 2 - 1
        self.second_layer = np.copy(self.first_layer).transpose()

    def modify_layer_weight(self, delta):
        for i in range(np.size(delta, 0)):
            self.current_error += delta[i] * delta[i]

    def adjustment_weight_on_second_layer(self, y, delta):
        self.second_layer -= self.learning_ratio(y) * y.transpose() @ delta

    def learning_ratio(self, block):
        elements_sum = sum(np.matmul(element, element) for element in block)

        return self.max_alpha if elements_sum == 0 else (1 / elements_sum)

    def learn_neurons_on_first_layer(self, block, delta):
        self.first_layer -= self.learning_ratio(block) * np.matmul(np.matmul(block.transpose(), delta), self.second_layer.transpose())

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
                            tmp.append(self.image_array[py * block_height + y, px * block_width + x, k])
                x_blocks.append(tmp)
            blocks.append(x_blocks)
        self.block_array = np.array(blocks)
        return self.block_array

    def to_image(self, blocks, block_width, block_height):
        array = []
        blocks_in_line = self.width // block_width
        for i in range(self.height // block_height):
            for y in range(block_height):
                line = []
                for j in range(blocks_in_line):
                    for x in range(block_width):
                        pixel = []
                        for color in range(3):
                            pixel.append(blocks[i, (y * block_width * 3) + (x * 3) + color])
                        line.append(pixel)
                array.append(line)
        return np.array(array)

    def restore_image(self, blocks, block_width, block_height):
        blocks = 1 * (self.to_image(blocks, block_width, block_height) + 1) / 2
        return blocks.reshape(256, 256, 3)


class Block:
    def __init__(self, width, height):
        self.width = width
        self.height = height



Studying().study()
