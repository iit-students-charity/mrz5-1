import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
from skimage.util.shape import view_as_blocks
import pry;

def initial_data_input():
    print("Enter part's width: ")
    part_width = int(input())

    print("Enter part's height: ")
    part_height= int(input())

    return part_width, part_height


def image_initialize():
    image = imread('simple.png')
    image = (2.0 * image / 1.0) - 1.0
    a = to_blocks(image)

def to_blocks(image):
    image_array = np.array(image)
    image_height = np.size(image_array, 0)
    image_width = np.size(image_array, 1)
    part_width, part_height = initial_data_input()

    blocks = []
    for py in range(image_width // part_width):
        x_blocks = []
        for px in range(image_height // part_height):
            tmp = []
            for y in range(part_height):
                for x in range(part_width):
                    for k in range(3):
                        tmp.append(image_array[y][x][k])
            x_blocks.append(tmp)
        blocks.append(x_blocks)
    return np.array(blocks)


image_initialize()
