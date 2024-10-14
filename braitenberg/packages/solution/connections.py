from typing import Tuple

import numpy as np




def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")

    height = shape[0]
    width = shape[1]

    #offset=0*height//10

    #width_gradient = np.exp(-0.01 * np.abs(np.arange(width) - width//2)) 

    #res[offset:height,:width] = np.tile(width_gradient, (height-offset, 1))
    #res[:, width//2:] *= -1

    # ---

    # Create meshgrid for the coordinates of each point in the 2D array
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Define the coordinates of the bottom-middle point
    bottom_middle_y = height - 1
    bottom_middle_x = width // 2

    # Calculate the Euclidean distance from each point to the bottom-middle
    distance = np.sqrt((x - bottom_middle_x)**2 + (y - bottom_middle_y)**2)

    # Normalize the distance so that the maximum distance (from borders) becomes 0 and bottom-middle becomes 1
    gradient = 1 - (distance / np.max(distance))**0.5

    # Multiply the other side with -1 
    gradient[:, width//2:width] *= -1

    # ---
    return gradient


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")

    height = shape[0]
    width = shape[1]

    #offset=0*height//10
    
    #width_gradient = np.exp(-0.01 * np.abs(np.arange(width) - width//2)) 

    #res[offset:height,:width] = np.tile(width_gradient, (height-offset, 1))
    #res[:, :] *= -1

    # ---

    # Create meshgrid for the coordinates of each point in the 2D array
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Define the coordinates of the bottom-middle point
    bottom_middle_y = height - 1
    bottom_middle_x = width // 2

    # Calculate the Euclidean distance from each point to the bottom-middle
    distance = np.sqrt((x - bottom_middle_x)**2 + (y - bottom_middle_y)**2)

    # Normalize the distance so that the maximum distance (from borders) becomes 0 and bottom-middle becomes 1
    gradient = 1 - (distance / np.max(distance))**0.5

    # Multiply the other side with -1 
    gradient[:, :width//2] *= -1

    # ---
    return gradient
