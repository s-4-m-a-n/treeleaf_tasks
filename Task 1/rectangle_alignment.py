import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

# config------------------------------
IMAGE_PATH = "image.png"
# first rectangle image size 
# is retrieved manually through ms.pain
RECTANGLE_FIRST_SIZE = (200,250)
# coordinate of two endpoints of the base of each rectangles
BASE_LINES = [[(50,75),(209,118)],[(359, 134,), (505,95)],[(203, 291),(56, 375)],[(328, 343), (489, 435)]]


# helper functions------------------------------
def rotate_image(image, angle):  #rotate the image about the given angle around the center of the image
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1],
                            flags=cv2.INTER_LINEAR,
                            borderMode = cv2.BORDER_CONSTANT,
                            borderValue=255)
 
    return result

def get_angle(point1, point2): #return the angle between two points
    base = point2[0] - point1[0]
    perpendicular = point2[1] - point1[1]
    theta = math.degrees(math.atan(perpendicular/base))
    return theta

def align_rectangle(rectangle_images, base_lines_coord):
    """It takes a list of cropped rectangles and a list of each rectangle base lines 
    and returns the list of aligned rectangle images."""
    aligned_rectangles = []
    for rect_img, base_coord in zip(rectangle_images,base_lines_coord):
        theta = get_angle(*base_coord)
        aligned_rectangles.append(rotate_image(rect_img,theta))  
    return aligned_rectangles


def merge_images(final_image_shape,
                 list_images,
                 rectangle_first_size=RECTANGLE_FIRST_SIZE):
    """It merges the given list of aligned rectangular images.
    It takes the shape of the image after merged and list of images"""
    
    merged_image = np.ones(final_image_shape) * 255 # initialize all the pixels are white, it is not mandatory
    merged_image[:rectangle_first_size[0],
                 :rectangle_first_size[1]] = list_images[0]
    merged_image[:rectangle_first_size[0],
                 rectangle_first_size[1]:] = list_images[1]
    merged_image[rectangle_first_size[0]:,
                 :rectangle_first_size[1]] = list_images[2]
    merged_image[rectangle_first_size[0]:,
                 rectangle_first_size[1]:] = list_images[3]
    return merged_image



# main method----------------------------------
def run(show_image = True, save_image = True): 
    # loading image
    image = cv2.imread(IMAGE_PATH)
    # transforming into gray image
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    # cropping the image into four different image each of them contains each rectangle of the given figure
    rectangle_first_image = gray_image[:RECTANGLE_FIRST_SIZE[0],:RECTANGLE_FIRST_SIZE[1]]
    rectangle_second_image = gray_image[:RECTANGLE_FIRST_SIZE[0],RECTANGLE_FIRST_SIZE[1]:]
    rectangle_third_image = gray_image[RECTANGLE_FIRST_SIZE[0]:,:RECTANGLE_FIRST_SIZE[1]]
    rectangle_fourth_image = gray_image[RECTANGLE_FIRST_SIZE[0]:,RECTANGLE_FIRST_SIZE[1]:]
    
    rect_images = [rectangle_first_image, rectangle_second_image, rectangle_third_image, rectangle_fourth_image]
    
    # align rectangles
    aligned_rects = align_rectangle(rect_images,BASE_LINES)
    
    # merged all the images of aligned rectangles
    final_image = merge_images(gray_image.shape,aligned_rects)
    
    # save the final aligned image
    if save_image:
        cv2.imwrite("aligned_rectangles.png",final_image)
    
    # display image
    if show_image:
        plt.imshow(final_image,cmap='gray')
        plt.title("aligned images")
        plt.show()
    
    
if __name__ == '__main__':
    run()
    
