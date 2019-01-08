from math import sin, cos, tan

class measurement:

    # provided values
    image_width = 640
    image_height = 480
    camera_pitch = 70
    camera_height = 1000
    h_fov = 69.4 # as provided by the specifications at : https://click.intel.com/intelr-realsensetm-depth-camera-d415.html
    v_fov = 42.5 # as provided by the specifications at : https://click.intel.com/intelr-realsensetm-depth-camera-d415.html

    # values from image processing. position of detected person in the image. initialize to zeros.
    image_pos_x = 0
    image_pos_y = 0

    three_d_point = [0,0,0]

    def __init__(self):
        print('Initiated measurement module')

    def set_image_positions(self, pos_x, pos_y):
        self.image_pos_x = pos_x
        self.image_pos_y = pos_y
        #print('the value of image_pos_x is : ', self.image_pos_x)
        #print('the value of image_pos_y is : ', self.image_pos_y)

    def calc_horizontal_angle(self):
        phi = self.image_pos_x * (self.h_fov/self.image_width)
        return phi

    def calc_vertical_angle(self):
        psi = self.camera_pitch + self.image_pos_y * (self.v_fov/self.image_height)
        return psi

    def calc_3d_position(self, vertical_angle, horizontal_angle):
        self.three_d_point[1] = self.camera_height * tan(vertical_angle)
        self.three_d_point[0] = self.three_d_point[1] * tan(horizontal_angle)
        self.three_d_point[2] = (self.camera_height * self.camera_height) + (self.three_d_point[0] * self.three_d_point[0]) + (self.three_d_point[1] * self.three_d_point[1])
        print("calculating 3d position inside method")
        return self.three_d_point

    def test_call_method(self, x, y):
        print('the value of x is : ', x)
        print('the value of y is : ', y)
        print('This is called from measurement instance')