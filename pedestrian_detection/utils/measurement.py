from math import sin, cos, tan, radians, sqrt

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

        # adjust image coordinates so the image center becomes the reference
        self.image_pos_x = pos_x - (self.image_width/2)
        self.image_pos_y = (self.image_height/2) - pos_y

    def set_camera_parameters(self, h_fov_input, v_fov_input):
        self.h_fov = h_fov_input
        self.v_fov = v_fov_input

    def set_camera_pitch_and_height(self, pitch, height):
        self.camera_height = height
        self.camera_pitch = pitch

    def calc_horizontal_angle(self):
        phi = self.image_pos_x * (self.h_fov/self.image_width)
        return phi

    def calc_vertical_angle(self):
        psi = self.camera_pitch + self.image_pos_y * (self.v_fov/self.image_height)
        return psi

    def calc_3d_position(self, vertical_angle, horizontal_angle):
        self.three_d_point[1] = self.camera_height * tan(radians(vertical_angle))
        self.three_d_point[0] = self.three_d_point[1] * tan(radians(horizontal_angle))
        self.three_d_point[2] = sqrt((self.camera_height * self.camera_height) + (self.three_d_point[0] * self.three_d_point[0]) + (self.three_d_point[1] * self.three_d_point[1]))

        return self.three_d_point

    def test_call_method(self, x, y):
        print('the value of x is : ', x)
        print('the value of y is : ', y)
        print('This is called from measurement instance')

    def calc_height_object_on_floor(self, floor_y, object_top_y):
        obj_height = self.camera_height * ((object_top_y - floor_y)/object_top_y)
        return obj_height