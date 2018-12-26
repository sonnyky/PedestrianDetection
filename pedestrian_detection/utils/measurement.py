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

    def __init__(self):
        print('Initiated measurement module')
    def calc_vertical_angle(self):
        phi = self.image_pos_x * (self.h_fov/self.image_width)
        return phi

    def test_call_method(self):
        print('This is called from measurement instance')