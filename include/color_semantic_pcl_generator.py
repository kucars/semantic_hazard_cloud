from __future__ import division
from __future__ import print_function
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from enum import Enum
#import time


class ColorPclSemanticGenerator:
    '''
    Generate a ros point cloud given a color image and a depth image
    \author Xuan Zhang
    \date May - July 2018
    '''
    def __init__(self, intrinsic, width = 640, height = 480, frame_id = "/camera_rgb_optical_frame"):
        '''
        width: (int) width of input images
        height: (int) height of input images
        '''
        print ("PCL Color Generation ")
        self.intrinsic = intrinsic
        self.num_semantic_colors = 3 # Number of semantic colors to be sent in the message
        # Allocate arrays
        x_index = np.array([range(width)*height], dtype = '<f4')
        y_index = np.array([[i]*width for i in range(height)], dtype = '<f4').ravel()
        self.xy_index = np.vstack((x_index, y_index)).T # x,y
        self.xyd_vect = np.zeros([width*height, 3], dtype = '<f4') # x,y,depth
        self.XYZ_vect = np.zeros([width*height, 3], dtype = '<f4') # real world coord
        self.ros_data = np.ones([width*height, 8], dtype = '<f4') # [x,y,z,0,bgr0,0,0,0] or [x,y,z,0,bgr0,semantics,confidence,0]
        self.bgr0_vect = np.zeros([width*height, 4], dtype = '<u1') #bgr0
        self.semantic_color_vect = np.zeros([width*height, 4], dtype = '<u1') #bgr0
        self.semantic_colors_vect = np.zeros([width*height, 4 * self.num_semantic_colors], dtype = '<u1') #bgr0bgr0bgr0 ...
        #self.confidences_vect = np.zeros([width*height, self.num_semantic_colors],dtype = '<f4') # class confidences
        # Prepare ros cloud msg
        # Cloud data is serialized into a contiguous buffer, set fields to specify offsets in buffer
        self.cloud_ros = PointCloud2()
        self.cloud_ros.header.frame_id = frame_id
        self.cloud_ros.height = 1
        self.cloud_ros.width = width*height
        self.cloud_ros.fields.append(PointField(
                             name = "x",
                             offset = 0,
                             datatype = PointField.FLOAT32, count = 1))
        self.cloud_ros.fields.append(PointField(
                             name = "y",
                             offset = 4,
                             datatype = PointField.FLOAT32, count = 1))
        self.cloud_ros.fields.append(PointField(
                             name = "z",
                             offset = 8,
                             datatype = PointField.FLOAT32, count = 1))
        self.cloud_ros.fields.append(PointField(
                             name = "rgb",
                             offset = 16,
                             datatype = PointField.FLOAT32, count = 1))
        self.cloud_ros.fields.append(PointField(
                            name = "semantic_color",
                            offset = 20,
                            datatype = PointField.FLOAT32, count = 1))
        

        self.cloud_ros.point_step = 8 * 4 # In bytes
        self.cloud_ros.row_step = self.cloud_ros.point_step * self.cloud_ros.width * self.cloud_ros.height
        self.cloud_ros.is_dense = False

    def generate_cloud_data_common(self, bgr_img, depth_img):
        """
        Do depth registration, suppose that rgb_img and depth_img has the same intrinsic
        \param bgr_img (numpy array bgr8)
        \param depth_img (numpy array float32 2d)
        [x, y, Z] = [X, Y, Z] * intrinsic.T
        """
        bgr_img = bgr_img.view('<u1')
        depth_img = depth_img.view('<f4')
        # Add depth information
        self.xyd_vect[:,0:2] = self.xy_index * depth_img.reshape(-1,1)
        self.xyd_vect[:,2:3] = depth_img.reshape(-1,1)
        self.XYZ_vect = self.xyd_vect.dot(self.intrinsic.I.T)
        # Convert to ROS point cloud message in a vectorialized manner
        # ros msg data: [x,y,z,0,bgr0,0,0,0,color0,color1,color2,0,confidence0,confidence1,confidenc2,0] (little endian float32)
        # Transform color
        self.bgr0_vect[:,0:1] = bgr_img[:,:,0].reshape(-1,1)
        self.bgr0_vect[:,1:2] = bgr_img[:,:,1].reshape(-1,1)
        self.bgr0_vect[:,2:3] = bgr_img[:,:,2].reshape(-1,1)
        # Concatenate data
        self.ros_data[:,0:3] = self.XYZ_vect
        self.ros_data[:,4:5] = self.bgr0_vect.view('<f4')

    def make_ros_cloud(self, stamp):
        # Assign data to ros msg
        # We should send directly in bytes, send in as a list is too slow, numpy tobytes is too slow, takes 0.3s.
        self.cloud_ros.data = np.getbuffer(self.ros_data.ravel())[:]
        self.cloud_ros.header.stamp = stamp
        return self.cloud_ros

    def generate_cloud(bgr_img, depth_img, point_type):
        stamp = self.cloud_ros.header.stamp


    def generate_cloud_color(self, bgr_img, depth_img, stamp):
        """
        Generate color point cloud
        \param bgr_img (numpy array bgr8) input color image
        \param depth_img (numpy array float32) input depth image
        """
        self.generate_cloud_data_common(bgr_img, depth_img)
        return self.make_ros_cloud(stamp)
    '''
    def generate_cloud_semantic_max(self, bgr_img, depth_img, semantic_color, stamp):
        self.generate_cloud_data_common(bgr_img, depth_img)
        #Transform semantic color
        self.semantic_color_vect[:,0:1] = semantic_color[:,:,0].reshape(-1,1)
        self.semantic_color_vect[:,1:2] = semantic_color[:,:,1].reshape(-1,1)
        self.semantic_color_vect[:,2:3] = semantic_color[:,:,2].reshape(-1,1)
        # Concatenate data
        self.ros_data[:,5:6] = self.semantic_color_vect.view('<f4')
        #self.ros_data[:,6:7] = confidence.reshape(-1,1)
        return self.make_ros_cloud(stamp)
    '''
    def generate_cloud_semantic(self, bgr_img, depth_img, semantic_color, stamp):
        self.generate_cloud_data_common(bgr_img, depth_img)
        #Transform semantic color
        self.semantic_color_vect[:,0:1] = semantic_color[:,:,0].reshape(-1,1)
        self.semantic_color_vect[:,1:2] = semantic_color[:,:,1].reshape(-1,1)
        self.semantic_color_vect[:,2:3] = semantic_color[:,:,2].reshape(-1,1)
        # Concatenate data 
        #self.ros_data[:,4:5] = self.bgr0_vect.view('<f4')
        #self.ros_data[:,4:5] =self.semantic_color_vect.view('<f4')
        self.ros_data[:,5:6] = self.semantic_color_vect.view('<f4')

        return self.make_ros_cloud(stamp)
    
  
# Test
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from skimage import io
    import time
    # Init ros
    rospy.init_node('pcl_test', anonymous=True)
    pcl_pub = rospy.Publisher("pcl_test",PointCloud2, queue_size = 1)
    # Read test images
    color_img = io.imread("../../pcl_test/color_image.png")
    depth_img = io.imread("../../pcl_test/depth_image.tiff")
    # Show test input images
    plt.ion()
    plt.show()
    plt.subplot(1,2,1), plt.imshow(color_img[:,:,::-1]), plt.title("color")
    plt.subplot(1,2,2), plt.imshow(depth_img), plt.title("depth")
    plt.draw()
    plt.pause(0.001)
    # Camera intrinsic matrix
    fx = 544.771755
    fy = 546.966312
    cx = 322.376103
    cy = 245.357925
    intrinsic = np.matrix([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype = np.float32)
    # Declare color point cloud generator
    cloud_gen = ColorPclSemanticGenerator(intrinsic,color_img.shape[1], color_img.shape[0])
    print("intrinsic matrix", intrinsic)
    # Generate point cloud and pulish ros message
    while not rospy.is_shutdown():
        since = time.time()
        cloud_ros = cloud_gen.generate_cloud_color(color_img, depth_img, cloud_gen.cloud_ros.header.stamp)
        pcl_pub.publish(cloud_ros)
        print("Generate and publish pcl took", time.time() - since)
    rospy.spin()
