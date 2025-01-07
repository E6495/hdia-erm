import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.counter = 0

    def listener_callback(self, msg):
        self.get_logger().info('Imagen recibida.')
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            filename = f'image_{self.counter}.jpg'
            cv2.imwrite(filename, cv_image)
            self.get_logger().info(f'Imagen guardada como {filename}')
            self.counter += 1
        except Exception as e:
            self.get_logger().error(f'Error procesando imagen: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Nodo interrumpido por el usuario.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

