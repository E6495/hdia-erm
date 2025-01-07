import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class StraightWalker(Node):
    def __init__(self):
        super().__init__('straight_walker')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.move_straight)

    def move_straight(self):
        msg = Twist()
        msg.linear.x = 0.2  # Velocidad lineal
        msg.angular.z = 0.0  # Sin rotación
        self.publisher_.publish(msg)

    def stop_robot(self):
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.publisher_.publish(stop_msg)
        self.get_logger().info('Robot detenido.')

def main(args=None):
    rclpy.init(args=args)
    node = StraightWalker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_robot()  # Detener el robot antes de cerrar el nodo
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

