import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import joblib
import numpy as np

class ObjectDetectingRobot(Node):
    def __init__(self, model_path):
        super().__init__('object_detecting_robot')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_subscriber_ = self.create_subscription(Image, '/camera/image_raw', self.process_image, 10)
        self.bridge = CvBridge()

        # Cargar el modelo de la red neuronal
        self.model = joblib.load(model_path)
        self.classes = ['semaforo', 'caja']

        # Estados del robot
        self.state = 'moving'  # Estados: moving, stopping
        self.stop_timer = None

        # Temporizador para movimiento continuo
        self.create_timer(0.1, self.move_robot)

    def process_image(self, msg):
        if self.state != 'moving':
            return  # Ignorar imágenes si el robot no está avanzando

        # Convertir la imagen ROS a OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Preprocesar la imagen para la red neuronal
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        flattened = resized.flatten().reshape(1, -1)

        # Predecir el objeto
        prediction = self.model.predict(flattened)[0]
        object_detected = self.classes[prediction]
        self.get_logger().info(f'Objeto detectado: {object_detected}')

        # Tomar acciones según el objeto detectado
        if object_detected == 'semaforo':
            self.stop_robot(10)  # Detener por 10 segundos
        elif object_detected == 'caja':
            self.get_logger().info('Caja detectada, continuando avance.')

    def move_robot(self):
        msg = Twist()

        if self.state == 'moving':
            msg.linear.x = 0.2  # Avanzar hacia adelante
            msg.angular.z = 0.0  # Sin rotación

        elif self.state == 'stopping':
            msg.linear.x = 0.0  # Detener
            msg.angular.z = 0.0
            if self.stop_timer and self.get_clock().now() > self.stop_timer:
                self.state = 'moving'

        self.publisher_.publish(msg)

    def stop_robot(self, duration):
        self.get_logger().info('Deteniendo robot...')
        self.state = 'stopping'
        self.stop_timer = self.get_clock().now() + rclpy.time.Duration(seconds=duration)

def main(args=None):
    rclpy.init(args=args)
    model_path = "modelo_red_neuronal.pkl"  # Ruta al modelo guardado
    node = ObjectDetectingRobot(model_path)

    try:
        rclpy.spin(node)  # Ejecutar el nodo
    except KeyboardInterrupt:
        node.get_logger().info('Apagando nodo...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

