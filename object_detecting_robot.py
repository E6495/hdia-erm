import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import joblib
import numpy as np
from time import sleep

class ObjectDetectingRobot(Node):
    def __init__(self, model_path):
        super().__init__('object_detecting_robot')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_subscriber_ = self.create_subscription(Image, '/camera/image_raw', self.process_image, 10)
        self.bridge = CvBridge()

        # Cargar el modelo de la red neuronal
        self.model = joblib.load(model_path)
        self.classes = ['semaforo', 'caja']

        self.moving_straight = True  # Indica si el robot se está moviendo hacia adelante

        # Temporizador para mover continuamente
        self.create_timer(0.1, self.move_straight)  # Llama a move_straight cada 100ms

    def process_image(self, msg):
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
            self.turn_right_and_continue()

    def move_straight(self):
        if self.moving_straight:
            msg = Twist()
            msg.linear.x = 0.2  # Velocidad lineal
            msg.angular.z = 0.0  # Sin rotación
            self.publisher_.publish(msg)

    def stop_robot(self, duration):
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.publisher_.publish(stop_msg)
        self.get_logger().info('Robot detenido.')
        sleep(duration)  # Esperar el tiempo indicado
        self.moving_straight = True  # Reanudar movimiento hacia adelante

    def turn_right_and_continue(self):
        turn_msg = Twist()
        turn_msg.linear.x = 0.0
        turn_msg.angular.z = -1.57  # Giro a la derecha (90 grados)
        self.publisher_.publish(turn_msg)
        sleep(2)  # Ajustar duración del giro según sea necesario
        self.moving_straight = True

    def stop_and_shutdown(self):
        self.stop_robot(0)  # Asegurarse de detener el robot
        self.get_logger().info('Apagando nodo.')

def main(args=None):
    rclpy.init(args=args)
    model_path = "modelo_red_neuronal.pkl"  # Ruta al modelo guardado
    node = ObjectDetectingRobot(model_path)

    try:
        rclpy.spin(node)  # Ejecutar el nodo
    except KeyboardInterrupt:
        node.stop_and_shutdown()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

