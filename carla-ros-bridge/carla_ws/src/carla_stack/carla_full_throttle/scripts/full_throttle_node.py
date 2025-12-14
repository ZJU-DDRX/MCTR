#!/usr/bin/env python

import rospy
from carla_msgs.msg import CarlaEgoVehicleControl

def main():
    rospy.init_node('full_throttle_controller', anonymous=True)
    pub = rospy.Publisher(
        '/carla/ego_vehicle/vehicle_control_cmd',
        CarlaEgoVehicleControl,
        queue_size=10
    )
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        msg = CarlaEgoVehicleControl()
        msg.throttle = 1.0
        msg.steer = 0.0
        msg.brake = 0.0
        msg.hand_brake = False
        msg.reverse = False
        pub.publish(msg)
        rospy.loginfo("Published throttle=1.0")
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

