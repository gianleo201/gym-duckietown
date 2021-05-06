"""
Controller for duckie bot in Duckietown environment
"""

# run "python3 basic_control.py --map-name <map_name>" to start simulation
# cd ~/.local/lib/python3.8/site-packages/duckietown_world/data/gd1/maps to show all the maps

import time
import sys
import argparse
import math
import numpy as np
import gym
from gym_duckietown.envs import DuckietownEnv
from pyglet.window import key

#libraries for apriltags detection
from dt_apriltags import Detector
from PIL import Image
import yaml
import cv2
from cv2 import imshow

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="udem1")
parser.add_argument("--no-pause", action="store_true", help="don't pause on failure")
args = parser.parse_args()

env: DuckietownEnv

if args.env_name is None:
    env = DuckietownEnv(map_name=args.map_name, domain_rand=False, draw_bbox=False)
else:
    env = gym.make(args.env_name)

obs = env.reset()
env.render()

total_recompense = 0

DEFAULT_STATE = -888
#DEFAULT_SPEED = 0.35
DEFAULT_SPEED = 0.5
STOP_WAITING = 0

sampling_time = 0.1 # not sure about this
prev_dist = 0.0     # distance_error(k-1)
prev_angle = 0.0    # angle_error(k-1)
counter = 0         # handle cycles flow
STATE = DEFAULT_STATE
first = True
driving_speed = DEFAULT_SPEED # driving speed of bot
angular_speed = 0.0   # angular speed of bot
rotation_strength = 2 #angular speed when turning left/right

#object from class detector for identifying apriltags
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.10,
                       debug=0)

# load settings from yaml file
test_images_path = '.'
with open(test_images_path + '/images.yaml', 'r') as stream:
    parameters = yaml.load(stream)

cameraMatrix = np.array(parameters['sample_test']['K']).reshape((3,3))
camera_params = ( cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2] )


while True:

    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rads = lane_pose.angle_rad

    # proportional constant on angle
    k_p_angle = 10
    prop_angle_action = k_p_angle * angle_from_straight_in_rads
    # proportional constant on distance 
    k_p_dist = 12
    prop_dist_action = k_p_dist * distance_to_road_center
    # derivative constant on distance
    k_d_dist = 10
    deriv_dist_action = k_d_dist * (distance_to_road_center - prev_dist)*sampling_time
    # derivative constant on angle
    k_d_angle = 25
    deriv_angle_action = k_d_angle * (angle_from_straight_in_rads - prev_angle)*sampling_time

    # ignore derivative actions on the first loop
    if first:
        deriv_angle_action = 0.0
        deriv_dist_action = 0.0
        first = False

    # New controller only proportional from state space linearized (it works worse)
    lambda_1 = -5
    lambda_2 = -5
    # command = ((lambda_1*lambda_2)/(driving_speed*1.4706))*distance_to_road_center+(lambda_1+lambda_2)*angle_from_straight_in_rads


    # angular speed of duckie_bot (positive when the duckie_bot rotate to the left)
    angular_speed = (
        prop_dist_action + deriv_dist_action + prop_angle_action + deriv_angle_action
    ) # also the distance from the center of road affect the angular speed in order to lead duckie_bot toward the center


    # update previous value to gain the incremental ratio in the next loop
    prev_dist = distance_to_road_center
    prev_angle = angle_from_straight_in_rads
    
    # code executed only every tot frames
    if STATE == -888 and ((counter % 5) != 0):
        # catch apriltags
        # imgage = Image.fromarray(obs)
        # this is the line code that makes the program slower beacuase interacts with hard disk
        # imgage.save("test_image.png")

        # img = cv2.imread(test_images_path+'/'+parameters['sample_test']['file'], cv2.IMREAD_GRAYSCALE)
        ignore = 0 #if the apriltag is too far, ignore it
        tags = at_detector.detect(cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY), True, camera_params, parameters['sample_test']['tag_size'])
        tag_ids = [tag.tag_id for tag in tags]
        tag_pos = [tag.pose_t[1] for tag in tags]
        if len(tags) > 0:
            if(abs(tags[0].pose_t[1])>0.05):
                ignore = 1
            if not ignore:
                print("TAG(S) FOUND AT STEP ",env.unwrapped.step_count,"!")
                print(tag_pos)
                print(len(tags), " tag(s) found: ", tag_ids)
                STATE = tag_ids[0]
        counter = 0
    elif STATE == 1:
        #driving_speed -= 0.005
        driving_speed -= 0.035
        if driving_speed <= 0:
            driving_speed = 0
            STOP_WAITING = 20
            STATE = 0
    elif STATE == 0:
        if (STOP_WAITING == 0):
            STATE = -1
        else:
            STOP_WAITING -= 1
    elif STATE == -1:
        driving_speed = DEFAULT_SPEED
        STATE = DEFAULT_STATE
    elif STATE == 11:
        driving_speed = 0.4
        STATE = DEFAULT_STATE
    elif STATE == 9:
        STOP_ROTATING = 120
        STATE = -2
    elif STATE == 10:
        STOP_ROTATING = 120
        STATE = -3
    elif STATE == -2:
        STOP_ROTATING -= rotation_strength
        angular_speed = prev_angle - rotation_strength
        if(STOP_ROTATING == 0):
            STATE = -1
    elif STATE == -3:
        STOP_ROTATING -= rotation_strength
        angular_speed = prev_angle + rotation_strength
        if(STOP_ROTATING == 0):
            STATE = -1
    if counter % 5 == 0:
        counter += 1
    
    # set controls
    obs, recompense, fini, info = env.step([driving_speed, angular_speed])
    total_recompense += recompense

    # prints variations of parameters
    """
    print(
        "dist_err: %.3f, angle_err: %.3f, prop_angle_action: %.3f, prop_dist_action: %.3f, deriv_dist_action: %.3f, deriv_angle action: %.3f"
        % (distance_to_road_center, angle_from_straight_in_rads, prop_angle_action, prop_dist_action, deriv_dist_action, deriv_angle_action, )
    )
    """

    #enables user to reset or exit the simulation
    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
    	"""
    	This handler processes keyboard commands that control the simulation
    	"""

    	if symbol == key.BACKSPACE or symbol == key.SLASH:
        	env.reset()
        	env.render()
        	return
    	elif symbol == key.ESCAPE:
        	env.close()
        	sys.exit(0)

    # should execute the render of the next frame
    env.render()
