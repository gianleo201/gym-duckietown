"""
Controller for duckie bot in Duckietown environment
"""

# run "python3 basic_control.py --map-name <map_name>" to start simulation

import time
import sys
import argparse
import math
import numpy as np
import gym
from gym_duckietown.envs import DuckietownEnv
from pyglet.window import key

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="udem1")
parser.add_argument("--no-pause", action="store_true", help="don't pause on failure")
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(map_name=args.map_name, domain_rand=False, draw_bbox=False)
else:
    env = gym.make(args.env_name)

obs = env.reset()
env.render()

total_recompense = 0

VARIABILE_DI_PROVA_INUTILE = "Bene"

sampling_time = 0.1  # not sure about this
prev_dist = 0.0    # distance_error(k-1)
prev_angle = 0.0   # angle_error(k-1)
storage = 0.0  # integrator state ( use this storing variable to simulate the state of integrator)
first = True

while True:

    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rads = lane_pose.angle_rad

    # proportional constant on angle
    k_p_angle = 10
    prop_angle_action = k_p_angle * angle_from_straight_in_rads
    # proportional constant on distance 
    k_p_dist = 15
    prop_dist_action = k_p_dist * distance_to_road_center
    # derivative constant on distance
    k_d_dist = 10
    deriv_dist_action = k_d_dist * (distance_to_road_center - prev_dist)*sampling_time
    # derivative constant on angle
    k_d_angle = 10
    deriv_angle_action = k_d_angle * (angle_from_straight_in_rads - prev_angle)*sampling_time

    # ignore derivative actions on the first loop
    if first:
        deriv_angle_action = 0.0
        deriv_dist_action = 0.0
        first = False

     # driving speed of duckie_bot (positive when robot goes forward)
    driving_speed = 0.25  # up to now errors on angle and dist doesn't affect driving speed

    # New controller only proportional from state space linearized (it works worse)
    lambda_1 = -5
    lambda_2 = -5

    command = ((lambda_1*lambda_2)/(driving_speed*1.4706))*distance_to_road_center+(lambda_1+lambda_2)*angle_from_straight_in_rads


    # angular speed of duckie_bot (positive when the duckie_bot rotate to the left)
    angular_speed = (
        prop_dist_action + deriv_dist_action + prop_angle_action + deriv_angle_action
    ) # also the distance from the center of road affect the angular speed in order to lead duckie_bot toward the center


    # update previous value to gain the incremental ratio in the next loop
    prev_dist = distance_to_road_center
    prev_angle = angle_from_straight_in_rads
    
    # set controls
    obs, recompense, fini, info = env.step([driving_speed, angular_speed])
    total_recompense += recompense

    # prints variations of parameters
    print(
        "dist_err: %.3f, angle_err: %.3f, prop_angle_action: %.3f, prop_dist_action: %.3f, deriv_dist_action: %.3f, deriv_angle action: %.3f, integral_dist_action: %.3f"
        % (distance_to_road_center, angle_from_straight_in_rads, prop_angle_action, prop_dist_action, deriv_dist_action, deriv_angle_action, integral_dist_action)
    )

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
