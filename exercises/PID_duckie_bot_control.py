"""
Controller for duckie bot in Duckietown environment
"""

# run "python3 basic_control.py --map-name <map_name>" to start simulation
# cd ~/.local/lib/python3.8/site-packages/duckietown_world/data/gd1/maps to show all the maps
import threading
from numpy.linalg.linalg import LinAlgError
from scipy.io import savemat
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
    env = DuckietownEnv(map_name=args.map_name, domain_rand=False, draw_bbox=False, user_tile_start=[2,4])
else:
    env = gym.make(args.env_name)

''' curve dx test config
# start tile = [3,0]
env.start_pose = [[0.63*0.5,0,0.7*0.63],0]
curve sx test config
# start tile = [3,2]
env.start_pose = [[0.63*0.5,0,0.7*0.63],0]
'''

env.start_pose = [[0, 0, 0.63*0.7],0]

obs = env.reset()
env.render()

total_recompense = 0
DEFAULT_STATE = -888
#DEFAULT_SPEED = 0.35
DEFAULT_SPEED = 0.5
STOP_WAITING = 0
STOP_ROTATING = 0
TILE_SIZE = 0.585
sampling_time = env.delta_time # 0.033
prev_dist = 0.0     # distance_error(k-1)
prev_angle = 0.0    # angle_error(k-1)
counter = 0         # handle cycles flow
STATE = DEFAULT_STATE
first = True
driving_speed = DEFAULT_SPEED # driving speed of bot
angular_speed = 0.0   # angular speed of bot
rotation_strength_left = 1.88 #angular speed when turning left
rotation_strength_right = 3.4 #angular speed when turning right

#object from class detector for identifying apriltags
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

# load settings from yaml file
test_images_path = '.'
with open(test_images_path + '/images.yaml', 'r') as stream:
    parameters = yaml.load(stream)

cameraMatrix = np.array(parameters['sample_test']['K']).reshape((3,3))
camera_params = ( cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2] )

'''
#create lane for path following reading the yaml map file
linea=[]
percorso = []
with open(test_images_path + '/VALERIO-8map.yaml', 'r') as stream:
    parametri = yaml.load(stream)

prova = np.array(parametri['lane_pos'])
for elem in prova:
    if(elem[0]=='curva'):
        if(elem[1]=='sx-up'):
            linea = np.array(parametri['curva_sx-up'])
        elif(elem[1]=='dx-up'):
            linea = np.array(parametri['curva_dx-up'])
        elif(elem[1]=='sx-dwn'):
            linea = np.array(parametri['curva_sx-dwn'])
        else:
            linea = np.array(parametri['curva_dx-dwn'])
        for coord in linea:
            coord[0]+= elem[2][0]
            coord[1]+= elem[2][1]
    else:
        if(elem[0]=='vertical'):
            linea = np.array(parametri['linea_verticale'])
        else:
            linea = np.array(parametri['linea_orizzontale'])
        #print(linea)
        #print(elem)
        for coord in linea:
            #print(coord)
            if(elem[1]=='up'): 
                coord[0]+= elem[2][0]
                coord[1]+= elem[2][1]
            else:
                coord[0] = elem[2][0] - coord[0]
                coord[1] = elem[2][1] - coord[1]
    percorso.extend(linea)

#print(prova)
#print(percorso)
'''

# data initialization to compute circle equation
radius_A = [None, None, None]
radius_B = [None, None, None]
pt = None
ff = 0
threashold = None
last_point = None
curvature = 0

# result dict
matlab_res = {"bot_path": [], "real_path": [], "angular_speed": [], "distance_error": [], "angle_error": [], "driving_speed":[]}

while True:
    '''
    min_dist = math.inf
    posx = env.cur_pos[0]/TILE_SIZE
    posy = env.cur_pos[2]/TILE_SIZE - 2

    if posy>-1.5 and posy<4.5:
        if posx<2:
            #if posx<2 and posx>1.5: 
            #    min_dist = posx-1.7
            #else:
            #    min_dist = posx-1.3
            min_dist = posx - 1.3
        else:
            #if posx<5 and posx>4.5:
            #    min_dist = posx-4.7
            #else:
            #    min_dist = posx-4.3
            min_dist = posx - 5.7
    else:
        if posy>4.5:
            #if posy>3 and posy<3.5:
            #    min_dist = posy-3.3
            #else:
            #    min_dist = posy-3.7
            min_dist = posy - 4.7
        elif posy>0:
            min_dist: posy - 0.7
        else:
            #if posy<0 and posy>-0.5:
            #    min_dist = posy+0.3
            #else:
            #    min_dist = posy+0.7
            min_dist = posy+1.7

    
    #if start==len(percorso)-1:
    #    start = 0
    min_dist = math.inf
    posx = env.cur_pos[0]/TILE_SIZE
    posy = env.cur_pos[2]/TILE_SIZE - 2
    for i in range(start, len(percorso)):
        punti = percorso[i]
        dist = math.sqrt((posx-punti[0])*(posx-punti[0])+(posy-punti[1])*(posy-punti[1]))
        if(dist < min_dist):
            min_dist = dist
            #start = i
            if(i!=len(percorso)-1):
                angolo = math.atan2(percorso[i+1][1]-punti[1],percorso[i+1][0]-punti[0]) - env.cur_angle
            else:
                angolo = math.atan2(percorso[0][1]-punti[1],percorso[0][0]-punti[0]) - env.cur_angle
            #angolo = math.atan2(punti[1]-percorso[i-1][1],punti[0]-percorso[i-1][0]) - env.cur_angle
            punto = punti
    print(min_dist, angolo)
    print(posx, posy, punto, env.cur_angle)'''

    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rads = lane_pose.angle_rad

    matlab_res["distance_error"].append(distance_to_road_center)
    matlab_res["angle_error"].append(angle_from_straight_in_rads)

    #print(env.cur_pos, env.cur_angle, "\n")
    #print(lane_pose.dist, lane_pose.angle_rad)
    #print(min_dist, posx, posy)
    #distance_to_road_center = min_dist
    #angle_from_straight_in_rads = angolo

    ppap = env.cur_pos
    matlab_res["bot_path"].append([ppap[0], ppap[2]])

    pt_new = env.closest_curve_point(env.cur_pos, env.cur_angle, mango=True)
    if pt_new != None:
        matlab_res["real_path"].append([pt_new[0][0], pt_new[0][2]])
    if pt_new == None:
        ff = 0
        pt = None
    elif pt == None or not (pt_new[2] == pt[2]).all():
        pt = pt_new
        # nearest curve
        curve = pt[2]
        # find farthest point on curve
        source_point = [pt[0][0], pt[0][2]]
        max_dist = -1
        farthest_point = None
        for p in curve:
            point = [p[0], p[2]]
            current_distance = np.linalg.norm([point[0]-source_point[0], point[1]-source_point[1]])
            if current_distance >= max_dist:
                max_dist = current_distance
                farthest_point = point
        last_point = farthest_point
        # fill A matrix
        radius_A[0] = [curve[0][0], curve[0][2], 1]
        radius_A[1] = [curve[len(curve)-1][0], curve[len(curve)-1][2], 1]
        radius_A[2] = [curve[len(curve)-2][0], curve[len(curve)-2][2], 1]
        # fill B matrix
        radius_B[0] = -(np.square(curve[0][0])+np.square(curve[0][2]))
        radius_B[1] = -(np.square(curve[len(curve)-1][0])+np.square(curve[len(curve)-1][2]))
        radius_B[2] = -(np.square(curve[len(curve)-2][0])+np.square(curve[len(curve)-2][2]))
        # compute circle equation
        try:
            mango = np.linalg.solve(radius_A, radius_B)
            # compute circle radius
            circle_radius = np.sqrt(np.square(mango[0]/2.0)+np.square(mango[1]/2.0)-mango[2])
            # compute circle center
            circle_center = [-(mango[0]/2.0),-(mango[1]/2.0)]
            # compute vector radius
            vector_radius = [source_point[0]-circle_center[0], source_point[1]-circle_center[1]]
            # compute angular speed sign
            tangent = [pt[1][0], pt[1][2]]
            sas = np.sign(np.cross(tangent, vector_radius))

            # compute curvature
            curvature = (circle_radius)**(-1)
            # ...

            if curvature < 0.0001:
                raise LinAlgError("curvature too small")
            if sas == 1:
                threashold = 0.25
            elif sas == -1:
                driving_speed = 0.35
                threashold = 0.12
            # finally compute feedforward action
            ff = 1.5 * sas* (driving_speed / circle_radius)
        except:
            ff = 0
            driving_speed = DEFAULT_SPEED
    elif threashold != None and np.linalg.norm([pt_new[0][0]-last_point[0], pt_new[0][2]-last_point[1]]) <= threashold:
        ff = 0
        threashold = None
        driving_speed = DEFAULT_SPEED

    



    # proportional constant on distance 
    k_p_dist = 15
    prop_dist_action = k_p_dist * distance_to_road_center
    # derivative constant on distance
    k_d_dist = 20
    deriv_dist_action = k_d_dist * (distance_to_road_center - prev_dist)/sampling_time
    

    # ignore derivative actions on the first loop
    if first:
        deriv_angle_action = 0.0
        deriv_dist_action = 0.0
        first = False


    # angular speed of duckie_bot (positive when the duckie_bot rotate to the left)
    angular_speed = (
        prop_dist_action + deriv_dist_action + ff 
    ) # also the distance from the center of road affect the angular speed in order to lead duckie_bot toward the center

    # update previous value to gain the incremental ratio in the next loop
    prev_dist = distance_to_road_center
    prev_angle = angle_from_straight_in_rads
    
    # code executed only every tot frames
    if STATE == DEFAULT_STATE and ((counter % 5) != 0):
        ignore = 0 #if the apriltag is too far, ignore it
        tags = at_detector.detect(cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY), True, camera_params, parameters['sample_test']['tag_size'])
        tag_ids = [tag.tag_id for tag in tags]
        tag_pos = [tag.pose_t[2] for tag in tags]
        if len(tags) > 0:
            # handle multiple tags
            nearest_tag = tags[0]
            for t in tags:
                if t.pose_t[2] < nearest_tag.pose_t[2]:
                    nearest_tag = t
            if nearest_tag.pose_t[2] > 0.36:
                ignore = 1
            if not ignore:
                '''
                print("TAG(S) FOUND AT STEP ",env.unwrapped.step_count,"!")
                print(tag_pos)
                print(len(tags), " tag(s) found: ", tag_ids)
                '''
                STATE = nearest_tag.tag_id
        counter = 0
    elif STATE == 1:
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
        print("giro a destra")
        STOP_ROTATING = 100
        STOP_WAITING = 18
        STATE = -2
    elif STATE == 10:
        print("giro a sinistra")
        STOP_ROTATING = 110
        STOP_WAITING = 23
        STATE = -3
    elif STATE == -2:
        if STOP_WAITING > 0:
            STOP_WAITING -= 1
        else:
            STOP_ROTATING -= rotation_strength_right
            angular_speed = - rotation_strength_right
            if(STOP_ROTATING <= 0):
                STATE = -1
    elif STATE == -3:
        if STOP_WAITING > 0:
            STOP_WAITING -= 1
        else:
            STOP_ROTATING -= rotation_strength_left
            angular_speed = rotation_strength_left
            if(STOP_ROTATING <= 0):
                STATE = -1
    elif STATE == 12:
        driving_speed = 0.18
        STOP_WAITING = 60
        STATE = 0
    elif STATE == 95:
        driving_speed -= 0.015
        if driving_speed <= 0.3:
            STOP_WAITING = 30
            STATE = 0
    elif STATE == 74:
        driving_speed = 0.7
        STOP_WAITING = 50
        STATE = 0
    if counter % 5 == 0:
        counter += 1
    
    # set controls
    obs, recompense, fini, info = env.step([driving_speed, angular_speed])
    total_recompense += recompense

    matlab_res["angular_speed"].append(angular_speed)
    matlab_res["driving_speed"].append(driving_speed)

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
        if symbol == key.BACKSPACE or symbol == key.SLASH:
            env.reset()
            env.render()
            return
        elif symbol == key.T:
            imgage = Image.fromarray(obs)
            imgage.save("shot.png")
        elif symbol == key.ESCAPE:
            env.close()
            savemat("tino.mat", matlab_res)
            sys.exit()

    # should execute the render of the next frame
    env.render()
