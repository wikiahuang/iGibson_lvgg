import numpy as np
import tqdm
import os

import argparse


def judge_success(data, collision_th, suc_dis, shortest_traj):
    shortest_dis = 0
    for i in range(len(shortest_traj)-1):
        shortest_dis += np.linalg.norm(shortest_traj[i][:2] - shortest_traj[i+1][:2])
    
    goal = shortest_traj[-1][:2]
    d_0 = np.linalg.norm(data[0][:2] - goal)
    collision_num = 0
    arrive = False
    actual_dis = 0
    lost = False    
    for i,d in enumerate(data):
        d_t = np.linalg.norm(d[:2] - goal)
        if i > 0:
            actual_dis += np.linalg.norm(d[:2] - data[i-1][:2])
        if collision_num >= collision_th:
            break
        if d[3] == 1 :
            collision_num += 1
        if d_t < suc_dis:
            arrive = True
            break

    return arrive, collision_num, shortest_dis, actual_dis, d_0, d_t

def main(traj_to_be_evaluated, shortest_traj_dir):
    scenes_ids = os.listdir(traj_to_be_evaluated)
    arrives = []
    collision_nums = []
    shortest_distance = []
    actual_distance = []
    SPL = []
    SoftSPL = []
    for scene_id in tqdm.tqdm(scenes_ids):
        scene_path = os.path.join(traj_to_be_evaluated, scene_id)
        traj_files = os.listdir(scene_path)
        for traj_file in traj_files:
            traj_path = os.path.join(scene_path, traj_file, "{}").format(traj_file + '.txt')
            if not os.path.exists(traj_path):
                continue
            data = np.loadtxt(traj_path)
            # load shortest traj
            shortest_traj_path = os.path.join(shortest_traj_dir, scene_id, traj_file, "{}").format(traj_file + '.txt')
            shortest_traj = np.loadtxt(shortest_traj_path)
            suc_dis = 1
            arrive, collision_num, shortest_dis, actual_dis, d_0, d_t= judge_success(data, 20, suc_dis, shortest_traj)
            arrives.append(arrive)
            collision_nums.append(collision_num)
            shortest_distance.append(shortest_dis)
            actual_distance.append(actual_dis)
            SPL.append(arrive * shortest_dis / max(actual_dis, shortest_dis))
            SoftSPL.append((1 - d_t / d_0) * shortest_dis / max(actual_dis, shortest_dis))
    print("Overall Results:")
    print("SR:", np.mean(arrives))
    print("SPL:", np.mean(SPL))
    print("SoftSPL:", np.mean(SoftSPL))
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_dir', type=str, required=True, help='Directory containing the trajectories to be evaluated')
    parser.add_argument('--shortest_traj_dir', type=str, required=True, help='Directory containing the shortest trajectories')
    args = parser.parse_args()
    traj_to_be_evaluated = args.traj_dir
    shortest_traj_dir = args.shortest_traj_dir
    main(traj_to_be_evaluated, shortest_traj_dir) 

# methods = ['loc_flona', 'f3_flona2', 'flona', 'F3_A-star_Random', 'F3_A-star_Random0', 'F3_A-star_Random_1', 'finetunedf3_flona', 'flona_plus', 'flona_turn']
# # methods = ['01noisy_loc_flona', '03noisy_loc_flona', '05noisy_loc_flona', '10noisy_loc_flona']
# traj_dir = '/home/user/data/vis_nav/iGibson/igibson/dataset/scenes_117/test/'


# collision_th = [1, 10, 30, 50, 5000]

# # loss_rate = 0
# for c_th in collision_th:
#         print('-------------------------------------')
#         print('collision count th:', c_th)
#         for m in methods:
#             if m != 'flona_plus':
#                 continue
#             data_dir = os.path.join('/home/user/data/vis_nav/iGibson/igibson/examples/scripts/results', m, 'trajectory_turn_15')
#             arrives = []
#             collision_nums = []
#             shortest_diss = []
#             cul_diss = []
#             SPL = []
#             SoftSPL = []
#             # loss_rate = 0
#             for f in os.listdir(data_dir):
#                 # get the shortest trajectory
                
#                 f_splits = f.split('_')
#                 f_splits[-1] = f_splits[-1].split('.')[0]
#                 scene_floor = f_splits[0] + '_' + f_splits[1]
#                 # if scene_floor not in test_scenes2:
#                 #     continue
#                 traj_name = f_splits[2] + '_' + f_splits[3]
#                 shortest_traj_file = os.path.join(traj_dir, scene_floor, traj_name, traj_name+'.npy')
#                 shortest_traj_file = np.load(shortest_traj_file)
#                 # print("shape of shortest traj:", shortest_traj_file.shape)      
#                 # cal
#                 data = np.loadtxt(os.path.join(data_dir, f)) 
#                 arrive, collision_num, shortest_dis, cul_dis, d_0, d_t= judge_success(data, c_th, s_dis, shortest_traj_file)
#                 # if lost :   
#                 #     continue
#                 arrives.append(arrive)
#                 collision_nums.append(collision_num)
#                 shortest_diss.append(shortest_dis)
#                 cul_diss.append(cul_dis)
#                 SPL.append(arrive * shortest_dis / max(cul_dis, shortest_dis))
#                 SoftSPL.append((1 - d_t / d_0) * shortest_dis / max(cul_dis, shortest_dis))
            
            
#             print('method:', m, "-----SR:", np.mean(arrives), "SPL:", np.mean(SPL), "SoftSPL:", np.mean(SoftSPL))
#             # collision_nums_of_success = [collision_nums[i] for i in range(len(arrives)) if arrives[i]]
#             # print('method:', m, "-----mean coll:", np.mean(collision_nums), "min coll:", np.min(collision_nums), "max coll:", np.max(collision_nums))
            
#             # print("loss rate:", loss_rate)  
#             # print('arrive rate:', np.mean(arrives))
#             print('collision mean nums:', np.mean(collision_nums))
            
            
