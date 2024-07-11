import math
import os.path
import sys
# sys.path.append("/feng_yi_sen/GroundNLQ")
sys.path.append("/home/feng_yi_sen/GroundNLQ-DINO")
from basic_utils import load_jsonl, save_jsonl

# file_list = ["ego4d_nlq_train_v2.jsonl","ego4d_nlq_val_v2.jsonl"]
file_list = ["/home/feng_yi_sen/GroundNLQ-DINO/ego4d_data/goalstep_data/ego4d_goal_step_train_v2.jsonl",
             "/home/feng_yi_sen/GroundNLQ-DINO/ego4d_data/goalstep_data/ego4d_goal_step_val_v2.jsonl"]

data = []

for filename in file_list:
    data.extend(load_jsonl(filename))

save_jsonl(data,"/home/feng_yi_sen/GroundNLQ-DINO/ego4d_data/goalstep_data/ego4d_goal_step_train+val_v2.jsonl")