import os
import yaml
import argparse

def get_options(parser):

    ## Config file
    parser.add_argument("--config_file",type=str)
    parser.add_argument("--gpu_id", default=0, type=int)
    
    
    opt = vars(parser.parse_args())
    with open(opt['config_file']) as f:
        config = yaml.safe_load(f)
        
    opt.update(config)
    opt["gpu_id"] = "cuda:%s"%opt["gpu_id"]
    opt['checkpoints_dir'] = os.path.join(opt['save_root'],opt['experiment_name'])
    opt['img_size'] = tuple(opt['img_size'])

 
    return opt