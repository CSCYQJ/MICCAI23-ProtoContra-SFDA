import argparse
import os
import shutil
import time
from trainers import SourceDomainTrainer,PFA_trainer,CL_trainer
import json
import glob
import itertools

from options import get_options

#CUDA_VISIBLE_DEVICES

def ensure_dirs(opt):
    checkpoints_dir = opt['checkpoints_dir']
    
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        
    curr_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    exp_name = 'exp_{}_time_{}'.format(len(os.listdir(checkpoints_dir)),curr_time)

    opt['checkpoint_dir'] = os.path.join(opt['checkpoints_dir'],exp_name)
    checkpoint_dir = opt['checkpoint_dir']
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        with open(os.path.join(checkpoint_dir,'config.json'),'w') as f:
            json.dump(opt,f)
    
    if not os.path.exists(os.path.join(checkpoint_dir,'console_logs')):
        os.makedirs(os.path.join(checkpoint_dir,'console_logs'))

    if not os.path.exists(os.path.join(checkpoint_dir, 'tf_logs')):
        os.makedirs(os.path.join(checkpoint_dir, 'tf_logs'))

    if not os.path.exists(os.path.join(checkpoint_dir, 'saved_models')):
        os.makedirs(os.path.join(checkpoint_dir, 'saved_models'))

    if not os.path.exists(os.path.join(checkpoint_dir, 'visuals')):
        os.makedirs(os.path.join(checkpoint_dir, 'visuals'))
        
    if not os.path.exists(os.path.join(checkpoint_dir, 'source_codes')):
        os.makedirs(os.path.join(checkpoint_dir, 'source_codes'))
        
        source_folders = ['.']
        sources_to_save = list(itertools.chain.from_iterable(
            [glob.glob(f'{folder}/*.py') for folder in source_folders]))
        sources_to_save.extend(['./dataloaders', './models','./losses','./trainers','./utils'])
        for source_file in sources_to_save:
            if os.path.isfile(source_file):
                shutil.copy(source_file,os.path.join(checkpoint_dir, 'source_codes'))
            if os.path.isdir(source_file):
                if os.path.exists(os.path.join(checkpoint_dir, 'source_codes', source_file)):
                    os.removedirs(os.path.join(checkpoint_dir, 'source_codes', source_file))
                shutil.copytree(source_file,os.path.join(checkpoint_dir, 'source_codes', source_file),ignore=shutil.ignore_patterns('__pycache__'))
                
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Segmentor on Source Images')
    opt = get_options(parser)
    ensure_dirs(opt)
    # Source Model training
    trainer = SourceDomainTrainer(opt)
    
    # # Target domain adaptaion PFA stage
    # trainer = PFA_trainer(opt)
    
    # # Target domain adaptation CL stage
    # trainer = CL_trainer(opt)
    
    trainer.launch()