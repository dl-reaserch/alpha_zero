from az.training.rl_policy_trainer import run_training
from az.models.tensorflow_models.unreal import PolicyValueNet
import os
import smyAlphaZero.go as go
from cProfile import Profile
# make a miniature model for playing on a miniature 7x7 board

data_dir = os.path.join('benchmarks', 'data')

model_file = os.path.join(data_dir, 'Az_Model.json')
ckpt_root = os.path.join(data_dir,'ckpts')
weights_path = os.path.join(data_dir, 'tf_pvnet_weights.hdf5')
outdir = os.path.join(data_dir, 'Az_Model_output')
stats_file = os.path.join(data_dir, 'Az_reinforcement_policy_trainer.prof')

ckpt_path = os.path.join(ckpt_root,'az_ckpt')

model_file = os.path.abspath(model_file)
ckpt_path = os.path.abspath(ckpt_path)
weights_path = os.path.abspath(weights_path)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(ckpt_root):
    os.makedirs(ckpt_root)

network_architecture = {'filters_per_layer': 32, 'layers': 4, 'board': 9, "learning_rate":0.05, 'momentum':0.9,'gpu_model':False}
features = ['board']
model_config = {'model_file':model_file,'ckpt_path':ckpt_path,'weights_path':weights_path}

policyValueNet = PolicyValueNet(features,model_config, **network_architecture)
#if not os.path.exists(weights_path):
    #policyValueNet.save_weights(weights_path)
#policyValueNet.save_ckpts(ckpt_path, policyValueNet.get_global_step())
profile = Profile()
arguments = (model_file, weights_path, outdir, '--learning-rate', '0.001', '--save-every', '2',
            '--game-batch', '20','--iter_start', '0','--iterations', '2000', '--verbose')
profile.runcall(run_training,policyValueNet,arguments)
profile.dump_stats(stats_file)

