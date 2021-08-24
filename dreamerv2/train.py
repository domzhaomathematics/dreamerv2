import collections
import functools
import logging
import os
import pathlib
import sys
import warnings

try:
  import rich.traceback
  rich.traceback.install()
except ImportError:
  pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#set environment variable to 3
logging.getLogger().setLevel('ERROR')
#Any event below won't be logged [NOTSET,DEBUG,INFO,WARNING,ERROR,CRITICAL]
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
#ignore this specific python warning

sys.path.append(str(pathlib.Path(__file__).parent))
#sys.path is a list of paths that point (contains) to modules that we want to import
#If the modules are in different directories (and not in python paths),
#We have to add to this list
sys.path.append(str(pathlib.Path(__file__).parent.parent))
#pathlib.Path(__file__).parent points to the current files parent directory
import numpy as np
import ruamel.yaml as yaml
import tensorflow as tf

import agent
import elements
import common


configs = pathlib.Path(sys.argv[0]).parent / 'configs.yaml'
#sys.argv[0] is the script name, if script.py calls train.py (_file_), sys.argv[0] is script.py
#in the directory with the script parent/script.py, there should be a configs.yaml 
# so we have parent/configs.yaml
configs = yaml.safe_load(configs.read_text())
#transform the configs into a dictionary. 
#pathlib.Path.read_text() creates a string of the file the path is reffering to
config = elements.Config(configs['defaults'])
# *args: pass varying number of positional arguments (for x in args)
#**kwargs accepts keyword arguments a=2,b=3,c=4
#elements.config takes in either iterable (tuples key-value) or actual dict
#creates a config object with defaults from the Config class in elements.config
parsed, remaining = elements.FlagParser(configs=['defaults']).parse_known(
    exit_on_help=False)
for name in parsed.configs:
  config = config.update(configs[name])
config = elements.FlagParser(config).parse(remaining)
logdir = pathlib.Path(config.logdir).expanduser()
#where the logs are outputed in
config = config.update(
    steps=config.steps // config.action_repeat,
    eval_every=config.eval_every // config.action_repeat,
    log_every=config.log_every // config.action_repeat,
    time_limit=config.time_limit // config.action_repeat,
    prefill=config.prefill // config.action_repeat)

tf.config.experimental_run_functions_eagerly(not config.jit)
#jit is a config, config.jit is for "not run eagerly".
#If true, will make all invocations of tf.function run eagearly instead of a graph
message = 'No GPU found. To actually train on CPU remove this assert.'
assert tf.config.experimental.list_physical_devices('GPU'), message
#list of all physical devices that are GPUs, if None (False) stop and outputs the message
for gpu in tf.config.experimental.list_physical_devices('GPU'):
  #iterate through all the possible gpus
  tf.config.experimental.set_memory_growth(gpu, True)
  # Runtime initialization will not allocate all memory on the device
assert config.precision in (16, 32), config.precision
#check if precision is set to 16 or 32. stops and output the precision
if config.precision == 16:
  from tensorflow.keras.mixed_precision import experimental as prec
  prec.set_policy(prec.Policy('mixed_float16'))
  #precision of floats

print('Logdir', logdir)
train_replay = common.Replay(logdir / 'train_replay', config.replay_size)
#initialize the replay buffer, save the episodes in logdir/train_replay, max size replay_size
eval_replay = common.Replay(logdir / 'eval_replay', config.time_limit or 1)
#initialize the replay buffer, save the episodes in logdir/eval_replay , max time limit (1 if 0)
step = elements.Counter(train_replay.total_steps)
#creates a counter object, initialized with the number of steps already in the replay buffer
outputs = [
    elements.TerminalOutput(),
    elements.JSONLOutput(logdir),
    elements.TensorBoardOutput(logdir),
]
#outputs in the terminal, json format in logidr and in tensorboard in logdir

logger = elements.Logger(step, outputs, multiplier=config.action_repeat)
#initialize the logger, desired outputs and actions per step
metrics = collections.defaultdict(list)
#default dictionary with list values
should_train = elements.Every(config.train_every)
#object that tracks when the model should be trained (every x steps)
#the object keeps in self._last, which is the last step where it was trained
should_log = elements.Every(config.log_every)
#tracks last time it was logged
should_video_train = elements.Every(config.eval_every)
#tracks last time it was train on video (FIND OUT WHAT THIS IS)
should_video_eval = elements.Every(config.eval_every)
#tracks last time it was evaluated on video (FIND OUT WHAT THIS IS)

def make_env(mode):
  suite, task = config.task.split('_', 1)
  #splits the task (first _)config atari_pong -> atari,pong  dmc_walker_walk-> dmc, walker_walk
  if suite == 'dmc':
    env = common.DMC(task, config.action_repeat, config.image_size)
    #DMC env object , domain, task <- walker, walk
    # from dm_control , we get env by dm_control.suite.load(domain,task)
    #we set image size and action repeat
    #when given an action, we can repeat multiple times if action_repeat>1
    env = common.NormalizeAction(env)
    #wraps around the env an normalize the actions for continuous action (denormalize at step)
  elif suite == 'atari':
    env = common.Atari(
        task, config.action_repeat, config.image_size, config.grayscale,
        life_done=False, sticky_actions=True, all_actions=True)
    #the task (Pong), number of times you do the action per step, image size and grayscale
    #All possible actions True
    env = common.OneHotAction(env)
    #have the actions one hot encoded instead of the true label
  else:
    raise NotImplementedError(suite)
  env = common.TimeLimit(env, config.time_limit)
  #puts a time limit on the env. If steps exceeds time_limit, next step will display error
  env = common.RewardObs(env)
  #reward is put as an observation (obs["reward"]), can reset reward to 0 with reset()
  env = common.ResetObs(env)
  #in observation, there's a reset value (true or false) , obs["reset"]
  # each step returns obs,reward,done,info, with obs["reward"],obs["reset"]
  return env

def per_episode(ep, mode):
  length = len(ep['reward']) - 1
  #check the length of the episode, the trajectory, every reward logged
  score = float(ep['reward'].astype(np.float64).sum())
  # Sum all reward to get the total reward of the trajectory
  print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
  replay_ = dict(train=train_replay, eval=eval_replay)[mode]
  #create a dict with key "train" with the dict train_replay, same for eval
  #chooses the appriopriate key for the mode
  replay_.add(ep)
  #add episode to buffer, handle if exceeds limit and save
  logger.scalar(f'{mode}_transitions', replay_.num_transitions)
  logger.scalar(f'{mode}_return', score)
  logger.scalar(f'{mode}_length', length)
  logger.scalar(f'{mode}_eps', replay_.num_episodes)
  should = {'train': should_video_train, 'eval': should_video_eval}[mode]
  if should(step):
    logger.video(f'{mode}_policy', ep['image'])
  logger.write()

print('Create envs.')
train_envs = [make_env('train') for _ in range(config.num_envs)]
#list of train environments, generally one
eval_envs = [make_env('eval') for _ in range(config.num_envs)
#list of eval environments, generally one
action_space = train_envs[0].action_space['action']
#get the action space depending on config (one hot or Normalized)
#for atari, n actions would have an action space of shape n, each with values between 0 and 1
train_driver = common.Driver(train_envs)
train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
train_driver.on_step(lambda _: step.increment())
eval_driver = common.Driver(eval_envs)
eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
#initialize the drivers

prefill = max(0, config.prefill - train_replay.total_steps)
#how much to prefill the replay buffer, replay.total_steps is the number already in there
if prefill:
  print(f'Prefill dataset ({prefill} steps).')
  #prefil the memory buffer with episodes following random actions
  random_agent = common.RandomAgent(action_space)
  train_driver(random_agent, steps=prefill, episodes=1)
  #perform the number of steps (and episodes) according to the driver logic (see driver.py)
  # Roughly speaking, it performs the steps and do the necessary functions per step, episode and on reset
  #the train_driver was intialized in a way so that each episode is saved properly with per_episode()
  eval_driver(random_agent, episodes=1)
  #no distiction between train and eval, just the environment that is different
  train_driver.reset()
  #rests completely the driver, like new for the step
  eval_driver.reset()

print('Create agent.')
train_dataset = iter(train_replay.dataset(**config.dataset))
#in config dataset: {batch: 50, length: 50, oversample_ends: True}
#create a dataset that we can sample from with these values
eval_dataset = iter(eval_replay.dataset(**config.dataset))
#create an eval dataset with episodes that the agent was never trained on.
agnt = agent.Agent(config, logger, action_space, step, train_dataset)
if (logdir / 'variables.pkl').exists():
  agnt.load(logdir / 'variables.pkl')
else:
  config.pretrain and print('Pretrain agent.')
  for _ in range(config.pretrain):
    agnt.train(next(train_dataset))

def train_step(tran):
  if should_train(step):
    for _ in range(config.train_steps):
      _, mets = agnt.train(next(train_dataset))
      [metrics[key].append(value) for key, value in mets.items()]
  if should_log(step):
    for name, values in metrics.items():
      logger.scalar(name, np.array(values, np.float64).mean())
      metrics[name].clear()
    logger.add(agnt.report(next(train_dataset)), prefix='train')
    logger.write(fps=True)
train_driver.on_step(train_step)

while step < config.steps:
  logger.write()
  print('Start evaluation.')
  logger.add(agnt.report(next(eval_dataset)), prefix='eval')
  eval_policy = functools.partial(agnt.policy, mode='eval')
  eval_driver(eval_policy, episodes=config.eval_eps)
  print('Start training.')
  train_driver(agnt.policy, steps=config.eval_every)
  agnt.save(logdir / 'variables.pkl')
for env in train_envs + eval_envs:
  try:
    env.close()
  except Exception:
    pass
