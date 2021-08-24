import numpy as np


class Driver:

  def __init__(self, envs, **kwargs):
    self._envs = envs
    self._kwargs = kwargs
    self._on_steps = []
    self._on_resets = []
    self._on_episodes = []
    self._actspaces = [env.action_space.spaces for env in envs]
    self.reset()

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_reset(self, callback):
    self._on_resets.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def reset(self):
    self._obs = [None] * len(self._envs)
    self._dones = [True] * len(self._envs)
    self._eps = [None] * len(self._envs)
    self._state = None

  def __call__(self, policy, steps=0, episodes=0):
    #policy is the agent
    step, episode = 0, 0
    while step < steps or episode < episodes:
      for i, done in enumerate(self._dones):
        if done:
          self._obs[i] = ob = self._envs[i].reset()
          #reset env and save the first image of render (game start), reset self._obs with new ob
          act = {k: np.zeros(v.shape) for k, v in self._actspaces[i].items()}
          #reset action space
          tran = {**ob, **act, 'reward': 0.0, 'discount': 1.0, 'done': False}
          #observation, action, reward,discount, done all re-initialized
          [callback(tran, **self._kwargs) for callback in self._on_resets]
          #perform all the callbacks in the self._on_rests list, on the tran object
          self._eps[i] = [tran]
          #episode is reset, initialize list of eps with the new obs
      obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
      #For each type of obs, create key for that type and stack each of these obs of each env together. 
      # Only one obs per ENV, but the observations could have multiple key (i.e image, reward,etc..), this is k 
      #this is to process each obs in batches. We basically have the shape of one OB, but with batches

      actions, self._state = policy(obs, self._state, **self._kwargs)
      #one action per env,also the next states
      actions = [
          {k: np.array(actions[k][i]) for k in actions}
          for i in range(len(self._envs))]
      #each action can have a action space with multiple Ks
      assert len(actions) == len(self._envs)
      results = [e.step(a) for e, a in zip(self._envs, actions)]
      #only action["action"] is kept. obs has "image" and "ram". step returns obs,rew, done, info
      #done is a boolean, checks if it's time to rest the environment. Info is a dict with info for debug
      for i, (act, (ob, rew, done, info)) in enumerate(zip(actions, results)):
        #iterate through each action and results of each env
        obs = {k: self._convert(v) for k, v in obs.items()}
        #convert obs to the appriopriate numpy image type
        disc = info.get('discount', np.array(1 - float(done)))
        #no documention on this part, figure out what it does
        tran = {**ob, **act, 'reward': rew, 'discount': disc, 'done': done}
        #every information about the step, obs ,actions ,reward, discount, done.
        [callback(tran, **self._kwargs) for callback in self._on_steps]
        #for each function in on_steps (what to do each step), perform on the full state
        self._eps[i].append(tran)
        #append the whole step to _eps array, one array per env
        if done:
          #if done, we have an episode (all the steps in that array)
          ep = self._eps[i]
          ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
          #for each info, obs, act, reward, etc. Concatenate the episode, convert and put under one key
          #{obs: [obs1,obs2,obs3,..],reward: [reward1,reward2,...]}
          [callback(ep, **self._kwargs) for callback in self._on_episodes]
          #perform on that episode all the functions in on_episodes
      obs, _, dones = zip(*[p[:3] for p in results])
      #p[:3]= obs, reward, done
      #zip all observation (for each env) and all dones (each env together)
      self._obs = list(obs)
      #put them together for next iteration, each i is one env 
      self._dones = list(dones)
      #put them together for next iteration, each i is one env 
      episode += sum(dones)
      #everytime we have a done, one episode is done
      step += len(dones)
      #number of env is number of steps

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
      return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
      return value.astype(np.uint8)
    return value
