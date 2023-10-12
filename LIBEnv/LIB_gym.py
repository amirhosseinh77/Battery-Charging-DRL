import gymnasium
from gymnasium import spaces
import numpy as np
from LIBEnv.packFullModel import LithiumIonPack
from LIBEnv.plots import plot_pack_SOC

class LIBPackEnv(gymnasium.Env):
  def __init__(self, model='models/PANmodel.mat', number_of_cells=1, T=25, dt=1, current_ratio=-100, use_priority=True, use_switch=False):
    super(LIBPackEnv, self).__init__()
    self.model = model
    self.number_of_cells = number_of_cells
    self.T = T
    self.dt = dt
    self.soc_idx = 0
    self.voltage_idx = 1
    self.temp_idx = 2
    self.prev_action = np.zeros(number_of_cells+1)
    self.MAX_STEP = 2e3
    self.step_counter = 0
    self.current_ratio = current_ratio
    self.coolant_ratio = T
    self.use_priority = use_priority
    self.use_switch = use_switch
    # Define action and observation space
    if self.use_switch:
      self.action_space = spaces.MultiDiscrete([2]*self.number_of_cells+[3])
    else:
      self.action_space = spaces.Box(low=0, high=np.hstack([np.ones(self.number_of_cells),[2]]), shape=(number_of_cells+1,), dtype=np.float32)
      
    self.observation_space = spaces.Box(low=np.stack([-0.05]*self.number_of_cells+[0]*self.number_of_cells+[0]*self.number_of_cells), 
                                        high=np.stack([1.05]*self.number_of_cells+[2]*self.number_of_cells+[4]*self.number_of_cells), 
                                        shape=(3*self.number_of_cells,), dtype=np.float32) # HEIGHT, WIDTH, N_CHANNELS

    # Normalization variables
    self.pack = LithiumIonPack(model, number_of_cells, T, dt, different_params=False)
    self.pack.ECM.z_k = np.ones(self.pack.ECM.z_k.shape)
    self.max_voltage = self.pack.updateModel(0, self.T).mean()

    self.pack = LithiumIonPack(model, number_of_cells, T, dt, different_params=False)
    self.pack.ECM.z_k = np.zeros(self.pack.ECM.z_k.shape)
    self.min_voltage = self.pack.updateModel(0, self.T).mean()
    
    # Define overcharge and overdischarge voltage limits
    self.overcharge_voltage = 4.1 # V
    self.overdischarge_voltage = 2.3 # V
    self.overTemp = 40/self.T
    self.underTemp = 10/self.T

    # Define reward function parameters
    self.target_soc = 1

  def step(self, action):
    # Execute one time step within the environment
    self.step_counter += 1
    current = action[:-1]*self.current_ratio
    coolant_temp = action[-1]*self.coolant_ratio
  
    # current, coolant_temp = action  # Assuming action is a tuple (current, coolant_temp)
    pack_voltage = self.pack.updateModel(current, coolant_temp)
    next_state = np.vstack([self.pack.ECM.z_k, (pack_voltage-self.min_voltage)/(self.max_voltage-self.min_voltage), self.pack.ThM.Tc/self.T])
    
    # Compute cost
    soc_dev = next_state[self.soc_idx].max() - next_state[self.soc_idx].min()
    C_soc = np.abs(next_state[self.soc_idx].mean() - self.target_soc)
    C_balance = np.abs(next_state[self.soc_idx] - next_state[self.soc_idx].mean()).sum()
    # C_soh
    C_heat = np.clip(next_state[self.temp_idx]-self.overTemp, 0, np.inf).sum() + np.clip(self.underTemp-next_state[self.temp_idx], 0, np.inf).sum()
    C_volt = np.clip(pack_voltage-self.overcharge_voltage, 0, np.inf).sum() + np.clip(self.overdischarge_voltage-pack_voltage, 0, np.inf).sum()
    C_smooth = np.abs(action-self.prev_action).sum()

    # balancing threshold
    balance_thresh = 0.02

    # priority-objective reward function
    if self.use_priority:
      T = -np.log(0.5)/balance_thresh
      # w1 = -0.9*np.exp(-T*(soc_dev))+0.95
      # w2 =  0.9*np.exp(-T*(soc_dev))+0.05
      w1 = -0.8*np.exp(-T*(soc_dev))+0.9
      w2 =  0.8*np.exp(-T*(soc_dev))+0.1
    else:
      w1,w2 = 1,1

    w_reg_balance = 0.7/np.sum(np.abs(balance_thresh*np.linspace(0,1,self.number_of_cells) - np.mean(balance_thresh*np.linspace(0,1,self.number_of_cells))))/2
    w_reg_heat = 0.7/(0.01*self.number_of_cells)
    w_reg_volt = 0.7/(0.01*self.number_of_cells)
    cost =  w_reg_balance*w1*C_balance + w2*C_soc + w_reg_heat*C_heat + w_reg_volt*C_volt # + 0.1*C_smooth
    
    reward = -cost
    
    # Check if the episode is done
    # if np.any(next_state[self.soc_idx] >= self.target_soc): || (next_state[self.soc_idx].mean() >= self.target_soc)
    if  np.any(self.target_soc - next_state[self.soc_idx] <= 5e-3): 
      done = True
    elif self.step_counter > self.MAX_STEP: # try other things
      done = True
    else:
      done = False

    # Info dictionary
    info = {
        'voltage': pack_voltage,
        'current': current,
        'temperature': self.pack.ThM.Tc,
        'reward': reward,
    }

    self.prev_action = action
    return next_state.ravel().astype(np.float32), reward, done, False, info


  def reset(self, seed=None, options={}):
    if seed is not None: np.random.seed(seed)
    self.step_counter = 0
    self.prev_action = np.zeros(self.number_of_cells+1)
    # Reset the state of the environment to an initial state
    self.pack = LithiumIonPack(self.model, self.number_of_cells, self.T, self.dt)
    self.pack.ECM.z_k = np.random.rand(*self.pack.ECM.z_k.shape)/np.random.randint(2,5)
    pack_voltage = self.pack.updateModel(0, self.T)
    state = np.vstack([self.pack.ECM.z_k, (pack_voltage-self.min_voltage)/(self.max_voltage-self.min_voltage), self.pack.ThM.Tc/self.T])

    # Info dictionary
    info = {
        'voltage': pack_voltage,
        'temperature': self.pack.ThM.Tc,
    }
    return state.ravel().astype(np.float32), info

  def render(self):
    # Render the environment to the screen
    return plot_pack_SOC(self.pack.ECM.z_k)