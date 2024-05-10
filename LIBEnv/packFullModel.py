import numpy as np
from scipy.interpolate import interp1d
from mat4py import loadmat

class LithiumIonPack:
    def __init__(self, model_path, number_of_cells, T, dt, different_params=True):
        self.ECM = ElectricalModel(model_path, number_of_cells=number_of_cells, T=T, dt=dt)
        self.ThM = LumpedModel(number_of_cells=number_of_cells, T=T, dt=dt)
        if different_params: 
            self.ECM.change_params(ratio=1e1)
            self.ThM.change_params(ratio=1e3)

    def updateModel(self, current, Tf):
        # update electrical model
        newState = self.ECM.stateEqn(current)
        voltage = self.ECM.outputEqn(current)
        self.ECM.updateState(newState)
        # update thermal model
        Q = current*(self.ECM.OCVfromSOC(self.ECM.z_k)-voltage)
        self.ThM.updateTemp(Q, Tf)
        # update electrical parameters
        self.ECM.update_params(self.ThM.Tc)
        return voltage


class ElectricalModel:
    def __init__(self, model_path, number_of_cells, T, dt):
        data = loadmat(model_path)
        # print(f"Initializing Battery : {data['model']['name']}")
        self.model_data = data['model']
        self.temps = data['model']['temps']
        self.number_of_cells = number_of_cells
        self.T = T

        self.etaFunction = interp1d(self.temps, self.model_data['etaParam'], fill_value="extrapolate")
        self.QFunction   = interp1d(self.temps, self.model_data['QParam'], fill_value="extrapolate")
        self.GFunction   = interp1d(self.temps, self.model_data['GParam'], fill_value="extrapolate")
        self.M0Function  = interp1d(self.temps, self.model_data['M0Param'], fill_value="extrapolate")
        self.MFunction   = interp1d(self.temps, self.model_data['MParam'], fill_value="extrapolate")
        self.R0Function  = interp1d(self.temps, self.model_data['R0Param'], fill_value="extrapolate")
        self.RCFunction  = interp1d(self.temps, np.ravel(self.model_data['RCParam']), fill_value="extrapolate")
        self.RFunction   = interp1d(self.temps, np.ravel(self.model_data['RParam']), fill_value="extrapolate")
        self.set_params(T)
        self.OCVfromSOC = make_OCVfromSOCtemp(self.model_data, T)
        self.SOCfromOCV = make_SOCfromOCVtemp(self.model_data, T)

        self.dt = dt
        self.sik = np.zeros((1,self.number_of_cells))
        self.iR_k = np.zeros((1,self.number_of_cells))
        self.h_k = np.zeros((1,self.number_of_cells))
        self.z_k = np.zeros((1,self.number_of_cells))

    def stateEqn(self, current, xnoise=0, oldState=None):
        if oldState is not None: 
            (iR_k, h_k, z_k) = oldState
            iR_k = iR_k.reshape(1,-1)
            h_k =  h_k.reshape(1,-1)
            z_k =  z_k.reshape(1,-1)
        else:
            iR_k = self.iR_k.reshape(1,-1)
            h_k =  self.h_k.reshape(1,-1)
            z_k =  self.z_k.reshape(1,-1)

        self.sik = np.where(abs(current)>self.QParam/100, np.sign(current), self.sik)
        current = np.where(current<0, current*self.etaParam, current)
        current = current + xnoise
        
        Ah = np.exp(-abs(current*self.GParam*self.dt/(3600*self.QParam)))  # hysteresis factor
        Arc = np.diag(np.exp(-self.dt/abs(self.RCParam)))
        Brc = 1-(np.exp(-self.dt/abs(self.RCParam)))

        iR_k1 = Arc@iR_k + Brc*current
        h_k1 = Ah*h_k - (1-Ah)*np.sign(current)
        z_k1 = z_k - (self.dt/(3600*self.QParam))*current

        h_k1 = np.clip(h_k1, -1, 1)
        z_k1 = np.clip(z_k1, -0.05, 1.05)

        newState = (iR_k1, h_k1, z_k1)
        return newState

    def outputEqn(self, current, ynoise=0, state=None):
        if state is not None: 
            (iR_k, h_k, z_k) = state
            iR_k = iR_k.reshape(1,-1)
            h_k =  h_k.reshape(1,-1)
            z_k =  z_k.reshape(1,-1)
        else:
            iR_k = self.iR_k
            h_k =  self.h_k
            z_k =  self.z_k

        voltage = self.OCVfromSOC(z_k) + self.MParam*h_k + self.M0Param*self.sik - self.RParam*iR_k - self.R0Param*current + ynoise
        return voltage

    def updateState(self, newState):
        (iR_k1, h_k1, z_k1) = newState
        self.iR_k = iR_k1
        self.h_k = h_k1
        self.z_k = z_k1

    def keep_params(self):
        self.etaParam_init = self.etaParam
        self.QParam_init   = self.QParam  
        self.GParam_init   = self.GParam  
        self.M0Param_init  = self.M0Param 
        self.MParam_init   = self.MParam  
        self.R0Param_init  = self.R0Param 
        self.RCParam_init  = self.RCParam  
        self.RParam_init   = self.RParam  

    def set_params(self, temp):
        self.etaParam = self.etaFunction(temp)
        self.QParam   =  self.QFunction(temp) 
        self.GParam   =  self.GFunction(temp) 
        self.M0Param  = self.M0Function(temp) 
        self.MParam   =  self.MFunction(temp) 
        self.R0Param  = self.R0Function(temp) 
        self.RCParam  = self.RCFunction(temp).reshape(-1) 
        self.RParam   =  self.RFunction(temp) 
        self.keep_params()
 
    def change_params(self, ratio):
        self.etaParam = np.random.normal(loc=self.etaParam, scale=self.etaParam/ratio, size=(1,self.number_of_cells))
        self.QParam   = np.random.normal(loc=self.QParam, scale=self.QParam/ratio, size=(1,self.number_of_cells))
        self.GParam   = np.random.normal(loc=self.GParam, scale=self.GParam/ratio, size=(1,self.number_of_cells))
        self.M0Param  = np.random.normal(loc=self.M0Param, scale=self.M0Param/ratio, size=(1,self.number_of_cells))
        self.MParam   = np.random.normal(loc=self.MParam, scale=self.MParam/ratio, size=(1,self.number_of_cells))
        self.R0Param  = np.random.normal(loc=self.R0Param, scale=self.R0Param/ratio, size=(1,self.number_of_cells))
        self.RCParam  = np.random.normal(loc=self.RCParam, scale=self.RCParam/ratio, size=(1,self.number_of_cells))
        self.RParam   = np.random.normal(loc=self.RParam, scale=self.RParam/ratio, size=(1,self.number_of_cells))
        self.keep_params()

    def update_params(self, temp):
        self.etaParam = self.etaFunction(temp) + (self.etaParam_init - self.etaFunction(self.T))
        self.QParam   = self.QFunction(temp)   + (self.QParam_init - self.QFunction(self.T))
        self.GParam   = self.GFunction(temp)   + (self.GParam_init - self.GFunction(self.T))
        self.M0Param  = self.M0Function(temp)  + (self.M0Param_init - self.M0Function(self.T))
        self.MParam   = self.MFunction(temp)   + (self.MParam_init - self.MFunction(self.T))
        self.R0Param  = self.R0Function(temp)  + (self.R0Param_init - self.R0Function(self.T))
        self.RCParam  = self.RCFunction(temp)  + (self.RCParam_init - self.RCFunction(self.T))
        self.RParam   = self.RFunction(temp)   + (self.RParam_init - self.RFunction(self.T))
        self.OCVfromSOC = make_OCVfromSOCtemp(self.model_data, np.mean(temp))
        self.SOCfromOCV = make_SOCfromOCVtemp(self.model_data, np.mean(temp))
    
    def show_params(self):
        print(f'eta   : {self.etaParam}\n')
        print(f'Q     : {self.QParam}\n')
        print(f'gamma : {self.GParam}\n')
        print(f'M0    : {self.M0Param}\n')
        print(f'M     : {self.MParam}\n')
        print(f'R0    : {self.R0Param}\n')
        print(f'RC    : {self.RCParam}\n')
        print(f'R     : {self.RParam}')


def make_OCVfromSOCtemp(model_data, T):
    SOC = np.array(model_data['SOC'])
    OCV0 = np.array(model_data['OCV0'])
    OCVrel = np.array(model_data['OCVrel'])
    OCV = OCV0 + T*OCVrel
    OCVfromSOC = interp1d(SOC, OCV, fill_value="extrapolate")
    return OCVfromSOC

def make_dOCVfromSOCtemp(model_data, T):
    SOC = np.array(model_data['SOC'])
    OCV0 = np.array(model_data['OCV0'])
    OCVrel = np.array(model_data['OCVrel'])
    OCV = OCV0 + T*OCVrel

    dZ = SOC[1] - SOC[0]
    dUdZ = np.diff(OCV)/dZ
    dOCV = (np.append(dUdZ[0],dUdZ) + np.append(dUdZ,dUdZ[-1]))/2
    dOCVfromSOC = interp1d(SOC, dOCV, fill_value="extrapolate")
    return dOCVfromSOC

def make_SOCfromOCVtemp(model_data, T):
    OCV = np.array(model_data['OCV'])
    SOC0 = np.array(model_data['SOC0'])
    SOCrel = np.array(model_data['SOCrel'])
    SOC = SOC0 + T*SOCrel
    SOCfromOCV = interp1d(OCV, SOC, fill_value="extrapolate")
    return SOCfromOCV


class LumpedModel:
    def __init__(self, number_of_cells, T, dt):
        self.number_of_cells = number_of_cells
        self.Rc = 1.94
        self.Ru = 3.19
        self.Cc = 62.7
        self.Cs = 4.5
        self.a1 = -1/(self.Rc*self.Cc)
        self.a2 = -1/(self.Rc*self.Cs) - 1/(self.Ru*self.Cs)

        self.Tc = T*np.ones((1,number_of_cells))
        self.Ts = T*np.ones((1,number_of_cells))
        self.dt = dt

    def updateTemp(self, Q, Tf):
        u1 = (self.Ts/(self.Rc*self.Cc)) + Q/self.Cc
        u2 = (self.Tc/(self.Rc*self.Cs)) + (Tf/(self.Ru*self.Cs))
        self.Tc = np.exp(self.a1*self.dt) * self.Tc + (np.exp(self.a1*self.dt)-1)/self.a1 * u1
        self.Ts = np.exp(self.a2*self.dt) * self.Ts + (np.exp(self.a2*self.dt)-1)/self.a2 * u2

    def change_params(self, ratio):
        self.Rc = np.random.normal(loc=self.Rc, scale=self.Rc/ratio, size=(1,self.number_of_cells))
        self.Ru = np.random.normal(loc=self.Ru, scale=self.Ru/ratio, size=(1,self.number_of_cells))
        self.Cs = np.random.normal(loc=self.Cs, scale=self.Cs/ratio, size=(1,self.number_of_cells))
        # self.Cc = np.random.normal(loc=self.Cc, scale=self.Cc/ratio, size=(1,self.number_of_cells))

    def show_params(self):
        print(f'Rc : {self.Rc}\n')
        print(f'Ru : {self.Ru}\n')
        print(f'Cs : {self.Cs}\n')
        print(f'Cc : {self.Cc}')