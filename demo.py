from hmf import *
import scipy.io as sio
import pickle
import pdb

settings = np.load('msc_settings_motor_session01.npy', allow_pickle=True).tolist()
input_dict = np.load('msc_input_motor_session01.npy', allow_pickle=True).tolist()

hmf = CanonicalHRFMatrixFactorizationFast(settings)
hmf.fit(input_dict)
out = hmf.get_params(input_dict)
sio.savemat('msc_out_motor_session01.mat', out)