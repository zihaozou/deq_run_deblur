from DataFidelities.MRIClass import *
import scipy.io as sio

IMG_PATCH = [325,256]
numCoils = 128

coils1 = MRIClass.get_coils(IMG_PATCH, numCoils)


coils2 = MRIClass.get_coils(IMG_PATCH, numCoils)

# coils1 = coils1.detach().cpu().numpy()

# coils1 = coils1[...,0] + 1j*coils1[...,1]
print(coils1.shape)

sio.savemat('coils1.mat',{'coils':coils1})

