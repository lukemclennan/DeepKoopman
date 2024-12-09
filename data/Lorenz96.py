import numpy as np
from scipy.integrate import odeint
import pandas as pd
import matplotlib.pyplot as plt

def gen_trajs(initial, t_eval, F):
    num_trajs, N = initial.shape
    odefunc = lambda x, t : (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    trajs = np.zeros((num_trajs, len(t_eval), N))
    for i in range(num_trajs):
        if i%1000 == 0:
            print(i)
        trajs[i,:,:] = odeint(odefunc, initial[i,:], t_eval)
    return trajs

names = ["train1", "train2", "train3", "val"]
for name in names:
    initial = np.random.randn(5000, 4)
    t_eval = np.arange(0, 0.10005, 0.001)
    print(t_eval.shape)
    F = 0

    trajs = gen_trajs(initial, t_eval, F)

    # i = 0

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot(trajs[i, :, 0], trajs[i, :, 1], trajs[i, :, 2])
    # plt.show()

    print(trajs.shape)
    trajs = trajs.reshape((trajs.shape[0]*trajs.shape[1], trajs.shape[2]))
    print(trajs.shape)

    trajs = pd.DataFrame(trajs)
    trajs.to_csv("data/Lorenz96_" + name + "_x.csv", header=False, index=False)

