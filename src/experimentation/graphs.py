import matplotlib.pyplot as plt
import numpy as np

def display_pk_wd_proximity(pred_thresholds, pk, wd, proximity):
    
    # plot lines
    plt.plot(pred_thresholds, pk, label = "pk")
    plt.plot(pred_thresholds, wd, label = "wd")
    plt.plot(pred_thresholds, proximity, label = "prox")
    plt.legend()
    plt.show()