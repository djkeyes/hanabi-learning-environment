
import numpy as np
import matplotlib.pyplot as plt

def plot_matrix(scores, stddevs, classes, title=None, cmap=plt.cm.viridis):

    print(scores)

    fig, ax = plt.subplots()
    im = ax.imshow(scores, interpolation='nearest', cmap=cmap, vmin=0, vmax=25)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(scores.shape[1]),
           yticks=np.arange(scores.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='First Agent Type',
           xlabel='Second Agent Type')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '{:.2f}\n({:.2f})'
    thresh = scores.max() / 2.
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            ax.text(j, i, fmt.format(scores[i, j], stddevs[i, j]),
                    ha="center", va="center",
                    color="white" if scores[i, j] < thresh else "black")
    fig.tight_layout()
    return ax
    
means = np.array([[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.3000e-02],
[0.0000e+00, 0.0000e+00, 2.1230e+00, 7.9600e-01, 1.3900e+00, 9.7900e-01],
[0.0000e+00, 0.0000e+00, 7.9600e-01, 1.8746e+01, 1.7570e+00, 1.4682e+01],
[0.0000e+00, 0.0000e+00, 1.3900e+00, 1.7570e+00, 1.2580e+00, 1.3010e+00],
[0.0000e+00, 1.3000e-02, 9.7900e-01, 1.4682e+01, 1.3010e+00, 1.7137e+01]])

stddevs = np.array([[0., 0., 0., 0., 0., 0., ],
[0., 0., 0., 0., 0., 0.3111768, ],
[0., 0., 1.82862544, 1.98655078, 1.59245094, 2.07281427],
[0., 0., 1.98655078, 4.08233806, 1.97786526, 6.05102272],
[0., 0., 1.59245094, 1.97786526, 1.87761444, 2.07615004],
[0., 0.3111768, 2.07281427, 6.05102272, 2.07615004, 3.89310044]])

agent_names = ['SimpleAgent', 'RandomAgent', 'LossAverseAgent', 'Rainbow', 'DQN', 'Rainbow-pretrained']

ax = plot_matrix(means, stddevs, agent_names, 'Mean (Std Dev) of Scores For 2-player Hanabi Games')
plt.show()
