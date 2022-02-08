import matplotlib.pyplot as plt

def set_plot_prop():
    plt.ioff()
    mm2inch = lambda x: x/10./2.54
    # plt.rcParams['xtick.direction']= 'out'
    # plt.rcParams['ytick.direction']= 'out'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['grid.color'] = 'k'
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.linewidth'] = 0.75
    # plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 24
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.linewidth'] = 2.
    plt.rcParams['figure.figsize'] = mm2inch(90*5),mm2inch(2./3*90*5)
    plt.rcParams["legend.handlelength"] = 1.
    plt.rcParams["legend.handletextpad"] = 0.15
    plt.rcParams["legend.borderpad"] = 0.15
    plt.rcParams["legend.labelspacing"] = 0.15
    cmap=plt.cm.get_cmap('RdYlBu')
    plt.rcParams.update({
        "figure.facecolor":  (1.0, 1.0, 1.0, 1),  # red   with alpha = 30%
        "axes.facecolor":    (1.0, 1.0, 1.0, 1),  # green with alpha = 50%
        "savefig.facecolor": (1.0, 1.0, 1.0, 1),  # blue  with alpha = 20%
    })

    plt.rcParams['axes.facecolor'] = 'white'
    print('Setup Plotting characteristics .... ')
