import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
from ruamel import yaml
import depoco.utils.point_cloud_utils as pcu

def plotResults(files, x_key, y_key, ax, draw_line=False, label=None, set_lim=False):
    x = []
    y = []
    for f in files:
        eval_dict = pcu.load_obj(f)
        
        if((x_key in eval_dict.keys()) & (y_key in eval_dict.keys())):
            for v in eval_dict.values():
                v = np.array(v)
            if not draw_line:
                ax.plot(np.mean(eval_dict[x_key]),
                        np.mean(eval_dict[y_key]), '.')
                ax.text(np.mean(eval_dict[x_key]), np.mean(
                    eval_dict[y_key]), f.split('/')[-1][:-4])

            x.append(np.mean(eval_dict[x_key]))
            y.append(np.mean(eval_dict[y_key]))
            # print(y_key, np.mean(eval_dict[y_key]))

    print('------------------')
    print(label)
    print(y_key)
    print(x)
    print(y)

    if draw_line:
        line, = ax.plot(x, y, '-x', label=label)
        # line.set_label(label)

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)

    if set_lim:
        ax.set_xlim(0,None)
        ax.set_ylim(0,None)
    # ax.grid()


def plotComparisonResults(x,y, x_key, y_key, ax, draw_line=False, label=None, set_lim=False):
    # x = []
    # y = []            

    if draw_line:
        line, = ax.plot(x, y, '-x', label=label)
        # line.set_label(label)

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)

    if set_lim:
        ax.set_xlim(0,None)
        ax.set_ylim(0,None)
    # ax.grid()


def genComparisonPlots(ax, draw_line=True, label=None, x_key='memory'):
    
    x= [0.021, 0.194, 0.875,1.25]
    y= [0.25, 0.117, 0.06,0.05]
    plotComparisonResults(x,y, x_key=x_key, y_key='chamfer_dist_abs',
                ax=ax[0], draw_line=draw_line, label=label)
    x= [0.021, 0.194, 0.875,1.25]
    y= [0.15, 0.07, 0.035,0.029]
    plotComparisonResults(x,y, x_key=x_key, y_key='chamfer_dist_plane',
                ax=ax[1], draw_line=draw_line, label=label)
    x= [0.021, 0.194, 0.875,1.25]
    y= [0.01, 0.12, 0.49,0.55]
    plotComparisonResults(x,y, x_key=x_key, y_key='iou',
                ax=ax[2], draw_line=draw_line, label=label)

def genComparisonPlots1(ax, draw_line=True, label=None, x_key='memory'):
    
    x= [0.125, 0.375, 0.75,1.00]
    y= [0.26, 0.13, 0.108,0.09]
    plotComparisonResults(x,y, x_key=x_key, y_key='chamfer_dist_abs',
                ax=ax[0], draw_line=draw_line, label=label)
    x= [0.125, 0.375, 0.75,1.00]
    y= [0.15, 0.055, 0.048,0.038]
    plotComparisonResults(x,y, x_key=x_key, y_key='chamfer_dist_plane',
                ax=ax[1], draw_line=draw_line, label=label)
    x= [0.125, 0.375, 0.75,1.00]
    y= [0.04, 0.17, 0.337,0.456]
    plotComparisonResults(x,y, x_key=x_key, y_key='iou',
                ax=ax[2], draw_line=draw_line, label=label)
          

def genComparisonPlots2(ax, draw_line=True, label=None, x_key='memory'):
    
    x= [0.125, 0.26, 0.651,1.00]
    y= [0.4, 0.218, 0.12,0.1]
    plotComparisonResults(x,y, x_key=x_key, y_key='chamfer_dist_abs',
                ax=ax[0], draw_line=draw_line, label=label)
    x= [0.125, 0.26, 0.651,1.00]
    y= [0.225, 0.125, 0.07,0.053]
    plotComparisonResults(x,y, x_key=x_key, y_key='chamfer_dist_plane',
                ax=ax[1], draw_line=draw_line, label=label)
    x= [0.125, 0.26, 0.651,1.00]
    y= [0.001, 0.01, 0.11,0.22]
    plotComparisonResults(x,y, x_key=x_key, y_key='iou',
                ax=ax[2], draw_line=draw_line, label=label)
          



def genPlots(files, f, ax, draw_line=False, label=None, x_key='memory'):
    # print('shape',ax[0,0])
    plotResults(files, x_key=x_key, y_key='chamfer_dist_abs',
                ax=ax[0], draw_line=draw_line, label=label)
    plotResults(files, x_key=x_key, y_key='chamfer_dist_plane',
                ax=ax[1], draw_line=draw_line, label=label)
    plotResults(files, x_key=x_key, y_key='iou',
                ax=ax[2], draw_line=draw_line, label=label)


if __name__ == "__main__":
    ####### radius fct ##############
    path = 'experiments/results/kitti/'
    files = sorted(glob.glob(path+'*.pkl'))

    f, ax = plt.subplots(1, 3)
    f.suptitle('Radius FPS')
    genPlots(files, f, ax, draw_line=True, label='our', x_key='bpp')
    for a in ax:
        a.grid()
        # a.set_ylim([0,None])
        a.legend()
    plt.show()
