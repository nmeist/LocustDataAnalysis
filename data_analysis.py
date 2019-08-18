import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.collections import LineCollection
import numba
from matplotlib import colors as mcolors

import matplotlib.pyplot as plt; plt.rcdefaults()
from scipy import stats
import matplotlib.mlab as mlab
import math

totalFrames=324399



def get_past(individ, frame, framesback, newX, zero, one, five, threeone):
    mid_point = int((2*framesback+1)/2)
    # get data from past frames (includes frame=0)
    for frameback in range(framesback):
        if newX[individ, frame-frameback]==0: 
            zero[mid_point-frameback]+=1
        elif newX[individ, frame-frameback]==1: 
            one[mid_point-frameback]+=1
        elif newX[individ, frame-frameback]==5: 
            five[mid_point-frameback]+=1
        elif newX[individ, frame-frameback]==31: 
            threeone[mid_point-frameback]+=1
            
    # get data from future frames
    for framefwd in range(1, framesback):
        if newX[individ, frame+framefwd]==0: 
            zero[mid_point+framefwd]+=1
        elif newX[individ, frame+framefwd]==1: 
            one[mid_point+framefwd]+=1
        elif newX[individ, frame+framefwd]==5: 
            five[mid_point+framefwd]+=1
        elif newX[individ, frame+framefwd]==31: 
            threeone[mid_point+framefwd]+=1
    
    return zero, one, five, threeone
    
def history(newX, y, framesback, region, want_PL_or_Greater, pause_length, filter_frames=0):
    size=(2*framesback)+1
    zero=np.zeros(size)
    one=np.zeros(size)
    five=np.zeros(size)
    threeone=np.zeros(size) 
                    
    speed_changes = 0
    for individ in range(y.shape[0]):
        print(individ)
        for frame in range(framesback, totalFrames-framesback-1):
            
            if filter_frames>0: #  Filtering out the brief changes in motion 
                past=np.sum(y[individ, frame-filter_frames:frame-1])
                future=np.sum(y[individ, frame:frame+filter_frames])
                
                # Keep data points where locust was paused for X frames, moving for X frames (some room for small movement changes)
                if (future-past)>filter_frames-5:
                    if (y[individ, frame] - y[individ, frame-1])==1: 
                        speed_changes+=1
                        print(frame)
                        zero, one, five, threeone=get_past(individ, frame, framesback, newX, zero, one, five, threeone) # count 
                    
            elif filter_frames==0: # no filtering
                past=np.sum(y[individ, frame-pause_length:frame-1]) # 0
                if want_PL_or_Greater: # if the locust has a consecutive pause of length X or greater 
                    if past==0:
                        if (y[individ, frame] - y[individ, frame-1])==1: 
                            speed_changes+=1
                            zero, one, five, threeone=get_past(individ, frame, framesback, newX, zero, one, five, threeone) # count 
                # if the locust has a consecutive pause of length and in the frame before the pause length, it was moving. Only counting interactions of pause length = 1 (no others!)
                else: 
                    if past==0 and y[individ, frame-(pause_length+1)]==1:
                        if (y[individ, frame] - y[individ, frame-1])==1: 
                            speed_changes+=1
                            zero, one, five, threeone=get_past(individ, frame, framesback, newX, zero, one, five, threeone) # count 
                                    
    if (region=='FB' or region=='LR'): # FB and LR only have 0 and 1 as classification numbers 
        print("speed changes: ", speed_changes)
        plot_two(zero/speed_changes, one/speed_changes, framesback, region, want_PL_or_Greater, filter_frames, pause_length)

    if region=='AFMB': # AFMB has 0, 1, 5, 31 as classification numbers
        ant, front, middle, back, = zero/speed_changes, one/speed_changes, five/speed_changes, threeone/speed_changes
        print("speed changes: ", speed_changes)
        plot_AFMB(ant, front, middle, back, framesback, region, want_PL_or_Greater, filter_frames, pause_length)
        

def plot_AFMB(ant, front, middle, back, framesback, region, want_PL_or_Greater, filter_frames, pause_length):

    labels=["Antenna", "Front", "Middle", "Back"]
    size=(2*(framesback))+1
    ind = np.arange(0, size)-framesback  # the x locations for the groups
    fig, ax = plt.subplots()

    ant_line, = plt.plot(ind[1:size-1], ant[1:size-1], label="%s" % labels[0], linestyle='-')
    front_line, = plt.plot(ind[1:size-1], front[1:size-1], label="%s" % labels[1], linestyle='-')
    middle_line, = plt.plot(ind[1:size-1], middle[1:size-1], label="%s" % labels[2], linestyle='-')
    back_line, = plt.plot(ind[1:size-1], back[1:size-1], label="%s" % labels[3], linestyle='-')
    
    # avg line
    '''for idx, avg in enumerate(avg_AFMB):
        plt.axhline(y=avg, linestyle='-', color=colors[idx])'''
    
    plt.axvline(x=0, linestyle=':', color='b', linewidth=2)
    if filter_frames>0:
        plt.axvspan(-1*filter_frames, filter_frames, facecolor='0.5', alpha=0.2)
    elif filter_frames==0:
        plt.axvspan(-1*pause_length, 0, facecolor='0.5', alpha=0.2)


    plt.legend(bbox_to_anchor=(1, 1), loc=1, fontsize=20)
    
    ax.set_xlabel('Frame before and after paused to moving transition occurs (transition occurs at frame=0)', fontsize=20)
    ax.set_ylabel('P(Interaction with Limb)', fontsize=20)
    
    ax.set_xticks(np.arange(0, 2*framesback+10, 10)-framesback)
    ax.set_xticklabels(np.arange(0, 2*framesback+10, 10)-framesback)
    fig.set_size_inches((30, 12))

    if filter_frames>0: 
        ax.set_title('Probability Locusts Touches with Limb %.2f Seconds \n Before and After a Change From %.2f Seconds of Pause to %.2f Seconds of Motion is Observed' % (framesback/90, filter_frames/90, filter_frames/90), fontsize=25)
    else: 
        ax.set_title('Probability Locusts Touches with Limb %.2f Seconds \n Before and After a Change From %.2f Seconds of Pause to Moving is Observed' % (framesback/90, pause_length/90), fontsize=25)

    if filter_frames>0: 
        fig.savefig('./plots/AFMB/AFMB_filter%d.png' % filter_frames) 
    elif want_PL_or_Greater: 
        fig.savefig('./plots/AFMB/AFMB_%d_or_more.png' % pause_length)
    else: 
        fig.savefig('./plots/AFMB/AFMB_only%d.png' % pause_length)
    
    plt.show()

def plot_two(front, back, framesback, region, want_PL_or_Greater, filter_frames, pause_length):
    size=(2*framesback)+1
    ind = np.arange(0, 2*framesback+10)-framesback  # the x locations for the groups
    if region=='FB': 
        labels = ['Front', 'Back']
    elif region=='LR':
        labels = ['Left', 'Right']

    fig, ax = plt.subplots()
    
    plt.plot(ind[1:size-1], front[1:size-1], label=labels[0], linestyle='-')
    plt.plot(ind[1:size-1], back[1:size-1], label=labels[1], linestyle='-')
    
    #plot avg line
    '''for idx, avg in enumerate(avg_two):
        plt.axhline(y=avg, linestyle='-', color=colors[idx])'''
    
    plt.axvline(x=0, linestyle=':', color='b', linewidth=2)
    if filter_frames>0:
        plt.axvspan(-1*filter_frames, filter_frames, facecolor='0.5', alpha=0.2)
    elif filter_frames==0:
        plt.axvspan(-1*pause_length, 0, facecolor='0.5', alpha=0.2)

    plt.legend(bbox_to_anchor=(1, 1), loc=1)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Frames before and after paused to moving transition occurs (transition occurs at frame=0)')
    ax.set_ylabel('P(Interaction with Limb)')
    
    if filter_frames>0: 
        ax.set_title('Probability Locusts Touches with Limb %.2f Seconds \n Before and After a Change From %.2f Seconds of Pause to %.2f Seconds of Motion is Observed' % (framesback/90, filter_frames/90, filter_frames/90))
    else: 
        ax.set_title('Probability Locusts Touches with Limb %.2f Seconds \n Before and After a Change From %.2f Seconds of Pause to Moving is Observed' % (framesback/90, pause_length/90))

    ax.set_xticks(np.arange(0, 2*framesback+60, 60)-framesback)
    ax.set_xticklabels(np.arange(0, 2*framesback+60, 60)-framesback)
    
    if filter_frames>0: 
        fig.savefig('./plots/%s/%s_filter%d.png' % (region, region, filter_frames))
    elif want_PL_or_Greater: 
        fig.savefig('./plots/%s/%s_%d_or_more.png' % (region, region, pause_length))
    else: 
        fig.savefig('./plots/%s/%s_only%d.png' % (region, region, pause_length))
    plt.show()
    

def heatmap(limb_count):
    # create heatman of limb interactions
    with h5py.File('Locust_Template.h5', 'r') as h5File:
        allLocusts = h5File.get('allLocusts')
        allLocusts = np.array(allLocusts)

    N_INDIVID = len(allLocusts)
    fig = plt.figure(figsize=(5,10))
    ax = plt.gca()
    # for limb 
    for limb in range(np.array(allLocusts).shape[1]):
        i=limb+1
        if i==26:
            line_collection = LineCollection((allLocusts[0][limb], allLocusts[1][0]), colors=plt.cm.gist_heat(limb_count[limb]/np.max(limb_count)), cmap='copper')
        else:
            line_collection = LineCollection((allLocusts[0][limb], allLocusts[0][i]), colors=plt.cm.gist_heat(limb_count[limb]/np.max(limb_count)), cmap='copper')
        ax.add_collection(line_collection)

    plt.xlim(970,1040)
    plt.ylim(1425,1510)
    plt.show()
