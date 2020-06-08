from __future__ import absolute_import
from tensorflow.keras import utils
import pandas as pd
import os
from utils.iter_utils import fold
import numpy as np
from sklearn.model_selection._split import LeaveOneGroupOut, train_test_split
import tqdm
from _datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns


# Use 3 decimal places in output display
pd.set_option("display.precision", 3)
# Don't wrap repr(DataFrame) across additional lines
pd.set_option("display.expand_frame_repr", False)
# Set max rows displayed in output to 25
pd.set_option("display.max_rows", 25)


sns.set(style="ticks", color_codes=True)

def show_groups(dataframe):
    for group, frame in dataframe:
        print(f"First 2 entries for {group!r}")
        print("------------------------")
        print(frame.head(2), end="\n")

def test_data():
    fname = 'hgest.hdf'
    origin = f'https://storage.googleapis.com/exoskelebox/{fname}'
    path: str = utils.get_file(
        fname, origin, )
    key = 'normalized'
    df = pd.read_hdf(path, key)

    # Create target Directory if don't exist
    def check_dir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    figuredir = os.path.join('figures', 'data_boxplots')
    check_dir(figuredir)
    
    subject_groups = df.groupby("subject_id", as_index=False)
    for id, _ in subject_groups:
        print(f"Subject: {id}", end=', ')

        subject_dir = os.path.join(figuredir, f'subject_{id}')
        check_dir(subject_dir)
        subject_data = df[df.subject_id == id]
        
        gesture_groups = subject_data.groupby("gesture", as_index=False)
        for gesture, _ in gesture_groups:
            print(f"Gesture: {gesture}")

            gesture_data = subject_data[subject_data.gesture == gesture]

            gesture_data.pop('subject_id')
            gesture_data.pop('gesture')
            gesture_data.pop('label')


            print(gesture_data.groupby('repetition')['repetition'].count().head(20))

            gesture_dir = os.path.join(subject_dir, f'{gesture}_distribution.pdf')
            fig = boxplot(gesture_data, id, gesture)
            fig.savefig(gesture_dir)
            plt.close(fig)



def barplot(df, subject, gesture):
    """
    Returns a matplotlib figure containing the plotted data.
    """


    labels = [f'sensor{i}' for i in range(1, 16)]
    reps = [[np.mean(df[df.repetition == i][l].to_numpy()) for l in labels] for i in range(1,6)]
    

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars
    rects = []
    fig, ax = plt.subplots()
    for i, rep in enumerate(reps, start=1):
        rects.append(ax.bar(x - ((3-i)*width), reps[i-1], width, label=f'R{i}'))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Repetitions')
    ax.set_title('Average sensor values by repetition')
    #ax.set_xticks(x)
    #ax.set_xticklabels(labels)
    # Rotate the tick labels and set their alignment
    plt.xticks(x, labels, rotation=45,
               ha="right", rotation_mode="anchor")
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    #for r in rects:
        #autolabel(r)

    fig.tight_layout()
    #plt.show()
    return fig


def boxplot(df: pd.DataFrame, subject, gesture):
    """
    Returns a matplotlib figure containing the plotted data.
    """


    labels = [f'sensor{i}' for i in range(1, 16)]
    reps = [[df[df.repetition == i][l].to_numpy() for l in labels] for i in range(1,6)]
    

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars
    rects = []
    fig, ax = plt.subplots()
    for i, rep in enumerate(reps, start=1):
        rects.append(ax.boxplot(x=reps[i-1], positions=x - ((3-i)*width), widths=width))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Repetitions')
    ax.set_title('Average sensor values by repetition')
    #ax.set_xticks(x)
    #ax.set_xticklabels(labels)
    # Rotate the tick labels and set their alignment
    plt.xticks(x, labels, rotation=45,
               ha="right", rotation_mode="anchor")
    ax.legend()

    fig.tight_layout()
    #plt.show()
    return fig

def s_boxplot(df, gesture):
    """
    Returns a seaborn figure containing the plotted data.
    """

    #labels = [f'sensor{i}' for i in range(1, 16)]
    
    df = df.melt('subject_id', var_name='Sensor',  value_name='Value')

    fig = sns.catplot(x='Sensor', y='Value', hue='subject_id', data=df, kind='box')
    
    plt.xticks(rotation=45, 
               ha="right", rotation_mode="anchor")
    return fig

def best_worst(best=18, worst=15):
    fname = 'hgest.hdf'
    origin = f'https://storage.googleapis.com/exoskelebox/{fname}'
    path: str = utils.get_file(
        fname, origin, )
    key = 'normalized'
    df = pd.read_hdf(path, key)

    # Create target Directory if don't exist
    def check_dir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    figuredir = os.path.join('figures', 'data_boxplots_best_worst')
    check_dir(figuredir)

    gesture_groups = df.groupby("gesture", as_index=False)

    for gesture, _ in gesture_groups:
        print(f"Gesture: {gesture}")

        gesture_data = df[df.gesture == gesture]
        gesture_data.pop('gesture')
        gesture_data.pop('label')
        gesture_data.pop('repetition')
        best_worst = gesture_data[(gesture_data.subject_id == best) | (gesture_data.subject_id == worst)]
        """best_data = gesture_data[gesture_data.subject_id == best]
        worst_data = gesture_data[gesture_data.subject_id == worst]

        best_data.pop('subject_id')
        worst_data.pop('subject_id')"""

        gesture_dir = os.path.join(figuredir, f'{gesture}_best_worst_comparison.pdf')
        fig = s_boxplot(best_worst, gesture)
        plt.savefig(gesture_dir)



if __name__ == "__main__":
    best_worst()
