from collections import OrderedDict
import numpy as np
from config import project_path
import torch
import os
import pickle as pkl
import matplotlib.pyplot as plt
from Vectorizers.vectorizers import nicknames


def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad+5),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(-1, 0),
                xytext=(-ax.yaxis.labelpad - row_pad-30, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )

def plot_one_blank(model, blanks, scores, y_lim=(-1.0, 1.0)):
    try:
        d2 = [scores[model][x]["rms"] for x in ('main', 'solvent', 'solute')]
        # print(f'd: {d2}')
        d = [d2[i]-blanks[i] for i in range(3)]
    except KeyError:
        d = [0,0,0]
    # print(f'd-b: {d}')

    c = [int(x<0) for x in d]
    # print(f'c: {c}')
    two_colors = [('#EE220C','#0433FF'), ('#FF9300' , '#16E7CF'), ('#D41876' , '#4B1F8C')]
    colors = [two_colors[i][c[i]] for i in range(3)]

    plt.bar(('main', 'solvent', 'solute'), d,
                color=colors,
                width=1)
    plt.ylim(*y_lim)
    plt.axis('off')
    # plt.tight_layout()
    # plt.axhline(2, color='k', linestyle='--')

def plot_one(model, scores, y_lim=(0.0, 2.0), relative_bars=None):
    try:
        d = []
        for x in ('main', 'solvent', 'solute'):
            score = scores[model][x]['rms']
            if relative_bars == 'mean':
                score /= scores['mean'][x]['rms']
                y_lim = (0.0, 1.0)
            elif relative_bars == 'smd':
                score /= scores['smd'][x]['rms']
                y_lim = (0.0, 1.0)
            d.append(score)


    except KeyError:
        print(model)
        d = [0,0,0]
    # print(d)
    bar = plt.bar(('main', 'solvent', 'solute'), d,
                color=['#0433FF', '#16E7CF', '#4B1F8C'],
                width=1)
    texts = tuple(f'{scores[model][x]["rho"]:.2f}' for x in ('main', 'solvent', 'solute'))
    plt.ylim(*y_lim)
    plt.axis('off')
    # plt.tight_layout()
    plt.axhline(y_lim[1], color='k', linestyle='--')

    for i, rect in enumerate(bar):
        height = rect.get_height()
        color = 'black'
        if height > 0.99:
            height = 0.95
            color = 'white'
        if height > 0.85:
            height -= 0.12
            color = 'white'
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{texts[i]}', ha='center', va='bottom', color=color)



# plt.figure(figsize=(16, 16))
def plot_NN(scores, save=False, output='/Users/balepka/Downloads/best_Res_v1.png', y_lim=(0., 2.0), relative_bars=None):
    """
    Plots figure with RMS data for all models
    :param scores: dictionary with best models performances
    :param save: whether to save a figure to file
    :param output: path to file to save the figure
    :param y_lim: lower and upper limits of y axis
    :param relative_bars: 'mean' to normalize to RMSD on mean, 'smd' to normalize on SMD results, None to absolute values
    :return: None
    """
    plt.axis('off')
    fig, ax = plt.subplots(10,10, sharex=True, sharey=True, figsize=(12,15))
    plt.tight_layout()
    tk = {'fontfamily': 'Helvetica', 'fontsize': 20, 'weight': 'extra bold'}
    add_headers(fig, row_headers=('Blank', 'Class', 'Comp', 'TESA', 'Morgan', 'Mor2to20', 'JB', 'BoB', 'BAT', 'SOAP'),
                col_headers=('Blank', 'Class', 'Macro', 'MacroX', 'Morgan', '  Mor2to20', 'JB', 'BoB', 'BAT', 'SOAP'),
                rotate_row_headers=True, **tk)
    plt.subplots_adjust(left=0.08,
                        bottom=0.05,
                        right=0.95,
                        top=0.95,
                        wspace=0.05,
                        hspace=0.05)

    for i_u, solute in enumerate(('Blank', 'Class', 'Comp', 'TESA', 'Morgan', 'Mor2to20', 'JB', 'BoB', 'BAT', 'SOAP')):
        for j_s, solvent in enumerate(('Blank', 'Class', 'Macro', 'MacroX', 'Morgan', 'Mor2to20', 'JB', 'BoB', 'BAT', 'SOAP')):
            if relative_bars:
                y_lim = (0., 1.0)
            if solute == 'Blank' and solvent == 'Blank':
                # TODO plot SMD
                plt.subplot(10,10,10*i_u+j_s+1)
                plot_one('smd', scores, y_lim, relative_bars=relative_bars)
            plt.ylim(*[1.1*y for y in y_lim])
            plt.axis('off')
            # plt.tight_layout()
            solute = nicknames[solute.replace('_', '').lower()]
            solvent = nicknames[solvent.replace('_', '').lower()]
            model = (solvent, solute)
            plt.subplot(10,10,10*i_u+j_s+1)
            if not (solute == 'blank' and solvent == 'blank'):
                plot_one(model, scores, y_lim, relative_bars=relative_bars)

    # additional text
    plt.text(-31.3, 10.26, f'SMD', size=20, weight='bold')
    plt.text(-17.2, 10.8, f'SOLVENTS', size=20, weight='bold')
    plt.text(-34.9 , 4.72, f'SOLUTES', size=20, rotation='vertical', weight='bold')

    if save:
        plt.savefig(output)

    plt.show()


# plt.figure(figsize=(16, 16))
def plot_NN_blank(scores, blank, save=False, output='/Users/balepka/Downloads/best_krr_v1.png', y_lim=(-1.0, 1.0)):
    """
    Plots figure with RMS data for all models with respect to blank experiments
    :param scores: dictionary with best models performances
    :param blank: "S" for solvent, "U" for solute
    :param save: whether to save a figure to file
    :param output: path to file to save the figure
    :param y_lim: lower and upper limits of y axis
    :return: None
    """
    assert blank.lower() in ('s', 'u', 'solvent', 'solute')
    if blank.lower() == 'solvent' or blank.lower() == 's':
        blank = 'S'
    elif blank.lower() == 'solute' or blank.lower() == 'u':
        blank = 'U'
    plt.axis('off')
    fig, ax = plt.subplots(9,9, sharex=True, sharey=True, figsize=(12,15))
    plt.tight_layout()
    tk = {'fontfamily': 'Helvetica', 'fontsize': 20, 'weight': 'extra bold'}
    add_headers(fig, row_headers=('Class', 'Comp', 'TESA', 'Morgan', 'Mor2to20', 'JB', 'BoB', 'BAT', 'SOAP'),
                col_headers=('Class', 'Macro', 'MacroX', 'Morgan', 'Mor2to20', 'JB', 'BoB', 'BAT', 'SOAP'),
                rotate_row_headers=True, **tk)
    plt.subplots_adjust(left=0.1,
                        bottom=0.05,
                        right=0.95,
                        top=0.95,
                        wspace=0.05,
                        hspace=0.05)

    for i_u, solute in enumerate(('Class', 'Computed', 'TESA', 'Morgan', 'Morgan_2_2to20', 'JB', 'BoB', 'BAT', 'SOAP')):
        solute = nicknames[solute.replace('_', '').lower()]
        if blank == 'S':
            blanks = [scores[('blank', solute)][x]['rms'] for x in ('main', 'solvent', 'solute')]
            # print(f'blanks: {blanks}')
        for j_s, solvent in enumerate(('Class', 'Macro', 'MacroExtra', 'Morgan', 'Morgan_2_2to20', 'JB', 'BoB', 'BAT', 'SOAP')):
            # print(f'{solvent} - {solute}')
            solvent = nicknames[solvent.replace('_', '').lower()]
            if blank =='U':
                blanks = [scores[(solvent, 'blank')][x]['rms'] for x in ('main', 'solvent', 'solute')]
                # print(f'blanks: {blanks}')
            model = (solvent, solute)
            # plt.ylim(*y_lim)
            plt.axis('off')
            # plt.tight_layout()
            plt.subplot(9,9,9*i_u+j_s+1)
            plot_one_blank(model, blanks, scores, y_lim)
    if save:
        plt.savefig(output)
    plt.show()


if __name__ == '__main__':
    NN = 'KRR'  # Choose NN from 'KRR', 'Lin', 'Res'
    scores_path = f'Tables/Scores_{NN}.pkl'
    with open(project_path(scores_path), 'rb') as f:
        scores = pkl.load(f)

    # Regular
    output = project_path(f'/Examples/results/RMS_{NN}_example.png')  # choose path to file
    plot_NN(scores, save=True, output=output, relative_bars='mean')

    # Blank Solvent
    output = project_path(f'/Examples/results/RMS_{NN}_blankS_example.png')  # choose path to file
    blank = 'S'
    plot_NN_blank(scores, blank, save=True, output=output)

    # Blank Solute
    output = project_path(f'/Examples/results/RMS_{NN}_blankU_example.png')  # choose path to file
    blank = 'U'
    plot_NN_blank(scores, blank, save=True, output=output)