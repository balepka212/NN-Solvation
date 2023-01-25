from collections import OrderedDict
import numpy as np
from config import project_path
import torch
import os
import pickle as pkl
import matplotlib.pyplot as plt

NN ='Res'

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

def plot_one_blank(folder, blanks, best_NN, y_lim=(-1.0, 1.0)):
    try:
        d2 = [best_NN[folder][x] for x in ('val', 'solvent', 'solute')]
        # print(f'd: {d2}')
        d = [d2[i]-blanks[i] for i in range(3)]
    except KeyError:
        d = [0,0,0]
    # print(f'd-b: {d}')

    c = [int(x<0) for x in d]
    # print(f'c: {c}')
    two_colors = [('#EE220C','#0433FF'), ('#FF9300' , '#16E7CF'), ('#D41876' , '#4B1F8C')]
    colors = [two_colors[i][c[i]] for i in range(3)]

    plt.bar(('val', 'solvent', 'solute'), d,
                color=colors,
                width=1)
    plt.ylim(*y_lim)
    plt.axis('off')
    # plt.tight_layout()
    # plt.axhline(2, color='k', linestyle='--')

def plot_one(folder, best_NN, y_lim=(0.0, 2.0)):
    try:
        d = []
        for x in ('val', 'solvent', 'solute'):
            d.append(best_NN[folder][x])

    except KeyError:
        d = [0,0,0]
    plt.bar(('val', 'solvent', 'solute'), d,
                color=['#0433FF', '#16E7CF', '#4B1F8C'],
                width=1)
    plt.ylim(*y_lim)
    plt.axis('off')
    # plt.tight_layout()
    plt.axhline(y_lim[1], color='k', linestyle='--')



# plt.figure(figsize=(16, 16))
def plot_NN(best_NN, save=False, output='/Users/balepka/Downloads/best_Res_v1.png', y_lim=(0., 2.0)):
    plt.axis('off')
    fig, ax = plt.subplots(10,10, sharex=True, sharey=True, figsize=(12,15))
    plt.tight_layout()
    tk = {'fontfamily': 'Helvetica', 'fontsize': 20, 'weight': 'extra bold'}
    add_headers(fig, row_headers=('Blank', 'Class', 'Comp', 'TESA', 'Morgan', 'Mor2to20', 'JB', 'BoB', 'BAT', 'SOAP'),
                col_headers=('Blank', 'Class', 'Macro', 'MacroX', 'Morgan', 'Mor2to20', 'JB', 'BoB', 'BAT', 'SOAP'),
                rotate_row_headers=True, **tk)
    plt.subplots_adjust(left=0.08,
                        bottom=0.05,
                        right=0.95,
                        top=0.95,
                        wspace=0.05,
                        hspace=0.05)

    for i_u, solute in enumerate(('Blank', 'Class', 'Computed', 'TESA', 'Morgan', 'Morgan_2_2to20', 'JB', 'BoB', 'BAT', 'SOAP')):
        for j_s, solvent in enumerate(('Blank', 'Class', 'Macro', 'MacroExtra', 'Morgan', 'Morgan_2_2to20', 'JB', 'BoB', 'BAT', 'SOAP')):
            if solute == 'Blank' and solvent == 'Blank':
                # TODO plot SMD
                plt.subplot(10,10,10*i_u+j_s+1)
                plt.bar(('val', 'solvent', 'solute'), (y_lim[1],y_lim[1],y_lim[1]),
                    color=['#0433FF', '#16E7CF', '#4B1F8C'],
                    width=1)
            plt.ylim(*[1.1*y for y in y_lim])
            plt.axis('off')
            # plt.tight_layout()
            folder = f'{solvent}_{solute}_{NN}1'
            if folder not in best_NN:
                folder = f'{solvent}_{solute}_{NN}2'
            if folder not in best_NN:
                folder = f'{solvent}_{solute}_{NN}1b'
            if folder not in best_NN:
                print(f'Absent: {solvent}_{solute}_{NN}')
            plt.subplot(10,10,10*i_u+j_s+1)
            plot_one(folder, best_NN, y_lim)
    if save:
        plt.savefig(output)
    plt.show()


output = '/Users/balepka/Downloads/best_Res_v1.png'

with open(project_path('Other/best_Res1.pkl'), 'rb') as f:
    best_NN = pkl.load(f)
plot_NN(best_NN, save=True, output=output)



# plt.figure(figsize=(16, 16))
def plot_NN_blank(best_NN, blank, save=False, output='/Users/balepka/Downloads/best_krr_v1.png', y_lim=(-1.0, 1.0)):
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
        if blank == 'S':
            blanks = [best_NN[f'Blank_{solute}_{NN}1'][x] for x in ('val', 'solvent', 'solute')]
            # print(f'blanks: {blanks}')
        for j_s, solvent in enumerate(('Class', 'Macro', 'MacroExtra', 'Morgan', 'Morgan_2_2to20', 'JB', 'BoB', 'BAT', 'SOAP')):
            # print(f'{solvent} - {solute}')
            if blank =='U':
                blanks = [best_NN[f'{solvent}_Blank_{NN}1'][x] for x in ('val', 'solvent', 'solute')]
                # print(f'blanks: {blanks}')
            if solute == 'Blank' and solvent == 'Blank':
                plt.subplot(9,9,9*i_u+j_s+1)
                plt.bar(('val', 'solvent', 'solute'), (2,2,2),
                    color=['#0433FF', '#16E7CF', '#4B1F8C'],
                    width=1)
            # plt.ylim(*y_lim)
            plt.axis('off')
            # plt.tight_layout()
            folder = f'{solvent}_{solute}_{NN}1'
            plt.subplot(9,9,9*i_u+j_s+1)
            plot_one_blank(folder, blanks, best_NN, y_lim)
    if save:
        plt.savefig(output)
    plt.show()

#
# if __name__ == '__main__':
#     output = '/Users/balepka/Downloads/best_Res_bS_1.png'
#
#     with open(project_path('Other/best_Res1.pkl'), 'rb') as f:
#         best_NN = pkl.load(f)
#     blank = 'S'
#     plot_NN_blank(best_NN, blank, save=True, output=output)