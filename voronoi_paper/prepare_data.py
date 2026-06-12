"""
prepare_data.py

Filtering of the raw data and outlier removal.

The raw data contains examples that were incorrectly recorded manually,
single-leak examples whose locations do not appear in any two-leak examples,
and two-leak examples in which neither location appears in a single-leak example.
These examples are removed.
The remaining examples are cleaned from outliers and the final cleaned dataset is stored in a csv file.

Author: Christoph Brauer (christoph.brauer@dlr.de)
Date: 2026-04-16
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from leak_localization import ClassicVoronoi

warnings.filterwarnings('ignore', message="Ignoring `palette` because no `hue` variable has been assigned.")


if __name__ == "__main__":

    # === step 1: data cleaning (get coherent one- and two-leak datasets)
    # === keeps 853 of 957 samples (the 413 single- and 410 two-leak samples mentioned in Section 3.1 of the paper)

    # remove incorrectly recorded data (out of coordinate system)
    df_filtered = pd.read_csv('data/data.csv')
    df_filtered = df_filtered[~(df_filtered['y1'] < 99) & ~(df_filtered['y2'] < 99)]

    # store original index in additional column
    df_filtered['_orig_index'] = df_filtered.index

    # get coordinates of all single-leak examples
    single_mask = df_filtered['x2'].isna() | df_filtered['y2'].isna()
    single_pairs = set(zip(df_filtered.loc[single_mask, 'x1'], df_filtered.loc[single_mask, 'y1']))

    # get all two-leak examples where both locations also appear in a single-leak example
    two_mask = ~single_mask
    two_mask_keep = df_filtered[two_mask].apply(lambda r: (r['x1'], r['y1']) in single_pairs and (r['x2'], r['y2']) in single_pairs, axis=1)
    df_two_kept = df_filtered[two_mask][two_mask_keep]

    # get all single_leak examples whose location appears in at least one two-leak example
    needed_single_pairs = (set(zip(df_two_kept['x1'], df_two_kept['y1'])) | set(zip(df_two_kept['x2'], df_two_kept['y2'])))
    single_mask_keep = df_filtered[single_mask].apply(lambda r: (r['x1'], r['y1']) in needed_single_pairs, axis=1)
    df_single_kept = df_filtered[single_mask][single_mask_keep]

    # concatenate kept single- and two-leak examples in one dataframe, restore original order and reset index
    df_filtered = pd.concat([df_two_kept, df_single_kept])
    df_filtered = df_filtered.sort_values('_orig_index').drop(columns='_orig_index')

    df_filtered.to_csv('data/data_filtered.csv', index=False)

    # === step 2: outlier removal (Voronoi- and threshold-based)
    # === removes additional 5 single- and 1 two-leak examples by Voronoi-based outlier detection
    # === subsequently removes 7 more two-leak examples containing at least one remove single-leak coordinate
    # === overall: keeps 840 of remaining 853 samples

    # load vacuum connection positions and nodes of contour
    points_vacuum = np.loadtxt('data/vacuumports.csv')
    points_contour = np.loadtxt('data/contour.csv')
    classic_voronoi = ClassicVoronoi(points_vacuum, points_contour)

    # split the data
    df_sl = df_filtered[df_filtered['n'] == 1].copy()
    df_tl = df_filtered[df_filtered['n'] == 2].copy()

    # prepare inputs for Voronoi
    flows_sl = df_sl[[f'mfc{i + 1}' for i in range(len(points_vacuum))]].values
    coords_sl = df_sl[['x1', 'y1']].values

    flows_tl = df_tl[[f'mfc{i + 1}' for i in range(len(points_vacuum))]].values
    coords_tl_1 = df_tl[['x1', 'y1']].values
    coords_tl_2 = df_tl[['x2', 'y2']].values

    # single-leak outlier removal
    _, _, idx_sl_cleaned = classic_voronoi.remove_outliers(flows_sl, coords_sl, threshold=2000)
    flows_sl_outliers, coords_sl_outliers, _ = classic_voronoi.remove_outliers(flows_sl, coords_sl, threshold=2000, return_outliers=True)
    df_sl_cleaned = df_sl.iloc[idx_sl_cleaned].copy()

    # two-leak outlier removal
    _, _, _, idx_tl_cleaned = classic_voronoi.remove_outliers(flows_tl, coords_tl_1, coords_tl_2, threshold=2000)
    flows_tl_outliers, coords_tl_1_outliers, coords_tl_2_outliers, _ = classic_voronoi.remove_outliers(flows_tl, coords_tl_1, coords_tl_2, threshold=2000, return_outliers=True)
    df_tl_cleaned = df_tl.iloc[idx_tl_cleaned].copy()

    # remove two-leak rows affected by single-leak outliers
    outlier_set = set(map(tuple, coords_sl_outliers))
    mask_remove = df_tl_cleaned.apply(lambda r: (r['x1'], r['y1']) in outlier_set or (r['x2'], r['y2']) in outlier_set, axis=1)
    df_tl_cleaned = df_tl_cleaned[~mask_remove]

    # combine cleaned datasets and save
    df_cleaned = pd.concat([df_sl_cleaned, df_tl_cleaned])
    df_cleaned = df_cleaned.sort_index().reset_index(drop=True)
    df_cleaned.to_csv('data/data_cleaned.csv', index=False)

    # === step 3 (optional): plot data

    flip = True
    flip_sign = -1

    # plot all single-leak examples (Figure 5 in paper)
    plt.figure(figsize=(6.5, 2.5))
    classic_voronoi.plot_diagram(flip=flip, outer_edges_factor=2, plot_edges=False, linewidth=1, markersize=3, plot_contour=True, plot_fill=True, color_fill='w', pad_fill=1000)
    classic_voronoi.plot_data(flows_sl, coords_sl, evaluate=False, flip=flip, color='blue', s=5)
    for i, point in enumerate(points_vacuum):
        if i < 5:
            plt.text(point[0], flip_sign * point[1] + 500, fr'$\mathbf{{p}}_{{{i + 1}}}$', fontsize=8, ha='center', va='top')
        else:
            plt.text(point[0], flip_sign * point[1] - 500, fr'$\mathbf{{p}}_{{{i + 1}}}$', fontsize=8, ha='center', va='bottom')
    x_range = np.arange(180, 16048, 250)
    y_range = np.arange(100, 5233, 250)
    X, Y = np.meshgrid(x_range, y_range)
    plt.hlines(flip_sign * 0, 0, 16048, linestyle='dashed', linewidth=1, color='gray')
    plt.hlines(flip_sign * 2600, 0, 16048, linestyle='dashed', linewidth=1, color='gray')
    plt.hlines(flip_sign * 5233, 0, 16048, linestyle='dashed', linewidth=1, color='gray')
    plt.vlines(0, 0, flip_sign * 5233, linestyle='dashed', linewidth=1, color='gray')
    plt.vlines(7930, 0, flip_sign * 5233, linestyle='dashed', linewidth=1, color='gray')
    plt.vlines(16048, 0, flip_sign * 5233, linestyle='dashed', linewidth=1, color='gray')
    plt.text(50, flip_sign * -175, '$y_2 = y_1 = 0$', fontsize=8)
    plt.text(50, flip_sign * (2600 + 1450), '$y_2 = 2600$', fontsize=8, rotation=-90)
    plt.text(50, flip_sign * (5233 + 350), '$y_2 = 5233$', fontsize=8)
    plt.text(7930 - 1500, flip_sign * -175, '$y_1=7930$', fontsize=8)
    plt.text(16048 - 1675, flip_sign * -175, '$y_1=16048$', fontsize=8)
    plt.scatter(X, flip_sign * Y, color='lightgray', s=0.25)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.box(False)
    plt.savefig('figures/data.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    # plot Voronoi diagram and outliers with projections (Figure 6 in paper)
    plt.figure(figsize=(6.5, 2.5))
    classic_voronoi.plot_diagram(flip=flip, outer_edges_factor=2, linewidth=1, markersize=3, plot_contour=True, plot_fill=True, color_fill='w', pad_fill=1000, s_fill=0.5, n_fill=5000000)
    palette_data = sns.color_palette('tab10')
    classic_voronoi.plot_projections_by_prediction(flows_sl_outliers, coords_sl_outliers, flip=flip, palette_data=palette_data, palette=len(classic_voronoi.indices) * ['k'], linewidth=1, linestyle='--', s=10)
    classic_voronoi.plot_projections_by_prediction(flows_tl_outliers, coords_tl_1_outliers, flip=flip, palette_data=palette_data, palette=len(classic_voronoi.indices) * ['r'], linewidth=1, linestyle='--', s=10)
    classic_voronoi.plot_projections_by_prediction(flows_tl_outliers, coords_tl_2_outliers, flip=flip, palette_data=palette_data, palette=len(classic_voronoi.indices) * ['r'], linewidth=1, linestyle='--', s=10)
    plt.plot((coords_tl_1_outliers[:, 0], coords_tl_2_outliers[:, 0]), (flip_sign * coords_tl_1_outliers[:, 1], flip_sign * coords_tl_2_outliers[:, 1]), 'r', linewidth=0.5)
    classic_voronoi.plot_cells(n_points=1000000, flip=flip, palette=palette_data, s=1)
    classic_voronoi.plot_cell_labels(flip=flip, fontsize=12)
    for i, point in enumerate(points_vacuum):
        if i < 5:
            plt.text(point[0], flip_sign * point[1] + 500, fr'$\mathbf{{p}}_{{{i + 1}}}$', fontsize=8, ha='center', va='top')
        else:
            plt.text(point[0], flip_sign * point[1] - 500, fr'$\mathbf{{p}}_{{{i + 1}}}$', fontsize=8, ha='center', va='bottom')
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.box(False)
    plt.savefig('figures/outliers.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
