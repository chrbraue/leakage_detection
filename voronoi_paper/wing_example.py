"""
wing_example.py

Application and evaluation of classic and refined Voronoi predictors on wing dataset.

This script loads the wing data from 'data_cleaned.csv' and 'data_filtered.csv'.
Execute 'prepare_data.py' first to generate these files.
Classic and refined Voronoi predictors are evaluated on single- and two-leak data.
Invalid predictions (in case of the refined predictor) are further analyzed.
Finally, the simultaneous search strategy is evaluated.

Author: Christoph Brauer (christoph.brauer@dlr.de)
Date: 2026-04-16
"""

import leak_localization as ll
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings('ignore', message='Ignoring `palette` because no `hue` variable has been assigned.')
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')

flip = True
flip_sign = -1

generate_plots = True


if __name__ == '__main__':

    # ===============================
    # === load and clean the data ===
    # ===============================
    # load vacuum connection positions and nodes of contour
    points_vacuum = np.loadtxt('data/vacuumports.csv')
    points_contour = np.loadtxt('data/contour.csv')
    # load original (only filtered) data
    df = pd.read_csv('data/data_filtered.csv')
    df_sl = df[df['n'] == 1]
    flows_sl = df_sl[[f'mfc{i + 1}' for i in range(len(points_vacuum))]].values
    coords_sl = df_sl[['x1', 'y1']].values
    df_tl = df[df['n'] == 2]
    flows_tl = df_tl[[f'mfc{i + 1}' for i in range(len(points_vacuum))]].values
    coords_tl_1 = df_tl[['x1', 'y1']].values
    coords_tl_2 = df_tl[['x2', 'y2']].values
    # load cleaned data
    df = pd.read_csv('data/data_cleaned.csv')
    df_sl = df[df['n'] == 1]
    flows_sl_cleaned = df_sl[[f'mfc{i + 1}' for i in range(len(points_vacuum))]].values
    coords_sl_cleaned = df_sl[['x1', 'y1']].values
    df_tl = df[df['n'] == 2]
    flows_tl_cleaned = df_tl[[f'mfc{i + 1}' for i in range(len(points_vacuum))]].values
    coords_tl_1_cleaned = df_tl[['x1', 'y1']].values
    coords_tl_2_cleaned = df_tl[['x2', 'y2']].values

    # ========================
    # === compute diagrams ===
    # ========================
    classic_voronoi = ll.ClassicVoronoi(points_vacuum, points_contour)
    refined_voronoi = ll.RefinedVoronoi(points_vacuum, points_contour)

    # ====================================================
    # === initialize dataframe for single-leak results ===
    # ====================================================
    df_results = pd.DataFrame(columns=['Voronoi Type', 'Dataset', 'ACC', 'MED (full data)', 'MED (incorrect only)'])

    # =================================================
    # === single-leak / classic Voronoi experiments ===
    # =================================================
    # evaluation on original data
    results = np.array(['Classic', 'Original',
                        classic_voronoi.evaluate_accuracy(flows_sl, coords_sl, mean=True),
                        classic_voronoi.evaluate_l2_error(flows_sl, coords_sl, mean=True),
                        classic_voronoi.evaluate_l2_error(flows_sl, coords_sl, mean=True, false_only=True)]).reshape(1, -1)
    df_results = pd.concat([df_results, pd.DataFrame(results, columns=df_results.columns)], axis=0)
    # evaluation on cleaned data
    results = np.array(['Classic', 'Cleaned', # evaluation on cleaned data
                        classic_voronoi.evaluate_accuracy(flows_sl_cleaned, coords_sl_cleaned, mean=True),
                        classic_voronoi.evaluate_l2_error(flows_sl_cleaned, coords_sl_cleaned, mean=True),
                        classic_voronoi.evaluate_l2_error(flows_sl_cleaned, coords_sl_cleaned, mean=True, false_only=True)]).reshape(1, -1)
    df_results = pd.concat([df_results, pd.DataFrame(results, columns=df_results.columns)], axis=0)
    if generate_plots:
        # plot confusion matrices
        classic_voronoi.evaluate_confusion_matrix(flows_sl_cleaned, coords_sl_cleaned, normalize='true', include_values=True, figsize=(3.0, 3.0), filename='figures/classic_cm_recall.png')
        classic_voronoi.evaluate_confusion_matrix(flows_sl_cleaned, coords_sl_cleaned, normalize='pred', include_values=True, figsize=(3.0, 3.0), filename='figures/classic_cm_precision.png')
        # plot diagram and cleaned data with projections
        plt.figure(figsize=(6.5, 2.5))
        classic_voronoi.plot_diagram(flip=flip, outer_edges_factor=2, linewidth=1, markersize=3, plot_contour=True, plot_fill=True, color_fill='w', pad_fill=1000, s_fill=0.5, n_fill=5000000)
        palette_data = sns.color_palette('tab10')
        classic_voronoi.plot_projections_by_prediction(flows_sl_cleaned, coords_sl_cleaned, flip=flip, palette_data=palette_data, palette=len(classic_voronoi.indices) * ['k'], linewidth=1, linestyle='--', s=10)
        classic_voronoi.plot_cells(n_points=1000000, flip=flip, palette=palette_data, s=1)
        classic_voronoi.plot_cell_labels(flip=flip, fontsize=12)
        classic_voronoi.plot_point_labels(flip=flip, fontsize=8)
        plt.xticks([])
        plt.yticks([])
        plt.gca().set_aspect('equal', 'box')
        plt.tight_layout()
        plt.box(False)
        plt.savefig('figures/classic_evaluation_single.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    # =================================================
    # === single-leak / refined Voronoi experiments ===
    # =================================================
    # evaluation on original data
    results = np.array(['Refined', 'Original',
                        refined_voronoi.evaluate_accuracy(flows_sl, coords_sl, mean=True),
                        refined_voronoi.evaluate_l2_error(flows_sl, coords_sl, mean=True),
                        refined_voronoi.evaluate_l2_error(flows_sl, coords_sl, mean=True, false_only=True)]).reshape(1, -1)
    df_results = pd.concat([df_results, pd.DataFrame(results, columns=df_results.columns)], axis=0)
    # evaluation on cleaned data
    results = np.array(['Refined', 'Cleaned',
                        refined_voronoi.evaluate_accuracy(flows_sl_cleaned, coords_sl_cleaned, mean=True),
                        refined_voronoi.evaluate_l2_error(flows_sl_cleaned, coords_sl_cleaned, mean=True),
                        refined_voronoi.evaluate_l2_error(flows_sl_cleaned, coords_sl_cleaned, mean=True, false_only=True)]).reshape(1, -1)
    df_results = pd.concat([df_results, pd.DataFrame(results, columns=df_results.columns)], axis=0)
    # display both classic and refined Voronoi results
    print(f'\n===========================================================================================================')
    print('=============== Single-Leak Results =======================================================================')
    print('===========================================================================================================')
    df_results = df_results.astype({'ACC': 'float64', 'MED (full data)': 'float64', 'MED (incorrect only)': 'float64'})
    df_results['ACC'] *= 100
    df_results[['MED (full data)', 'MED (incorrect only)']] /= 10
    print(df_results.to_string(index=False, float_format=lambda x: f'{x:6.2f}'))
    if generate_plots:
        # plot confusion matrices
        refined_voronoi.evaluate_confusion_matrix(flows_sl_cleaned, coords_sl_cleaned, normalize='true', include_values=True, figsize=(4.2, 4.2), filename='figures/refined_cm_recall.png')
        refined_voronoi.evaluate_confusion_matrix(flows_sl_cleaned, coords_sl_cleaned, normalize='pred', include_values=True, figsize=(4.2, 4.2), filename='figures/refined_cm_precision.png')
        # plot diagram and cleaned data with projections
        plt.figure(figsize=(6.5, 2.5))
        refined_voronoi.plot_diagram(flip=flip, outer_edges_factor=2, linewidth=1, markersize=3, plot_contour=True, plot_fill=True, color_fill='w', pad_fill=1000, s_fill=0.5, n_fill=5000000)
        refined_voronoi.plot_projections_by_prediction(flows_sl_cleaned, coords_sl_cleaned, flip=flip, palette=len(refined_voronoi.indices) * ['k'], linewidth=1, linestyle='--', s=10)
        refined_voronoi.plot_cells(n_points=1000000, flip=flip, s=1)
        refined_voronoi.plot_cell_labels(flip=flip, fontsize=10)
        refined_voronoi.plot_point_labels(flip=flip, fontsize=8)
        plt.xticks([])
        plt.yticks([])
        plt.gca().set_aspect('equal', 'box')
        plt.tight_layout()
        plt.box(False)
        plt.savefig('figures/refined_evaluation_single.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    # =================================================
    # === initialize dataframe for two-leak results ===
    # =================================================
    df_results = pd.DataFrame(columns=['Voronoi Type', 'Dataset', 'ACC Step 1', 'ACC Step 2',
                                       'MED (full data) Step 1', 'MED (full data) Step 2',
                                       'MED (incorrect only) Step 1', 'MED (incorrect only) Step 2'])

    # ==============================================
    # === two-leak / classic Voronoi experiments ===
    # ==============================================
    # evaluation on original data
    acc_step_1, acc_step_2 = classic_voronoi.evaluate_accuracy_two_step(flows_sl, coords_sl, flows_tl, coords_tl_1, coords_tl_2, mean=True)
    err_step_1, err_step_2 = classic_voronoi.evaluate_l2_error_two_step(flows_sl, coords_sl, flows_tl, coords_tl_1, coords_tl_2, mean=True)
    err_step_1_fo, err_step_2_fo = classic_voronoi.evaluate_l2_error_two_step(flows_sl, coords_sl, flows_tl, coords_tl_1, coords_tl_2, mean=True, false_only=True)
    results = np.array(['Classic', 'Original', acc_step_1, acc_step_2, err_step_1, err_step_2, err_step_1_fo, err_step_2_fo]).reshape(1, -1)
    df_results = pd.concat([df_results, pd.DataFrame(results, columns=df_results.columns)], axis=0)
    # evaluation on cleaned data
    acc_step_1, acc_step_2 = classic_voronoi.evaluate_accuracy_two_step(flows_sl_cleaned, coords_sl_cleaned, flows_tl_cleaned, coords_tl_1_cleaned, coords_tl_2_cleaned, mean=True)
    err_step_1, err_step_2 = classic_voronoi.evaluate_l2_error_two_step(flows_sl_cleaned, coords_sl_cleaned, flows_tl_cleaned, coords_tl_1_cleaned, coords_tl_2_cleaned, mean=True)
    err_step_1_fo, err_step_2_fo = classic_voronoi.evaluate_l2_error_two_step(flows_sl_cleaned, coords_sl_cleaned, flows_tl_cleaned, coords_tl_1_cleaned, coords_tl_2_cleaned, mean=True, false_only=True)
    results = np.array(['Classic', 'Cleaned', acc_step_1, acc_step_2, err_step_1, err_step_2, err_step_1_fo, err_step_2_fo]).reshape(1, -1)
    df_results = pd.concat([df_results, pd.DataFrame(results, columns=df_results.columns)], axis=0)
    if generate_plots:
        # plot diagram and cleaned data with projections
        l2_errors_1 = classic_voronoi.evaluate_l2_error(flows_tl_cleaned, coords_tl_1_cleaned)
        l2_errors_2 = classic_voronoi.evaluate_l2_error(flows_tl_cleaned, coords_tl_2_cleaned)
        idx_1 = np.where(l2_errors_1 < l2_errors_2)[0]
        idx_2 = np.where(l2_errors_2 < l2_errors_1)[0]
        idx_equal = np.where(l2_errors_1 == l2_errors_2)[0]
        coords_best = np.zeros(coords_tl_1_cleaned.shape)
        coords_best[idx_1] = coords_tl_1_cleaned[idx_1]
        coords_best[idx_2] = coords_tl_2_cleaned[idx_2]
        coords_1_equal = coords_tl_1_cleaned[idx_equal]
        coords_2_equal = coords_tl_2_cleaned[idx_equal]
        plt.figure(figsize=(6.5, 2.5))
        classic_voronoi.plot_diagram(flip=flip, outer_edges_factor=2, linewidth=1, markersize=3, plot_contour=True, plot_fill=True, color_fill='w', pad_fill=1000, s_fill=0.5, n_fill=5000000)
        palette_data = sns.color_palette('tab10')
        classic_voronoi.plot_projections_by_prediction(flows_tl_cleaned[idx_1], coords_tl_1_cleaned[idx_1], flip=flip, palette_data=palette_data, palette=len(classic_voronoi.indices) * ['k'], linewidth=1, linestyle='--', s=10)
        classic_voronoi.plot_projections_by_prediction(flows_tl_cleaned[idx_2], coords_tl_2_cleaned[idx_2], flip=flip, palette_data=palette_data, palette=len(classic_voronoi.indices) * ['k'], linewidth=1, linestyle='--', s=10)
        classic_voronoi.plot_projections_by_prediction(flows_tl_cleaned[idx_equal], coords_tl_2_cleaned[idx_equal], flip=flip, palette_data=palette_data, palette=len(classic_voronoi.indices) * ['k'], linewidth=1, linestyle='--', s=10)
        classic_voronoi.plot_projections_by_prediction(flows_tl_cleaned[idx_equal], coords_tl_2_cleaned[idx_equal], flip=flip, palette_data=palette_data, palette=len(classic_voronoi.indices) * ['k'], linewidth=1, linestyle='--', s=10)
        classic_voronoi.plot_cells(n_points=1000000, flip=flip, palette=palette_data, s=1)
        classic_voronoi.plot_cell_labels(flip=flip, fontsize=12)
        classic_voronoi.plot_point_labels(flip=flip, fontsize=8)
        plt.xticks([])
        plt.yticks([])
        plt.gca().set_aspect('equal', 'box')
        plt.tight_layout()
        plt.box(False)
        plt.savefig('figures/classic_evaluation_multi.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    # ==============================================
    # === two-leak / refined Voronoi experiments ===
    # ==============================================
    # evaluation on original data
    acc_step_1, acc_step_2 = refined_voronoi.evaluate_accuracy_two_step(flows_sl, coords_sl, flows_tl, coords_tl_1, coords_tl_2, mean=True)
    err_step_1, err_step_2 = refined_voronoi.evaluate_l2_error_two_step(flows_sl, coords_sl, flows_tl, coords_tl_1, coords_tl_2, mean=True)
    err_step_1_fo, err_step_2_fo = refined_voronoi.evaluate_l2_error_two_step(flows_sl, coords_sl, flows_tl, coords_tl_1, coords_tl_2, mean=True, false_only=True)
    results = np.array(['Refined', 'Original', acc_step_1, acc_step_2, err_step_1, err_step_2, err_step_1_fo, err_step_2_fo]).reshape(1, -1)
    df_results = pd.concat([df_results, pd.DataFrame(results, columns=df_results.columns)], axis=0)
    # evaluation on cleaned data
    acc_step_1, acc_step_2 = refined_voronoi.evaluate_accuracy_two_step(flows_sl_cleaned, coords_sl_cleaned, flows_tl_cleaned, coords_tl_1_cleaned, coords_tl_2_cleaned, mean=True)
    err_step_1, err_step_2 = refined_voronoi.evaluate_l2_error_two_step(flows_sl_cleaned, coords_sl_cleaned, flows_tl_cleaned, coords_tl_1_cleaned, coords_tl_2_cleaned, mean=True)
    err_step_1_fo, err_step_2_fo = refined_voronoi.evaluate_l2_error_two_step(flows_sl_cleaned, coords_sl_cleaned, flows_tl_cleaned, coords_tl_1_cleaned, coords_tl_2_cleaned, mean=True, false_only=True)
    results = np.array(['Refined', 'Cleaned', acc_step_1, acc_step_2, err_step_1, err_step_2, err_step_1_fo, err_step_2_fo]).reshape(1, -1)
    df_results = pd.concat([df_results, pd.DataFrame(results, columns=df_results.columns)], axis=0)
    # display both classic and refined Voronoi results
    print(f'\n===========================================================================================================')
    print('=============== Two-Leak Results ==========================================================================')
    print('===========================================================================================================')
    df_results = df_results.astype({'ACC Step 1': 'float64', 'ACC Step 2': 'float64', 'MED (full data) Step 1': 'float64', 'MED (full data) Step 2': 'float64', 'MED (incorrect only) Step 1': 'float64', 'MED (incorrect only) Step 2': 'float64'})
    df_results[['ACC Step 1', 'ACC Step 2']] *= 100
    df_results[['MED (full data) Step 1', 'MED (full data) Step 2', 'MED (incorrect only) Step 1', 'MED (incorrect only) Step 2']] /= 10
    print(df_results.to_string(index=False, float_format=lambda x: f'{x:6.2f}'))
    if generate_plots:
        # plot diagram and cleaned data with projections
        l2_errors_1 = refined_voronoi.evaluate_l2_error(flows_tl_cleaned, coords_tl_1_cleaned)
        l2_errors_2 = refined_voronoi.evaluate_l2_error(flows_tl_cleaned, coords_tl_2_cleaned)
        idx_1 = np.where(l2_errors_1 < l2_errors_2)[0]
        idx_2 = np.where(l2_errors_2 < l2_errors_1)[0]
        idx_equal = np.where(l2_errors_1 == l2_errors_2)[0]
        coords_best = np.zeros(coords_tl_1_cleaned.shape)
        coords_best[idx_1] = coords_tl_1_cleaned[idx_1]
        coords_best[idx_2] = coords_tl_2_cleaned[idx_2]
        coords_1_equal = coords_tl_1_cleaned[idx_equal]
        coords_2_equal = coords_tl_2_cleaned[idx_equal]
        plt.figure(figsize=(6.5, 2.5))
        refined_voronoi.plot_diagram(flip=flip, outer_edges_factor=2, linewidth=1, markersize=3, plot_contour=True, plot_fill=True, color_fill='w', pad_fill=1000, s_fill=0.5, n_fill=5000000)
        refined_voronoi.plot_projections_by_prediction(flows_tl_cleaned[idx_1], coords_tl_1_cleaned[idx_1], flip=flip, palette=len(refined_voronoi.indices) * ['k'], linewidth=1, linestyle='--', s=10)
        refined_voronoi.plot_projections_by_prediction(flows_tl_cleaned[idx_2], coords_tl_2_cleaned[idx_2], flip=flip, palette=len(refined_voronoi.indices) * ['k'], linewidth=1, linestyle='--', s=10)
        refined_voronoi.plot_projections_by_prediction(flows_tl_cleaned[idx_equal], coords_tl_2_cleaned[idx_equal], flip=flip, palette=len(refined_voronoi.indices) * ['k'], linewidth=1, linestyle='--', s=10)
        refined_voronoi.plot_projections_by_prediction(flows_tl_cleaned[idx_equal], coords_tl_2_cleaned[idx_equal], flip=flip, palette=len(refined_voronoi.indices) * ['k'], linewidth=1, linestyle='--', s=10)
        refined_voronoi.plot_cells(n_points=1000000, flip=flip, s=1)
        refined_voronoi.plot_cell_labels(flip=flip, fontsize=10)
        refined_voronoi.plot_point_labels(flip=flip, fontsize=8)
        plt.xticks([])
        plt.yticks([])
        plt.gca().set_aspect('equal', 'box')
        plt.tight_layout()
        plt.box(False)
        plt.savefig('figures/refined_evaluation_multi.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    # ==========================================================
    # === plot invalid prediction coords in single-leak case ===
    # ==========================================================
    if generate_plots:
        idx_none = refined_voronoi.predict(flows_sl_cleaned) == -1
        plt.figure(figsize=(6.5, 2.5))
        refined_voronoi.plot_diagram(flip=flip, outer_edges_factor=2, linewidth=1, markersize=3, plot_contour=True, plot_fill=True, color_fill='w', pad_fill=1000, s_fill=0.5, n_fill=5000000)
        plt.scatter(x=coords_sl_cleaned[idx_none, 0], y=flip_sign * coords_sl_cleaned[idx_none, 1], s=20, c='b', marker='x', zorder=4)
        refined_voronoi.plot_cell_labels(flip=flip, fontsize=10)
        refined_voronoi.plot_point_labels(flip=flip, fontsize=8)
        plt.xticks([])
        plt.yticks([])
        plt.gca().set_aspect('equal', 'box')
        plt.tight_layout()
        plt.box(False)
        plt.savefig('figures/none_predictions_single.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    # ====================================================
    # === analyze invalid predictions in two-leak case ===
    # ====================================================
    pred_cat = refined_voronoi.predict(flows_tl_cleaned)
    idx_invalid = pred_cat == -1
    print(f'\n===========================================================================================================')
    print('=============== Refined Voronoi Step 1 Accuracy w/o invalid predictions ===================================')
    print('===========================================================================================================')
    print(refined_voronoi.evaluate_accuracy_two_step(flows_sl_cleaned, coords_sl_cleaned, flows_tl_cleaned, coords_tl_1_cleaned, coords_tl_2_cleaned, mean=True, ignore_invalid=True)[0])
    # plot histogram of invalid predictions and leak locations corresponding to most frequent categories
    if generate_plots:
        pred_tuple = [(i + 1, j + 1) for i, j in refined_voronoi.predict_tuples(flows_tl_cleaned)]
        plt.figure(figsize=(1.25 * 3.5, 1.25 * 7.5))
        pd.Series(pred_tuple)[pred_cat == -1].value_counts().plot.barh()
        plt.xlabel('Frequency')
        plt.ylabel('Invalid prediction category')
        plt.xticks([0, 2, 4, 6, 8, 10])
        plt.tight_layout()
        plt.savefig('figures/refined_invalid_hist.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        for t in [(1, 3), (5, 2), (10, 1)]:
            rows = [i for i, val in enumerate(pred_tuple) if val == t]
            coords = np.vstack([coords_tl_1_cleaned[rows], coords_tl_2_cleaned[rows]])
            plt.figure(figsize=(6.5, 2.5))
            refined_voronoi.plot_diagram(flip=flip, outer_edges_factor=2, linewidth=1, markersize=3, plot_contour=True, plot_fill=True, color_fill='w', pad_fill=1000, s_fill=0.5, n_fill=5000000)
            refined_voronoi.plot_cell_labels(flip=flip, fontsize=10)
            plt.scatter(x=coords[:, 0], y=flip_sign * coords[:, 1], s=20, c='b', marker='x', zorder=4)
            plt.plot((coords_tl_1_cleaned[rows, 0], coords_tl_2_cleaned[rows, 0]), (flip_sign * coords_tl_1_cleaned[rows, 1], flip_sign * coords_tl_2_cleaned[rows, 1]), 'r', linewidth=0.5)
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
            plt.savefig(f'figures/refined_invalid_{t}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.show()

    # ================================================================
    # === analyze simultaneous strategy on invalid prediction data ===
    # ================================================================
    print(f'\n===========================================================================================================')
    print('=============== Simultaneous Strategy on Invalid Prediction Data ==========================================')
    print('===========================================================================================================')
    labels_tl_1 = classic_voronoi.categorical_labels(coords_tl_1_cleaned[idx_invalid])
    labels_tl_2 = classic_voronoi.categorical_labels(coords_tl_2_cleaned[idx_invalid])
    labels_tl = list(zip(labels_tl_1, labels_tl_2))
    predictions_tl = [tuple(a) for a in np.argsort(-flows_tl_cleaned[idx_invalid], axis=1)[:, :2]]
    idx_tp_1 = np.zeros(len(labels_tl), dtype=bool)
    idx_tp_2 = np.zeros(len(labels_tl), dtype=bool)
    idx_true_1 = np.zeros(len(labels_tl), dtype=bool)
    idx_true_2 = np.zeros(len(labels_tl), dtype=bool)
    for i in range(len(labels_tl)):
        idx_tp_1[i] = predictions_tl[i][0] in labels_tl[i]
        idx_tp_2[i] = predictions_tl[i][1] in labels_tl[i]
        idx_true_1[i] = labels_tl[i][0] in predictions_tl[i]
        idx_true_2[i] = labels_tl[i][1] in predictions_tl[i]
    rows = []
    def add_row(category, description, count):
        rows.append({'# detected leaks': category, 'case': description, 'count': count, 'percent': count / n * 100})
    n = len(labels_tl)
    add_row('0', 'both cells wrong', np.sum(~(idx_tp_1 | idx_tp_2)))
    add_row('1', 'cell 1 correct', np.sum((idx_tp_1 & ~idx_tp_2) & (idx_true_1 ^ idx_true_2)))
    add_row('1', 'cell 2 correct', np.sum((~idx_tp_1 & idx_tp_2) & (idx_true_1 ^ idx_true_2)))
    add_row('2', 'both cells correct', np.sum(idx_tp_1 & idx_tp_2))
    add_row('2', 'cell 1 contains both', np.sum((idx_tp_1 & ~idx_tp_2) & (idx_true_1 & idx_true_2)))
    add_row('2', 'cell 2 contains both', np.sum((~idx_tp_1 & idx_tp_2) & (idx_true_1 & idx_true_2)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f'{x:6.2f}'))

    # ================================================================
    # === analyze simultaneous strategy on invalid prediction data ===
    # ================================================================
    print(f'\n===========================================================================================================')
    print('=============== Simultaneous Strategy on Full Two-Leak Data ===============================================')
    print('===========================================================================================================')
    labels_tl_1 = classic_voronoi.categorical_labels(coords_tl_1_cleaned)
    labels_tl_2 = classic_voronoi.categorical_labels(coords_tl_2_cleaned)
    labels_tl = list(zip(labels_tl_1, labels_tl_2))
    predictions_tl = [tuple(a) for a in np.argsort(-flows_tl_cleaned, axis=1)[:, :2]]
    idx_tp_1 = np.zeros(len(labels_tl), dtype=bool)
    idx_tp_2 = np.zeros(len(labels_tl), dtype=bool)
    idx_true_1 = np.zeros(len(labels_tl), dtype=bool)
    idx_true_2 = np.zeros(len(labels_tl), dtype=bool)
    for i in range(len(labels_tl)):
        idx_tp_1[i] = predictions_tl[i][0] in labels_tl[i]
        idx_tp_2[i] = predictions_tl[i][1] in labels_tl[i]
        idx_true_1[i] = labels_tl[i][0] in predictions_tl[i]
        idx_true_2[i] = labels_tl[i][1] in predictions_tl[i]
    rows = []
    def add_row(category, description, count):
        rows.append({'# detected leaks': category, 'case': description, 'count': count, 'percent': count / n * 100})
    n = len(labels_tl)
    add_row('0', 'both cells wrong', np.sum(~(idx_tp_1 | idx_tp_2)))
    add_row('1', 'cell 1 correct', np.sum((idx_tp_1 & ~idx_tp_2) & (idx_true_1 ^ idx_true_2)))
    add_row('1', 'cell 2 correct', np.sum((~idx_tp_1 & idx_tp_2) & (idx_true_1 ^ idx_true_2)))
    add_row('2', 'both cells correct', np.sum(idx_tp_1 & idx_tp_2))
    add_row('2', 'cell 1 contains both', np.sum((idx_tp_1 & ~idx_tp_2) & (idx_true_1 & idx_true_2)))
    add_row('2', 'cell 2 contains both', np.sum((~idx_tp_1 & idx_tp_2) & (idx_true_1 & idx_true_2)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f'{x:6.2f}'))
