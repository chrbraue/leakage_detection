"""
leak_localization.py

Classic and refined Voronoi classes for leak localization.

This file contains two classes for classic and refined Voronoi diagrams and predictors.
The classes include methods for are construction, prediction, evaluation, and plotting.

Author: Christoph Brauer (christoph.brauer@dlr.de)
Date: 2026-04-16
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns

from itertools import combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import ConvexHull
from sklearn.metrics import confusion_matrix


standard_palette = sns.color_palette('tab20', 20) + sns.color_palette('Set1', 9) + sns.color_palette('Set2', 8) + sns.color_palette('Set3', 12)


class ClassicVoronoi:

    def __init__(self, points, points_contour=None, points_fill=None, tol=1e-10):
        self.points = points
        self.points_contour = points_contour
        self.points_fill = points_fill
        self.tol = tol
        self.edges_inner = []
        self.edges_outer = []
        self.normals = []
        self.offsets = []
        self.normals_contour = None
        self.offsets_contour = None
        self.nfacets = []
        self.circumcenters = []
        self.triangles = []
        self.triangle_edges_inner = []
        self.triangle_edges_outer = []
        self.triangle_edges_map = {}
        self.order = 1
        self.indices = [(i,) for i in range(self.points.shape[0])]
        self.classification_model = None
        self.compute_diagram()
        self.compute_hyperplanes()

    def compute_diagram(self):
        # delaunay triangulation
        points_3d = np.c_[self.points, np.sum(self.points ** 2, axis=1)]
        hull = ConvexHull(points_3d)
        for simplex, equation in zip(hull.simplices, hull.equations):
            if equation[2] < 0:
                self.triangles.append(simplex)

        # delaunay edges to triangles map
        for i, simplex in enumerate(self.triangles):
            for edge in combinations(simplex, 2):
                edge_key = tuple(sorted(edge))
                if edge_key in self.triangle_edges_map:
                    self.triangle_edges_map[edge_key].append(i)
                else:
                    self.triangle_edges_map[edge_key] = [i]

        # delaunay triangles circumcenters
        for simplex in self.triangles:
            p0, p1, p2 = self.points[simplex]
            tmp = 2 * (p0[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1]))
            cx = ((p0[0] ** 2 + p0[1] ** 2) * (p1[1] - p2[1]) + (p1[0] ** 2 + p1[1] ** 2) * (p2[1] - p0[1]) + (p2[0] ** 2 + p2[1] ** 2) * (p0[1] - p1[1])) / tmp
            cy = ((p0[0] ** 2 + p0[1] ** 2) * (p2[0] - p1[0]) + (p1[0] ** 2 + p1[1] ** 2) * (p0[0] - p2[0]) + (p2[0] ** 2 + p2[1] ** 2) * (p1[0] - p0[0])) / tmp
            self.circumcenters.append([cx, cy])

        # delaunay triangulation inner and outer edges
        for edge, triangles in self.triangle_edges_map.items():
            if len(triangles) < 2:
                self.triangle_edges_outer.append(edge)
            elif np.linalg.norm(np.array(self.circumcenters[triangles[0]]) - np.array(self.circumcenters[triangles[1]])) > self.tol:
                self.triangle_edges_inner.append(edge)
                # each delaunay triangulation inner edge has a related voronoi diagram inner edge
                self.edges_inner.append(tuple(triangles))

        # voronoi diagram outer edges (outer normals of delaunay triangulation outer edges)
        for vp0, vp1 in self.triangle_edges_outer:
            i = self.triangle_edges_map[(vp0, vp1)][0]
            vp2 = (set(self.triangles[i]) - {vp0, vp1}).pop()
            vp01 = self.points[vp1] - self.points[vp0]
            vp02 = self.points[vp2] - self.points[vp0]
            normal = np.array([-vp01[1], vp01[0]])
            if np.dot(normal, vp02) > 0:
                normal = -normal
            self.edges_outer.append(normal)

    def compute_hyperplanes(self):
        for i in range(len(self.points)):
            idx_inner = list(np.where([i in edge for edge in self.triangle_edges_inner])[0])
            idx_outer = list(np.where([i in edge for edge in self.triangle_edges_outer])[0])
            normals = np.zeros((len(idx_inner) + len(idx_outer), 2))
            offsets = np.zeros((len(idx_inner) + len(idx_outer), 1))
            n = 0
            for idx, edges in zip((idx_inner, idx_outer), (self.triangle_edges_inner, self.triangle_edges_outer)):
                if idx:
                    for j in idx:
                        p0, p1 = edges[j]
                        normals[n] = (self.points[p0] - self.points[p1]) / np.linalg.norm(self.points[p0] - self.points[p1])
                        offsets[n] = -np.inner(normals[n], (self.points[p0] + self.points[p1]) / 2)
                        if np.inner(normals[n], self.points[i]) + offsets[n] > 0:
                            normals[n] *= -1
                            offsets[n] *= -1
                        n += 1
            self.normals.append(normals)
            self.offsets.append(offsets)
            self.nfacets.append(n)

    def compute_contour_hyperplanes(self):
        assert(self.points_contour is not None)
        self.normals_contour = np.zeros((len(self.points_contour), 2))
        self.offsets_contour = np.zeros((len(self.points_contour), 1))
        for i in range(-1, len(self.points_contour) - 1):
            d = self.points_contour[i + 1] - self.points_contour[i]
            d_rot = np.array([d[1], -d[0]])
            self.normals_contour[i] = d_rot / np.linalg.norm(d_rot)
            self.offsets_contour[i] = -np.inner(self.normals_contour[i], (self.points_contour[i + 1] + self.points_contour[i]) / 2)
            if np.inner(self.normals_contour[i], np.mean(self.points, axis=0)) + self.offsets_contour[i] > 0:
                self.normals_contour[i] *= -1
                self.offsets_contour[i] *= -1

    def binary_labels(self, coords, cell_index):
        return np.prod(self.normals[cell_index] @ coords.T + self.offsets[cell_index] <= 0, axis=0)

    def categorical_labels(self, coords):
        tmp = np.zeros((len(self.normals), coords.shape[0]))
        for i in range(len(self.normals)):
            tmp[i, :] = self.binary_labels(coords, i)
        cells = np.argmax(tmp, axis=0)
        cells[np.max(tmp, axis=0) == 0] = -1
        return cells

    def predict(self, flows):
        assert flows.shape[1] == len(self.points)
        top_indices = np.argsort(-flows, axis=1)[:, :self.order]
        predictions = [self.indices.index(tuple(item)) if tuple(item) in self.indices else -1 for item in top_indices]
        return np.array(predictions)

    def predict_tuples(self, flows):
        assert flows.shape[1] == len(self.points)
        top_indices = np.argsort(-flows, axis=1)[:, :self.order]
        predictions = [tuple(item) for item in top_indices]
        return predictions

    def split_by_prediction(self, flows, coords):
        predictions = self.predict(flows)
        idx_by_prediction = [np.where(predictions == i)[0] for i in range(len(self.indices))]
        flows_by_prediction = [flows[idx] for idx in idx_by_prediction]
        coords_by_prediction = [coords[idx] for idx in idx_by_prediction]
        return flows_by_prediction, coords_by_prediction, idx_by_prediction

    def projections_by_prediction(self, coords_by_prediction):
        return [dykstra(c, normals, offsets, 100) for c, normals, offsets in zip(coords_by_prediction, self.normals, self.offsets)]

    def remove_outliers(self, flows, coords_1, coords_2=None, threshold=1, return_outliers=False):
        err_1 = self.evaluate_l2_error(flows, coords_1)
        if coords_2 is not None:
            err_2 = self.evaluate_l2_error(flows, coords_2)
            keep_indices = np.minimum(err_1, err_2) < threshold
            if return_outliers:
                return flows[~keep_indices], coords_1[~keep_indices], coords_2[~keep_indices], ~keep_indices
            else:
                return flows[keep_indices], coords_1[keep_indices], coords_2[keep_indices], keep_indices
        else:
            keep_indices = err_1 < threshold
            if return_outliers:
                return flows[~keep_indices], coords_1[~keep_indices], ~keep_indices
            else:
                return flows[keep_indices], coords_1[keep_indices], keep_indices

    def evaluate_accuracy(self, flows, *coords, mean=False, ignore_invalid=False):
        predictions = self.predict(flows)
        if ignore_invalid:
            idx = predictions > -1
        else:
            idx = np.arange(len(predictions))
        label_sets = [self.categorical_labels(c) for c in coords]
        if len(label_sets) == 1:
            acc = label_sets[0][idx] == predictions[idx]
            if mean:
                return np.mean(acc)
            else:
                return acc, idx
        else:
            acc_sets = [labels[idx] == predictions[idx] for labels in label_sets]
            if mean:
                combined_acc = np.logical_or.reduce(acc_sets)
                return np.mean(combined_acc)
            else:
                return tuple(acc_sets), idx

    def evaluate_l2_error(self, flows, *coords, mean=False, false_only=False):
        def _compute_err(c):
            _, coords_by_prediction, idx_by_prediction = self.split_by_prediction(flows, c)
            projections_regions = self.projections_by_prediction(coords_by_prediction)
            projections = np.zeros(c.shape)
            for i, idx in enumerate(idx_by_prediction):
                projections[idx] = projections_regions[i]
            idx_invalid = np.where(np.all(projections == 0, axis=1))[0]
            projections[idx_invalid] = np.nan
            return np.linalg.norm(projections - c, axis=1)
        if len(coords) == 1:
            err = _compute_err(coords[0])
            if mean:
                if false_only:
                    err = err[err > self.tol]
                return np.nanmean(err)
            return err
        err1 = _compute_err(coords[0])
        err2 = _compute_err(coords[1])
        if mean:
            err_min = np.minimum(err1, err2)
            if false_only:
                err_min = err_min[err_min > self.tol]
            return np.nanmean(err_min)
        return err1, err2

    def evaluate_accuracy_two_step(self, flows_sl, coords_sl, flows_tl, coords_tl_1, coords_tl_2, mean=False, ignore_invalid=False):
        acc_sl, _ = self.evaluate_accuracy(flows_sl, coords_sl)
        acc_tl, idx_tl = self.evaluate_accuracy(flows_tl, coords_tl_1, coords_tl_2, ignore_invalid=ignore_invalid)
        acc_step_2 = []
        for i, (a_1, a_2) in enumerate(zip(*acc_tl)):
            c_1 = coords_tl_1[idx_tl][i]
            c_2 = coords_tl_2[idx_tl][i]
            i_sl_1 = np.where(np.all(coords_sl == c_1, axis=1))[0][0]
            i_sl_2 = np.where(np.all(coords_sl == c_2, axis=1))[0][0]
            if a_1 and a_2:
                if acc_sl[i_sl_1] and acc_sl[i_sl_2]:
                    acc_step_2.append(True)
                else:
                    acc_step_2.append(False)
            elif a_1:
                if acc_sl[i_sl_2]:
                    acc_step_2.append(True)
                else:
                    acc_step_2.append(False)
            elif a_2:
                if acc_sl[i_sl_1]:
                    acc_step_2.append(True)
                else:
                    acc_step_2.append(False)
            else:
                acc_step_2.append(np.nan)
        acc_step_2 = np.array(acc_step_2)
        acc_step_1 = np.logical_or.reduce([acc_tl[0], acc_tl[1]])
        if mean:
            return np.mean(acc_step_1), np.nanmean(acc_step_2)
        else:
            return acc_step_1, acc_step_2, idx_tl

    def evaluate_l2_error_two_step(self, flows_sl, coords_sl, flows_tl, coords_tl_1, coords_tl_2, mean=False, false_only=False):
        err_sl = self.evaluate_l2_error(flows_sl, coords_sl)
        err_tl_1, err_tl_2 = self.evaluate_l2_error(flows_tl, coords_tl_1, coords_tl_2)
        err_step_1 = np.minimum(err_tl_1, err_tl_2)
        err_step_2 = []
        for i, (e_1, e_2) in enumerate(zip(err_tl_1, err_tl_2)):
            c_1 = coords_tl_1[i]
            c_2 = coords_tl_2[i]
            i_sl_1 = np.where(np.all(coords_sl == c_1, axis=1))[0][0]
            i_sl_2 = np.where(np.all(coords_sl == c_2, axis=1))[0][0]
            if e_1 < self.tol and e_2 < self.tol:
                err_step_2.append(0.5 * (err_sl[i_sl_1] + err_sl[i_sl_2]))
            elif e_1 < self.tol:
                err_step_2.append(err_sl[i_sl_2])
            elif e_2 < self.tol:
                err_step_2.append(err_sl[i_sl_1])
            else:
                err_step_2.append(np.nan)
        err_step_2 = np.array(err_step_2)
        if false_only:
            err_step_1 = err_step_1[err_step_1 > self.tol]
            err_step_2 = err_step_2[err_step_2 > self.tol]
        if mean:
            return np.nanmean(err_step_1), np.nanmean(err_step_2)
        else:
            return err_step_1, err_step_2

    def evaluate_confusion_matrix(self, flows, coords, normalize=None, include_values=True, figsize=None, filename=None):
        labels = self.categorical_labels(coords)
        predictions = self.predict(flows)
        all_labels = range(-1, len(self.indices))
        cm_full = confusion_matrix(labels, predictions, labels=all_labels)
        cm = cm_full[1:, :]
        if type(self) is ClassicVoronoi:
            display_labels = list(range(1, len(self.indices) + 1))
        else:
            display_labels = [(i0 + 1, i1 + 1) for i0, i1 in self.indices]
        has_none_pred = -1 in predictions
        if has_none_pred:
            col_labels = ['None'] + display_labels
        else:
            cm = cm[:, 1:]
            col_labels = display_labels
        row_labels = display_labels
        # normalization
        if normalize == 'true':
            cm = cm / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
            fmt = '.2f'
            cbar_label = 'Proportion (row norm)'
        elif normalize == 'pred':
            cm = cm / cm.sum(axis=0)
            cm = np.nan_to_num(cm)
            fmt = '.2f'
            cbar_label = 'Proportion (col norm)'
        else:
            fmt = 'd'
            cbar_label = 'Count'
        # figure
        if figsize is None:
            figsize = (6.2, 6.2)  # full-column width for LaTeX
        fig, ax = plt.subplots(figsize=figsize)
        n_rows, n_cols = cm.shape
        # axis and tick fonts
        # used 6 and 7 for classic / 4 and 7 for refined
        tick_font = 4
        label_font = 7
        im = ax.imshow(cm, cmap='Reds', aspect='equal')
        # compute cell font to fill cell with small margin
        fig.canvas.draw()  # needed to compute accurate axes bbox
        renderer = fig.canvas.get_renderer()
        ax_bbox = ax.get_window_extent(renderer=renderer)
        # cell size in pixels
        cell_width_px = ax_bbox.width / n_cols
        cell_height_px = ax_bbox.height / n_rows
        # margin factor to leave some padding
        # used 0.65 for classic / 0.85 for refined
        margin_factor = 0.85
        cell_font_pt = min(cell_width_px, cell_height_px) * 0.5 * margin_factor * 72 / fig.dpi
        v_max = cm.max()
        # values
        if include_values:
            for i in range(n_rows):
                for j in range(n_cols):
                    v = cm[i, j]
                    if v == 0:
                        continue
                    ax.text(j, i, format(v, fmt), ha='center', va='center', fontsize=cell_font_pt, color='white' if v > 0.5 * v_max else 'black')
        # axes
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels, fontsize=tick_font)
        ax.set_yticklabels(row_labels, fontsize=tick_font)
        ax.set_xlabel('Predicted Voronoi cell', fontsize=label_font)
        ax.set_ylabel('True Voronoi cell', fontsize=label_font)
        if not type(self) is ClassicVoronoi:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        # colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=tick_font)
        cbar.ax.set_ylabel(cbar_label, fontsize=label_font, rotation=270, labelpad=12)
        # layout
        plt.tight_layout(pad=0.3)
        if filename:
            fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.show()

    def plot_diagram(self, plot_points=True, plot_edges=True, plot_contour=False, plot_fill=False, plot_circumcenters=False, plot_triangles=False, flip=False,
                     outer_edges_factor=1, linewidth=2, markersize=5, color_diagram='k', color_triangles='r', color_fill='lightgray', s_fill=1, n_fill=2000000, pad_fill=0.5):
        flip_sign = -1 if flip else 1

        if plot_points:
            plt.plot(self.points[:, 0], flip_sign * self.points[:, 1], 'ko', markersize=markersize, zorder=4)

        if plot_edges:
            for edge in self.edges_inner:
                cc0 = self.circumcenters[edge[0]]
                cc1 = self.circumcenters[edge[1]]
                plt.plot((cc0[0], cc1[0]), (flip_sign * cc0[1], flip_sign * cc1[1]), '-', linewidth=linewidth, color=color_diagram, zorder=1)

            for i, edge in enumerate(self.triangle_edges_outer):
                triangle = self.triangle_edges_map[edge][0]
                cc = self.circumcenters[triangle]
                plt.plot((cc[0], cc[0] + outer_edges_factor * self.edges_outer[i][0]), (flip_sign * cc[1], flip_sign * (cc[1] + outer_edges_factor * self.edges_outer[i][1])), linewidth=linewidth, color=color_diagram, zorder=1)

        if plot_circumcenters:
            for cc in self.circumcenters:
                plt.plot(cc[0], flip_sign * cc[1], 'o', color=color_triangles)

        if plot_triangles:
            for triangle in self.triangles:
                plt.plot((self.points[triangle[0]][0], self.points[triangle[1]][0]), (flip_sign * self.points[triangle[0]][1], flip_sign * self.points[triangle[1]][1]), '-', color=color_triangles, linewidth=3)
                plt.plot((self.points[triangle[0]][0], self.points[triangle[2]][0]), (flip_sign * self.points[triangle[0]][1], flip_sign * self.points[triangle[2]][1]), '-', color=color_triangles, linewidth=3)
                plt.plot((self.points[triangle[2]][0], self.points[triangle[1]][0]), (flip_sign * self.points[triangle[2]][1], flip_sign * self.points[triangle[1]][1]), '-', color=color_triangles, linewidth=3)

        if plot_contour:
            assert(self.points_contour is not None)
            for i, p1 in enumerate(self.points_contour[:-1]):
                p2 = self.points_contour[i + 1]
                plt.plot([p1[0], p2[0]], [flip_sign * p1[1], flip_sign * p2[1]], linewidth=linewidth, color='k', zorder=3)
            p1 = self.points_contour[-1]
            p2 = self.points_contour[0]
            plt.plot([p1[0], p2[0]], [flip_sign * p1[1], flip_sign * p2[1]], linewidth=linewidth, color='k', zorder=3)

        if plot_fill:
            if self.normals_contour is None:
                self.compute_contour_hyperplanes()
            x_lim = (self.points_contour[:, 0].min() - pad_fill, self.points_contour[:, 0].max() + pad_fill)
            y_lim = (self.points_contour[:, 1].min() - pad_fill, self.points_contour[:, 1].max() + pad_fill)
            coords_plot_x = np.random.uniform(low=x_lim[0], high=x_lim[1], size=n_fill)
            coords_plot_y = np.random.uniform(low=y_lim[0], high=y_lim[1], size=n_fill)
            coords = np.column_stack((coords_plot_x, coords_plot_y))
            labels = np.where(self.normals_contour @ coords.T + self.offsets_contour > 0)[1]
            coords_plot_x = coords_plot_x[labels]
            coords_plot_y = coords_plot_y[labels]
            plt.scatter(coords_plot_x, flip_sign * coords_plot_y, s=s_fill, linewidth=0, color=color_fill, zorder=2)

        if self.points_contour is not None:
            pad_x = (self.points_contour[:, 0].max() - self.points_contour[:, 0].min()) / 50
            pad_y = (self.points_contour[:, 1].max() - self.points_contour[:, 1].min()) / 50
            pad = np.maximum(pad_x, pad_y)
            plt.xlim([self.points_contour[:, 0].min() - pad, self.points_contour[:, 0].max() + pad])
            if not flip:
                plt.ylim([self.points_contour[:, 1].min() - pad, self.points_contour[:, 1].max() + pad])
            else:
                plt.ylim([-self.points_contour[:, 1].max() - pad, -self.points_contour[:, 1].min() + pad])

    def plot_data(self, flows, coords, palette=None, flip=False, shuffle_colors=False, evaluate=True, color='k', s=40):
        if palette is None:
            palette = standard_palette
        palette = palette[:len(self.normals)]
        if shuffle_colors is not None:
            random.seed(shuffle_colors)
            random.shuffle(palette)
        flip_sign = -1 if flip else 1
        if evaluate:
            labels = self.categorical_labels(coords)
            predictions = self.predict(flows)
            correctness = (labels == predictions)
            sns.scatterplot(x=coords[:, 0], y=flip_sign * coords[:, 1], hue=predictions, hue_order=range(len(self.normals)), style=correctness, size=correctness,
                            style_order=[True, False], size_order=[True, False], sizes=[s, s / 4 * 10], palette=palette, legend=False, zorder=4)
        else:
            sns.scatterplot(x=coords[:, 0], y=flip_sign * coords[:, 1], color=color, s=s, legend=False, zorder=4)

    def plot_cells(self, n_points=2000000, palette=None, s=5, flip=False, shuffle_colors=False):
        if palette is None:
            palette = standard_palette
        palette = palette[:len(self.normals)]
        if shuffle_colors is not None:
            random.seed(shuffle_colors)
            random.shuffle(palette)
        flip_sign = -1 if flip else 1
        x_lim = (self.points_contour[:, 0].min(), self.points_contour[:, 0].max())
        y_lim = (self.points_contour[:, 1].min(), self.points_contour[:, 1].max())
        coords_plot_x = np.random.uniform(low=x_lim[0], high=x_lim[1], size=n_points)
        coords_plot_y = np.random.uniform(low=y_lim[0], high=y_lim[1], size=n_points)
        coords_labels = self.categorical_labels(np.column_stack((coords_plot_x, coords_plot_y)))
        sns.scatterplot(x=coords_plot_x, y=flip_sign * coords_plot_y, hue=coords_labels, hue_order=range(len(self.normals)), palette=palette, s=s, linewidth=0, legend=False, zorder=0)

    def plot_cell_labels(self, n_points=2000000, fontsize=20, flip=False, labels=None, label_offsets=None):
        if self.normals_contour is None:
            self.compute_contour_hyperplanes()
        if labels is not None:
            assert(len(labels) == len(self.indices))
        if label_offsets is not None:
            assert (len(label_offsets) == len(self.indices))
        else:
            label_offsets = [[0, 0]] * len(self.indices)
        flip_sign = -1 if flip else 1
        x_lim = (self.points_contour[:, 0].min(), self.points_contour[:, 0].max())
        y_lim = (self.points_contour[:, 1].min(), self.points_contour[:, 1].max())
        coords_plot_x = np.random.uniform(low=x_lim[0], high=x_lim[1], size=n_points)
        coords_plot_y = np.random.uniform(low=y_lim[0], high=y_lim[1], size=n_points)
        coords = np.column_stack((coords_plot_x, coords_plot_y))
        for cell_index, cell_label in enumerate(self.indices):
            cell_hp_eval = np.row_stack([self.normals[cell_index], self.normals_contour]) @ coords.T + np.row_stack([self.offsets[cell_index], self.offsets_contour])
            cell_indices = np.where(np.prod(cell_hp_eval <= 0, axis=0))[0]
            if len(cell_indices) > n_points / 200:
                cell_hp_eval = cell_hp_eval[:, cell_indices]
                cell_coords = coords[cell_indices]
                cell_hp_eval = cell_hp_eval / np.mean(np.abs(cell_hp_eval), axis=1, keepdims=True)
                cell_hp_eval_std = np.std(cell_hp_eval, axis=0)
                min_std_index = np.argmin(cell_hp_eval_std)
                min_std_coords = cell_coords[min_std_index]
                if labels is not None:
                    plt.text(min_std_coords[0] + label_offsets[cell_index][0], flip_sign * (min_std_coords[1]  + label_offsets[cell_index][1]), labels[cell_index],fontsize=fontsize, ha='center', va='center', zorder=5)
                elif type(self) is ClassicVoronoi:
                    plt.text(min_std_coords[0] + label_offsets[cell_index][0], flip_sign * (min_std_coords[1]  + label_offsets[cell_index][1]),fr'$V_{{{cell_label[0] + 1}}}$', fontsize=fontsize, ha='center', va='center', zorder=5)
                else:
                    plt.text(min_std_coords[0] + label_offsets[cell_index][0], flip_sign * (min_std_coords[1]  + label_offsets[cell_index][1]), fr'$V_{{({cell_label[0] + 1},{cell_label[1] + 1})}}$', fontsize=fontsize, ha='center', va='center', zorder=5)

    def plot_point_labels(self, flip=False, fontsize=8):
        flip_sign = -1 if flip else 1
        for i, point in enumerate(self.points):
            # the 5 is adapted to the wing case (first five labels top of point locations, others below --> adapt as required)
            if i < 5:
                plt.text(point[0], flip_sign * point[1] + 500, fr'$\mathbf{{p}}_{{{i + 1}}}$', fontsize=fontsize, ha='center', va='top')
            else:
                plt.text(point[0], flip_sign * point[1] - 500, fr'$\mathbf{{p}}_{{{i + 1}}}$', fontsize=fontsize, ha='center', va='bottom')

    def plot_projections_by_prediction(self, flows, coords, cell_indices=None, palette=None, palette_data=None, flip=False, shuffle_colors=False, shuffle_colors_data=False, linewidth=2, linestyle='-', halo=False, s=40):
        if cell_indices is None:
            cell_indices = np.arange(len(self.indices))
        if palette is None:
            palette = standard_palette
        palette = palette[:len(self.normals)]
        if shuffle_colors is not None:
            random.seed(shuffle_colors)
            random.shuffle(palette)
        flip_sign = -1 if flip else 1
        flows_by_prediction, coords_by_prediction, idx_by_prediction = self.split_by_prediction(flows, coords)
        projections_regions = self.projections_by_prediction(coords_by_prediction)
        for cell_idx in cell_indices:
            self.plot_data(flows_by_prediction[cell_idx], coords_by_prediction[cell_idx], palette=palette_data, flip=flip, shuffle_colors=shuffle_colors_data, s=s)
            if halo:
                plt.plot((coords_by_prediction[cell_idx][:, 0], projections_regions[cell_idx][:, 0]), (flip_sign * coords_by_prediction[cell_idx][:, 1], flip_sign * projections_regions[cell_idx][:, 1]), color='w', linewidth=linewidth + 1)
            plt.plot((coords_by_prediction[cell_idx][:, 0], projections_regions[cell_idx][:, 0]), (flip_sign * coords_by_prediction[cell_idx][:, 1], flip_sign * projections_regions[cell_idx][:, 1]), color=palette[cell_idx], linewidth=linewidth, linestyle=linestyle)

class RefinedVoronoi(ClassicVoronoi):

    def __init__(self, points, points_contour=None, points_fill=None, tol=1e-10):
        super().__init__(points=points, points_contour=points_contour, points_fill=points_fill, tol=tol)
        self.original_normals = self.normals
        self.original_offsets = self.offsets
        self.sub_diagrams = []
        self.normals = []
        self.offsets = []
        self.indices = []
        self.order = 2
        self.compute()

    def compute(self):
        triangles = np.array(self.triangles)
        for i in range(self.points.shape[0]):
            excluding_i_indices = [j for j in range(self.points.shape[0]) if j != i]
            excluding_i_diagram = ClassicVoronoi(self.points[excluding_i_indices, :])
            normals = []
            offsets = []
            indices = []
            neighbors = set(triangles[np.any(triangles == i, axis=1)].ravel()) - {i}
            for j in neighbors:
                normals.append(np.concatenate([self.original_normals[i], excluding_i_diagram.normals[excluding_i_indices.index(j)]], axis=0))
                offsets.append(np.concatenate([self.original_offsets[i], excluding_i_diagram.offsets[excluding_i_indices.index(j)]], axis=0))
                indices.append((i, j))
            self.normals += normals
            self.offsets += offsets
            self.indices += indices
            self.sub_diagrams.append(excluding_i_diagram)

    def plot_diagram(self, plot_points=True, plot_edges=True, plot_contour=False, plot_fill=False, plot_circumcenters=False, plot_triangles=False, flip=False,
                     outer_edges_factor=1, linewidth=2, markersize=5, color_diagram='k', color_triangles='r', color_fill='lightgray', s_fill=1, n_fill=2000000, pad_fill=0.5):

        for voronoi_diagram in self.sub_diagrams:
            voronoi_diagram.plot_diagram(plot_points=False, plot_edges=plot_edges, outer_edges_factor=outer_edges_factor, color_diagram=color_diagram, flip=flip, linewidth=linewidth * 0.5)

        args = dict(locals())
        for key in ['self', '__class__', 'voronoi_diagram']: args.pop(key, None)
        super().plot_diagram(**args)


def project_on_halfspace(coords, normal, offset):
    w = np.reshape(normal, (1, 2))
    b = np.reshape(offset, (1, 1))
    return coords - w * np.maximum(coords @ w.T + b, 0.)


def dykstra(coords, normals, offsets, n_iter=10):
    n = len(normals)
    z = coords
    e = [np.zeros(coords.shape) for _ in range(n)]
    for m in range(n_iter):
        i = np.mod(m, n)
        y = project_on_halfspace(z + e[i], normals[i], offsets[i])
        e[i] = (z + e[i]) - y
        z = y
    return z
