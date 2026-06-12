"""
random_example.py

Compute and visualize classic and refined Voronoi diagrams using random locations.

This script reproduces figures from the paper that illustrate classic and refined
Voronoi diagrams and their construction by means of a random example.
Other examples can be obtained by changing the random seed in the beginning.
Note that label positions were manually tuned for the example given in the paper.


Author: Christoph Brauer
Date: 2026-04-16
"""

import leak_localization as ll
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    # create random example setup
    np.random.seed(11)
    points_vacuum = np.random.uniform(low=-1, high=1, size=(6, 2))
    points_contour = 1.1 * np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    classic_voronoi = ll.ClassicVoronoi(points_vacuum, points_contour)
    refined_voronoi = ll.RefinedVoronoi(points_vacuum, points_contour)

    # plot generic diagram (Figure 1 in paper)
    plt.figure(figsize=(7, 7))
    classic_voronoi.plot_diagram(outer_edges_factor=2, plot_contour=True, plot_fill=True, color_fill='w')
    classic_voronoi.plot_cell_labels(n_points=100000, fontsize=14)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.box(False)
    plt.savefig(f'figures/classic_voronoi_example.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    # plot generic diagram with labels in brackets (Figure 16 (left) in paper)
    plt.figure(figsize=(7, 7))
    classic_voronoi.plot_diagram(outer_edges_factor=2, plot_contour=True, plot_fill=True, color_fill='w')
    labels = [fr'$V_{{({i})}}$' for i in range(1, len(classic_voronoi.indices) + 1)]
    classic_voronoi.plot_cell_labels(n_points=100000, fontsize=14, labels=labels)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.box(False)
    plt.savefig(f'figures/classic_voronoi_example_relabeled.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    # plot generic diagram with triangles (Figure 15 in paper)
    plt.figure(figsize=(7, 7))
    classic_voronoi.plot_diagram(outer_edges_factor=2, plot_contour=True, plot_fill=True, color_fill='w', plot_triangles=True, plot_circumcenters=True, color_triangles='tab:orange')
    label_offsets = [[0, 0]] * 6
    label_offsets[0] = [-0.01, 0.1]
    label_offsets[2] = [-0.04, -0.1]
    label_offsets[4] = [-0.04, -0.1]
    label_offsets[5] = [0.04, 0]
    classic_voronoi.plot_cell_labels(n_points=100000, fontsize=14, label_offsets=label_offsets)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.box(False)
    plt.savefig(f'figures/classic_voronoi_example_with_triangles.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    # plot generalized diagram (Figure 16 (right) in paper)
    plt.figure(figsize=(7, 7))
    refined_voronoi.plot_diagram(outer_edges_factor=2, plot_contour=True, plot_fill=True, color_fill='w')
    classic_voronoi.plot_diagram(outer_edges_factor=2, color_diagram='tab:orange', linewidth=3)
    refined_voronoi.plot_cell_labels(n_points=100000, fontsize=14)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.box(False)
    plt.savefig(f'figures/refined_voronoi_example.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    # classic diagram of all points (smaller figsize than previously) (Figure 17 (left) in paper)
    plt.figure(figsize=(2.25, 2.25))
    classic_voronoi.plot_diagram(outer_edges_factor=2, linewidth=0.5, markersize=0.5, plot_contour=True, plot_fill=True, color_fill='w')
    labels = [fr'$V_{{({i})}}$' for i in range(1, len(classic_voronoi.indices) + 1)]
    label_offsets = [[0, 0]] * len(labels)
    label_offsets[2] = [0, -0.1]
    classic_voronoi.plot_cell_labels(n_points=10000, fontsize=7, labels=labels, label_offsets=label_offsets)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.box(False)
    plt.savefig(f'figures/refined_voronoi_construction_example_0.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    # remove third point and plot classic diagram on reduced point set (Figure 17 (middle) in paper)
    classic_voronoi_reduced = ll.ClassicVoronoi(points_vacuum[[0, 1, 3, 4, 5]], points_contour)
    plt.figure(figsize=(2.25, 2.25))
    classic_voronoi_reduced.plot_diagram(outer_edges_factor=2, linewidth=0.5, markersize=0.5, plot_contour=True, plot_fill=True, color_fill='w')
    labels = [fr'$R_{{({i})}}$' for i in [1, 2, 4, 5, 6]]
    label_offsets = [[0, 0]] * len(labels)
    label_offsets[1] = [0, 0.1]
    classic_voronoi_reduced.plot_cell_labels(n_points=10000, fontsize=7, labels=labels, label_offsets=label_offsets)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.box(False)
    plt.savefig(f'figures/refined_voronoi_construction_example_1.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    # plot original and reduced diagram on top of each other to get refined third cell (Figure 17 (right) in paper)
    plt.figure(figsize=(2.25, 2.25))
    classic_voronoi_reduced.plot_diagram(outer_edges_factor=2, linewidth=0.5, markersize=0.5, plot_contour=True, plot_fill=True, color_fill='w')
    classic_voronoi.plot_diagram(outer_edges_factor=2, linewidth=1, color_diagram='tab:orange', plot_points=False)
    labels = [fr'$R_{{({i})}}$' for i in [1, 2, 4, 5, 6]]
    label_offsets = [[0, 0]] * len(labels)
    label_offsets[1] = [0, 0.1]
    classic_voronoi_reduced.plot_cell_labels(n_points=10000, fontsize=7, labels=labels, label_offsets=label_offsets)
    labels_gen = [fr'$V_{{({cell_label[0] + 1},{cell_label[1] + 1})}}$' for cell_label in refined_voronoi.indices]
    for i in [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19]: labels_gen[i] = ''
    label_offsets = [[0, 0]] * len(labels_gen)
    label_offsets[8] = [0.009, 0]
    label_offsets[10] = [0, 0.01]
    refined_voronoi.plot_cell_labels(n_points=10000, fontsize=5, labels=labels_gen, label_offsets=label_offsets)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.box(False)
    plt.savefig(f'figures/refined_voronoi_construction_example_2.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
