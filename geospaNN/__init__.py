"""
geospaNN: A Python package for geospatial deep learning

recommend keeping namespaces separate...
"""

from . import r_utils
from .utils import (
    DropoutLayer,
    EarlyStopping,
    LRScheduler,
    SparseB,
    NNGPCov,
    confidence_interval,
    coord_basis,
    distance,
    distance_np,
    krig_pred,
    make_bf,
    make_bf_from_cov,
    make_cov_full,
    make_cov,
    make_graph,
    make_rank,
    rmvn,
    simulation,
    spatial_order,
    split_data,
    split_loader,
    edit_batch,
    theta_update,
)
from .model import NNGLS, linear_gls
from .main import NNTrain, NNGLSTrain
from .visualize import spatial_plot_surface, plot_pdp, plot_pdp_list, plot_log

__all__ = [
    "NNTrain",
    "NNGLSTrain",
    "NNGLS",
    "r_utils",
    "spatial_plot_surface",
    "plot_pdp",
    "plot_pdp_list",
    "plot_log",
    "DropoutLayer",
    "EarlyStopping",
    "LRScheduler",
    "SparseB",
    "NNGPCov",
    "NNTrain",
    "NNGLSTrain",
    "NNGLS",
    "linear_gls",
    "confidence_interval",
    "coord_basis",
    "distance",
    "distance_np",
    "krig_pred",
    "make_bf",
    "make_bf_from_cov",
    "make_cov_full",
    "make_cov",
    "make_graph",
    "make_rank",
    "rmvn",
    "simulation",
    "spatial_order",
    "split_data",
    "split_loader",
    "edit_batch",
    "theta_update",
]
