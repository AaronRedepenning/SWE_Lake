from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch

from model import Array, Grid, StateHistory


##################################################
# Datastructures                                 #
##################################################
@dataclass
class VorticityEKE:
    mean_vorticity: Array
    mean_eke: Array


##################################################
# Helpers                                        #
##################################################
def velocity_from_transport(
    history: StateHistory,
    grid: Grid,
) -> tuple[Array, Array]:
    """
    Convert depth-integrated transports U, V into velocities u, v.
    """
    u = np.full_like(history.U, np.nan, dtype=np.float64)
    v = np.full_like(history.V, np.nan, dtype=np.float64)

    np.divide(history.U, grid.H[None, :, :], out=u, where=grid.water_mask)
    np.divide(history.V, grid.H[None, :, :], out=v, where=grid.water_mask)

    return u, v


def mask_land_2d(a: Array, grid: Grid) -> Array:
    out = np.array(a, dtype=np.float64, copy=True)
    out[grid.land_mask] = np.nan
    return out


def grid_xy(grid: Grid) -> tuple[Array, Array]:
    """
    Cell-center coordinates, assuming regular spacing.
    """
    ny, nx = grid.shape
    x = np.arange(nx, dtype=np.float64) * grid.dx
    y = np.arange(ny, dtype=np.float64) * grid.dy
    return x, y


def image_extent(grid: Any) -> tuple[float, float, float, float]:
    """
    Extent for imshow using cell-center coordinates.
    """
    ny, nx = grid.shape
    return (
        -0.5 * grid.dx,
        (nx - 0.5) * grid.dx,
        -0.5 * grid.dy,
        (ny - 0.5) * grid.dy,
    )


##################################################
# Plotting                                       #
##################################################
def plot_zeta_timeseries(
    history: StateHistory,
    row: int,
    col: int,
    *,
    label: str | None = None,
    ax: Axes | None = None,
):
    """
    Plot zeta(t) at a a grid index (row, col).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    else:
        fig = ax.figure

    series_label = label or f"row={row}, col={col}"

    ax.plot(history.t, history.z[:, row, col], label=series_label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Sea level ζ [m]")
    ax.set_title("Sea-level time series")
    ax.grid(True)

    if label is not None:
        ax.legend()

    return fig, ax


def plot_summary_maps(
    histories: Mapping[str, StateHistory],
    grid: Grid,
    *,
    symmetric_mean_scale: bool = True,
    cmap_mean: str = "RdBu_r",
    cmap_std: str = "viridis",
):
    """
    Plot side-by-side summary maps for two model runs.
    """
    if len(histories) != 2:
        raise ValueError(f"Expected exactly two histories, got {len(histories)}.")

    run_names = list(histories.keys())
    n_runs = len(run_names)

    water = grid.water_mask

    # Compute velocity fields for each run
    run_fields: dict[str, dict[str, Array]] = {}

    for run_name, history in histories.items():
        u, v = velocity_from_transport(history, grid)

        run_fields[run_name] = {
            "ζ": history.z,
            "u": u,
            "v": v,
        }

    field_names = ["ζ", "u", "v"]

    # Precompute mean/std maps
    stats: dict[str, dict[str, dict[str, Array]]] = {}

    for run_name in run_names:
        stats[run_name] = {}

        for field_name in field_names:
            values = run_fields[run_name][field_name]

            # Mask land through time before taking statistics
            masked_values = np.where(water[None, :, :], values, np.nan)

            mean_map = np.nanmean(masked_values, axis=0)
            std_map = np.nanstd(masked_values, axis=0)

            mean_map = mask_land_2d(mean_map, grid)
            std_map = mask_land_2d(std_map, grid)

            stats[run_name][field_name] = {
                "mean": mean_map,
                "std": std_map,
            }

    fig, axes = plt.subplots(
        2,
        3 * n_runs,
        figsize=(18, 8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    extent = image_extent(grid)

    for field_idx, field_name in enumerate(field_names):
        # Shared color scale for means of this field
        mean_values = np.concatenate(
            [stats[run_name][field_name]["mean"].ravel() for run_name in run_names]
        )
        mean_values = mean_values[np.isfinite(mean_values)]

        if mean_values.size == 0:
            mean_vmin = mean_vmax = None
        elif symmetric_mean_scale:
            mean_abs = np.nanmax(np.abs(mean_values))
            mean_vmin, mean_vmax = -mean_abs, mean_abs
        else:
            mean_vmin, mean_vmax = np.nanmin(mean_values), np.nanmax(mean_values)

        # Shared color scale for std maps of this field
        std_values = np.concatenate(
            [stats[run_name][field_name]["std"].ravel() for run_name in run_names]
        )
        std_values = std_values[np.isfinite(std_values)]

        if std_values.size == 0:
            std_vmin = std_vmax = None
        else:
            std_vmin, std_vmax = 0.0, np.nanmax(std_values)

        mean_images = []
        std_images = []
        mean_axes = []
        std_axes = []

        for run_idx, run_name in enumerate(run_names):
            col_idx = field_idx * n_runs + run_idx

            mean_map = stats[run_name][field_name]["mean"]
            std_map = stats[run_name][field_name]["std"]

            # Mean map
            ax = axes[0, col_idx]
            im = ax.imshow(
                mean_map,
                # extent=extent,
                aspect="equal",
                cmap=cmap_mean,
                origin="lower",
                vmin=mean_vmin,
                vmax=mean_vmax,
            )
            ax.set_title(f"{run_name}\nMean {field_name}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            mean_images.append(im)
            mean_axes.append(ax)

            # Std map
            ax = axes[1, col_idx]
            im = ax.imshow(
                std_map,
                # extent=extent,
                aspect="equal",
                cmap=cmap_std,
                origin="lower",
                vmin=std_vmin,
                vmax=std_vmax,
            )
            ax.set_title(f"{run_name}\nStd. dev. {field_name}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            std_images.append(im)
            std_axes.append(ax)

        # One shared colorbar for the two mean panels of this field
        fig.colorbar(
            mean_images[-1],
            ax=mean_axes,
            shrink=0.8,
            label=f"Mean {field_name}",
        )

        # One shared colorbar for the two std panels of this field
        fig.colorbar(
            std_images[-1],
            ax=std_axes,
            shrink=0.8,
            label=f"Std. dev. {field_name}",
        )

    fig.suptitle("Summary map comparison")

    return fig, axes


def plot_zeta_hovmoller(
    histories: Mapping[str, Any],
    grid: Any,
    *,
    y_index: int = 25,
    symmetric_scale: bool = True,
):
    """
    Plot Hovmöller diagrams of zeta along a fixed y-index.
    """
    x, _ = grid_xy(grid)

    panel_data: dict[str, Array] = {}
    for name, hist in histories.items():
        panel_data[name] = hist.z[:, y_index, :]

    n = len(panel_data)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 8),
        constrained_layout=True,
        squeeze=False,
    )

    all_values = np.concatenate([d.ravel() for d in panel_data.values()])
    all_values = all_values[np.isfinite(all_values)]

    if all_values.size == 0:
        vmin = vmax = None
    elif symmetric_scale:
        vmax_abs = np.nanmax(np.abs(all_values))
        vmin, vmax = -vmax_abs, vmax_abs
    else:
        vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)

    im = None

    for ax, (name, data) in zip(axes.ravel(), panel_data.items()):
        hist = histories[name]
        t = hist.t[: data.shape[0]]

        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=(x[0], x[-1], t[0], t[-1]),
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
        )

        ax.set_title(name)
        ax.set_xlabel("x along transect")
        ax.set_ylabel("Time")

    for ax in axes.ravel()[len(panel_data) :]:
        ax.axis("off")

    if im is not None:
        active_axes = axes.ravel()[: len(panel_data)].tolist()
        fig.colorbar(im, ax=active_axes, shrink=0.85, label="ζ")

    fig.suptitle(f"Hovmöller plot of ζ along y-index {y_index}")

    return fig, axes


def show_streamplot_animation(
    histories: Mapping[str, Any],
    grid_or_H: Any,
    *,
    frame_stride: int = 10,
    interval: int = 250,
    density: float = 0.9,
    cmap: str = "Spectral_r",
    speed_percentile: float = 95.0,
):
    """
    Animate streamplots from two histories side by side.
    """
    if len(histories) != 2:
        raise ValueError(f"Expected exactly two histories, got {len(histories)}.")

    # Accept either Grid or H directly
    H = grid_or_H.H if hasattr(grid_or_H, "H") else np.asarray(grid_or_H)

    water = H > 0
    ny, nx = H.shape

    x = np.arange(nx)
    y = np.arange(ny)

    # Use only frames that exist in both histories
    n_frames_available = min(hist.U.shape[0] for hist in histories.values())
    frames = list(range(0, n_frames_available, frame_stride))

    if not frames:
        raise ValueError("No frames selected. Check frame_stride and history length.")

    # Shared speed color scale across both histories
    speed_samples = []
    for hist in histories.values():
        for n in frames:
            speed = np.sqrt(hist.U[n] ** 2 + hist.V[n] ** 2)
            speed_samples.append(speed[water])

    all_speeds = np.concatenate(speed_samples)
    all_speeds = all_speeds[np.isfinite(all_speeds)]

    if all_speeds.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.percentile(all_speeds, speed_percentile))
        if vmax <= 0.0:
            vmax = float(np.nanmax(all_speeds)) if np.nanmax(all_speeds) > 0 else 1.0

    color_scale = Normalize(vmin=0.0, vmax=vmax)

    lake_color = ListedColormap(["#d9edf7"])
    lake_color.set_bad(alpha=0)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 9),
        facecolor="white",
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()

    # Shared colorbar
    colorbar_object = ScalarMappable(norm=color_scale, cmap=cmap)
    colorbar_object.set_array([])
    fig.colorbar(
        colorbar_object,
        ax=axes.tolist(),
        shrink=0.85,
        label="Flow speed",
    )

    lake_area = np.where(water, 1.0, np.nan)
    lake_legend = Patch(
        facecolor="#d9edf7",
        edgecolor="gray",
        label="Lake area",
    )

    def update(frame_index: int):
        artists = []

        for ax, (name, hist) in zip(axes, histories.items()):
            ax.clear()
            ax.set_facecolor("white")

            ax.imshow(
                lake_area,
                origin="lower",
                cmap=lake_color,
            )

            U = np.ma.array(hist.U[frame_index], mask=~water)
            V = np.ma.array(hist.V[frame_index], mask=~water)
            speed = np.ma.sqrt(U**2 + V**2)

            stream = ax.streamplot(
                x,
                y,
                U,
                V,
                color=speed,
                cmap=cmap,
                norm=color_scale,
                density=density,
            )

            ax.legend(handles=[lake_legend], loc="upper right")

            if hasattr(hist, "t"):
                title_time = f", t = {hist.t[frame_index]:.1f} s"
            else:
                title_time = ""

            ax.set_title(f"{name}, frame {frame_index}{title_time}")
            ax.set_xlabel("X index")
            ax.set_ylabel("Y index")
            ax.set_xlim(0, nx - 1)
            ax.set_ylim(0, ny - 1)

            artists.extend(stream.lines.get_segments())

        fig.suptitle("Streamplot comparison")

        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=interval,
        blit=False,
    )

    return fig, axes, animation


def plot_vorticity_eke_comparison(
    diagnostics_by_scenario: Mapping[str, Mapping[str, VorticityEKE]],
    *,
    scenario_order: Sequence[str] | None = None,
    method_order: Sequence[str] | None = None,
    vorticity_percentile: float | None = 99.0,
    eke_percentile: float | None = 99.0,
    cmap_vorticity: str = "RdBu_r",
    cmap_eke: str = "viridis",
):
    """
    Create one combined figure comparing mean vorticity and mean EKE
    across scenarios and methods.
    """
    if scenario_order is None:
        scenario_order = list(diagnostics_by_scenario.keys())

    if method_order is None:
        first_scenario = scenario_order[0]
        method_order = list(diagnostics_by_scenario[first_scenario].keys())

    n_scenarios = len(scenario_order)
    n_methods = len(method_order)
    ncols = n_scenarios * n_methods

    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(3.0 * ncols, 7.0),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    # --------------------------------------------------
    # Shared color limits
    # --------------------------------------------------
    all_vorticity = []
    all_eke = []

    for scenario_name in scenario_order:
        for method_name in method_order:
            diag = diagnostics_by_scenario[scenario_name][method_name]
            all_vorticity.append(diag.mean_vorticity.ravel())
            all_eke.append(diag.mean_eke.ravel())

    all_vorticity = np.concatenate(all_vorticity)
    all_vorticity = all_vorticity[np.isfinite(all_vorticity)]

    all_eke = np.concatenate(all_eke)
    all_eke = all_eke[np.isfinite(all_eke)]

    if all_vorticity.size == 0:
        vort_vmin = vort_vmax = None
    else:
        if vorticity_percentile is None:
            vort_limit = np.nanmax(np.abs(all_vorticity))
        else:
            vort_limit = np.nanpercentile(
                np.abs(all_vorticity),
                vorticity_percentile,
            )

        if vort_limit <= 0.0:
            vort_limit = 1.0

        vort_vmin = -vort_limit
        vort_vmax = vort_limit

    if all_eke.size == 0:
        eke_vmin = eke_vmax = None
    else:
        eke_vmin = 0.0

        if eke_percentile is None:
            eke_vmax = np.nanmax(all_eke)
        else:
            eke_vmax = np.nanpercentile(all_eke, eke_percentile)

        if eke_vmax <= 0.0:
            eke_vmax = 1.0

    # --------------------------------------------------
    # Plot maps
    # --------------------------------------------------
    vorticity_image = None
    eke_image = None

    for scenario_idx, scenario_name in enumerate(scenario_order):
        for method_idx, method_name in enumerate(method_order):
            col_idx = scenario_idx * n_methods + method_idx
            diag = diagnostics_by_scenario[scenario_name][method_name]

            ax = axes[0, col_idx]
            vorticity_image = ax.imshow(
                diag.mean_vorticity,
                origin="lower",
                cmap=cmap_vorticity,
                vmin=vort_vmin,
                vmax=vort_vmax,
                aspect="equal",
            )

            ax.set_title(f"{scenario_name}\n{method_name}", fontsize=10)
            ax.set_xlabel("x")

            if col_idx == 0:
                ax.set_ylabel("Mean vorticity\n1/s")
            else:
                ax.set_ylabel("")

            ax = axes[1, col_idx]
            eke_image = ax.imshow(
                diag.mean_eke,
                origin="lower",
                cmap=cmap_eke,
                vmin=eke_vmin,
                vmax=eke_vmax,
                aspect="equal",
            )

            ax.set_title(f"{scenario_name}\n{method_name}", fontsize=10)
            ax.set_xlabel("x")

            if col_idx == 0:
                ax.set_ylabel("Mean EKE\nm²/s²")
            else:
                ax.set_ylabel("")

    # --------------------------------------------------
    # Shared row colorbars
    # --------------------------------------------------
    if vorticity_image is not None:
        fig.colorbar(
            vorticity_image,
            ax=axes[0, :].tolist(),
            shrink=0.8,
            label="Mean vorticity 1/s",
        )

    if eke_image is not None:
        fig.colorbar(
            eke_image,
            ax=axes[1, :].tolist(),
            shrink=0.8,
            label="Mean EKE m²/s²",
        )

    fig.suptitle("Mean vorticity and EKE comparison")

    return fig, axes
