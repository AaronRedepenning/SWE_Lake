from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from model import Array, Grid, StateHistory


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
    history: StateHistory,
    grid: Grid,
):
    """
    Plot mean and standard deviation maps of:
        zeta, u velocity, v velocity.
    """
    # Convert transport U and V to velocities u and v
    u, v = velocity_from_transport(history, grid)

    fields = {
        "ζ": history.z,
        "u": u,
        "v": v,
    }

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    extent = image_extent(grid)

    for col_idx, (name, values) in enumerate(fields.items()):
        mean_map = np.nanmean(values, axis=0, where=grid.water_mask)
        std_map = np.nanstd(values, axis=0, where=grid.water_mask)

        mean_map = mask_land_2d(mean_map, grid)
        std_map = mask_land_2d(std_map, grid)

        ax = axes[0, col_idx]
        im = ax.imshow(
            mean_map,
            extent=extent,
            aspect="equal",
            cmap="RdBu_r",
        )
        ax.set_title(f"Mean {name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.8)

        ax = axes[1, col_idx]
        im = ax.imshow(
            std_map,
            extent=extent,
            aspect="equal",
            cmap="viridis",
        )
        ax.set_title(f"Std. dev. {name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.8)

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

    for ax, (name, data) in zip(axes.ravel(), panel_data.items()):
        hist = histories[name] if name in histories else next(iter(histories.values()))
        t = hist.t[: data.shape[0]]

        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=(x[0], x[-1], t[0], t[-1]),
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(name)
        ax.set_xlabel("x along transect")
        ax.set_ylabel("Time")
        fig.colorbar(im, ax=ax, shrink=0.85, label="ζ")

    for ax in axes.ravel()[len(panel_data) :]:
        ax.axis("off")

    fig.suptitle(f"Hovmöller plot of ζ along y-index {y_index}")

    return fig, axes


def plot_scenario_stream_snapshots(
    history,
    grid,
    *,
    scenario_name: str = "Scenario",
    nrows: int = 2,
    ncols: int = 4,
    use_total_depth: bool = False,
    min_depth: float = 1e-6,
    density: float = 1.4,
    cmap: str = "Spectral_r",
    speed_percentile: float = 99.0,
    linewidth_min: float = 0.4,
    linewidth_max: float = 2.0,
    show_land_boundary: bool = True,
    flip_y: bool = False,
):
    """
    Plot one scenario as a grid of streamplots sampled evenly over time.
    """

    n_frames = nrows * ncols

    if len(history.t) < n_frames:
        raise ValueError(
            f"Requested {n_frames} frames, but history only has {len(history.t)}."
        )

    indices = np.linspace(
        0,
        len(history.t) - 1,
        n_frames,
        dtype=int,
    )

    ny, nx = grid.shape

    x = np.arange(nx, dtype=np.float64) * grid.dx
    y = np.arange(ny, dtype=np.float64) * grid.dy

    water = grid.water_mask.copy()

    if flip_y:
        water_plot = np.flipud(water)
    else:
        water_plot = water

    def velocity_at(k: int):
        if use_total_depth:
            depth = grid.H + history.z[k]
        else:
            depth = grid.H

        valid = grid.water_mask & (depth > min_depth)

        u = np.full(grid.shape, np.nan, dtype=np.float64)
        v = np.full(grid.shape, np.nan, dtype=np.float64)

        np.divide(history.U[k], depth, out=u, where=valid)
        np.divide(history.V[k], depth, out=v, where=valid)

        speed = np.sqrt(u**2 + v**2)

        u[~valid] = np.nan
        v[~valid] = np.nan
        speed[~valid] = np.nan

        if flip_y:
            u = np.flipud(u)
            v = np.flipud(v)
            speed = np.flipud(speed)

        return u, v, speed

    # Shared speed color scale across all panels in this scenario
    selected_speeds = []

    for k in indices:
        _, _, speed = velocity_at(k)
        selected_speeds.append(speed[np.isfinite(speed)])

    selected_speeds = np.concatenate(selected_speeds)

    if selected_speeds.size == 0:
        vmax = 1.0
    else:
        vmax = np.nanpercentile(selected_speeds, speed_percentile)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = np.nanmax(selected_speeds)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

    norm = Normalize(vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for ax, k in zip(axes.ravel(), indices):
        u, v, speed = velocity_at(k)

        u_masked = np.ma.masked_invalid(u)
        v_masked = np.ma.masked_invalid(v)
        speed_masked = np.ma.masked_invalid(speed)

        linewidth = linewidth_min + (linewidth_max - linewidth_min) * np.clip(
            speed / vmax, 0.0, 1.0
        )

        linewidth = np.ma.masked_invalid(linewidth)

        ax.streamplot(
            x,
            y,
            u_masked,
            v_masked,
            color=speed_masked,
            linewidth=linewidth,
            density=density,
            cmap=cmap,
            norm=norm,
            arrowsize=1.0,
            broken_streamlines=True,
        )

        if show_land_boundary and np.any(water_plot) and np.any(~water_plot):
            ax.contour(
                x,
                y,
                water_plot.astype(float),
                levels=[0.5],
                colors="black",
                linewidths=0.5,
            )

        ax.set_aspect("equal")
        ax.set_title(f"t = {history.t[k]:.3g}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        shrink=0.9,
        label="Velocity magnitude",
    )

    fig.suptitle(f"Velocity streamplots — {scenario_name}", fontsize=16)

    return fig, axes
