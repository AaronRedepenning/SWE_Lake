from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model import (
    Constants,
    ConstantWind,
    Forcing,
    Grid,
    GridType,
    IntegrationMethod,
    State,
    StateHistory,
    StepWind,
    SWEModel,
)
from plotting import (
    VorticityEKE,
    plot_summary_maps,
    plot_vorticity_eke_comparison,
    plot_zeta_hovmoller,
    plot_zeta_timeseries,
    show_streamplot_animation,
)

##################################################
# Output configs ;)                              #
##################################################
SHOW = True  # Show plots
SAVE = False  # Save plots
OUT_DIR = Path("out")

##################################################
# Model configs                                  #
##################################################
CONSTANTS = Constants.from_latitude(60.0)
BASE_GRID = Grid.from_txt("data/bathymetry.txt", 1000.0, 1000.0)

METHOD_CONFIGS = [
    (
        True,  # fully linear
        IntegrationMethod.EULER,  # Integration method
        GridType.CENTERED,  # Grid type
        5.0,  # Time step [sec]
        1_000,  # Number of iterations
        1,  # Save every (save resolution)
    ),
    (
        True,  # fully linear
        IntegrationMethod.RK4,  # Integration method
        GridType.C_GRID,  # Grid type
        5.0,  # Time step [sec]
        1_000,  # Number of iterations
        1,  # Save every (save resolution)
    ),
]


##################################################
# Scenario definitions!                          #
##################################################
GridBuilder = Callable[[Grid], Grid]


def basic_grid(base_grid: Grid) -> Grid:
    return base_grid


def barrier_grid(*, row: int) -> GridBuilder:
    def build(base_grid: Grid) -> Grid:
        H = base_grid.H.copy()
        H[row, :] = 0.0
        return Grid(base_grid.dx, base_grid.dy, H)

    return build


@dataclass
class Scenario:
    name: str
    forcing: Forcing
    grid_builder: GridBuilder = basic_grid


def create_scenarios(constants: Constants) -> tuple[Scenario, ...]:
    forcing_1 = Forcing(
        wind=ConstantWind(10.0, 0.0),
        constants=constants,
    )

    forcing_2 = Forcing(
        wind=StepWind(
            changes=(
                (0, 0.0, 10.0),
                (50, 10.0, 0.0),
                (250, 0.0, 0.0),
            )
        ),
        constants=constants,
    )

    forcing_3 = Forcing(
        wind=StepWind(
            changes=(
                (0, 10.0, 5.0),
                (50, 0.0, 0.0),
            )
        ),
        constants=constants,
    )

    return (
        Scenario(
            name="Scenario 1",
            forcing=forcing_1,
        ),
        Scenario(
            name="Scenario 2",
            forcing=forcing_2,
        ),
        Scenario(
            name="Scenario 3",
            forcing=forcing_3,
        ),
        Scenario(
            name="Scenario 4",
            forcing=forcing_3,
            grid_builder=barrier_grid(row=20),
        ),
    )


##################################################
# Helper functions                               #
##################################################
def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")


def calculate_vorticity_eke(history: StateHistory, grid: Grid) -> VorticityEKE:
    U_all, V_all = history.U, history.V
    H_model, dx, dy = grid.H, grid.dx, grid.dy
    water = grid.water_mask

    u_all = np.zeros_like(U_all)
    v_all = np.zeros_like(V_all)
    u_all[:, water] = U_all[:, water] / H_model[water]
    v_all[:, water] = V_all[:, water] / H_model[water]

    dv_dx = np.gradient(v_all, dx, axis=2)
    du_dy = np.gradient(u_all, dy, axis=1)
    mean_vorticity = np.mean(dv_dx - du_dy, axis=0)

    u_prime = u_all - np.mean(u_all, axis=0)
    v_prime = v_all - np.mean(v_all, axis=0)
    mean_eke = np.mean(0.5 * (u_prime**2 + v_prime**2), axis=0)

    mean_vorticity[~water] = np.nan
    mean_eke[~water] = np.nan

    return VorticityEKE(mean_vorticity, mean_eke)


##################################################
# Main                                           #
##################################################
def main(show: bool = True, save: bool = True) -> None:
    # Run all different scenarios
    all_vorticity_eke: dict[str, dict[str, VorticityEKE]] = {}

    for scenario in create_scenarios(CONSTANTS):
        print(f"Running {scenario.name}:")

        # Run each scenario using both method (Andy vs. Aaron)
        results: dict[str, StateHistory] = {}
        scenario_vorticity_eke: dict[str, VorticityEKE] = {}
        grid = scenario.grid_builder(BASE_GRID)

        scenario_dir = OUT_DIR / scenario.name.replace(" ", "_").lower()
        scenario_dir.mkdir(parents=True, exist_ok=True)

        for (
            linear,
            integration,
            grid_type,
            dt,
            n_steps,
            save_every,
        ) in METHOD_CONFIGS:
            print(f"  linear = {linear}")
            print(f"  method = {integration.name}")
            print(f"  grid_type = {grid_type.name}")

            model = SWEModel(
                constants=CONSTANTS,
                grid=grid,
                forcing=scenario.forcing,
                state=State.init_zeros(grid, grid_type),
                dt=dt,
                linear=linear,
                integration=integration,
                grid_type=grid_type,
            )

            history = model.run_with_history(n_steps, save_every=save_every)
            results[f"{integration.name} / {grid_type.name}"] = history

            # Calculate vorticity and eddy kenetic energy
            vorticity_eke = calculate_vorticity_eke(
                history,
                grid,
            )
            scenario_vorticity_eke[integration.name] = vorticity_eke

            print(
                f"    mean abs vorticity = {np.nanmean(np.abs(vorticity_eke.mean_vorticity)):.6e}"
            )
            print(f"    mean EKE = {np.nanmean(vorticity_eke.mean_eke):.6e}")

        all_vorticity_eke[scenario.name] = scenario_vorticity_eke

        ##################################################
        # 1. Shared sea-level time series comparison     #
        ##################################################
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

        for method, history in results.items():
            plot_zeta_timeseries(
                history,
                row=25,
                col=10,
                label=method,
                ax=ax,
            )

        ax.set_title(f"{scenario.name}: Sea level at [25, 10]")
        ax.legend()

        # Save figure
        if save:
            save_figure(
                fig,
                scenario_dir / "timeseries.png",
            )

        ##################################################
        # 2. Summary maps                                #
        ##################################################
        fig, ax = plot_summary_maps(results, grid)

        fig.suptitle(f"{scenario.name}: Summary maps")

        # Save figure
        if save:
            save_figure(
                fig,
                scenario_dir / "summary_maps.png",
            )

        ##################################################
        # 3. Hovmöller plot comparison                   #
        ##################################################
        fig, ax = plot_zeta_hovmoller(
            results,
            grid,
            y_index=25,
        )

        fig.suptitle(f"{scenario.name}: Hovmöller plot of ζ along y-index 25")

        # Save figure
        if save:
            save_figure(
                fig,
                scenario_dir / "hovmoller.png",
            )

        ##################################################
        # 4. Streamplot animations                       #
        ##################################################
        fig, ax, anim = show_streamplot_animation(results, grid)

        fig.suptitle(f"{scenario.name}: Streamplots")

        # Save animation
        if save:
            anim.save(
                scenario_dir / "streamplot.gif",
                writer="pillow",
                dpi=200,
            )

        print(f"  plots saved to {scenario_dir}")
        print()

        # Show ALL plots
        if show:
            plt.show()
        plt.close("all")

    ##################################################
    # 5. Vorticity + EKE plot                        #
    ##################################################
    fig, ax = plot_vorticity_eke_comparison(
        all_vorticity_eke,
    )

    fig.suptitle("Mean vorticity and EKE")

    if save:
        save_figure(
            fig,
            OUT_DIR / "vorticity_eke.png",
        )

    if show:
        plt.show()

    plt.close("all")


if __name__ == "__main__":
    main(show=SHOW, save=SAVE)
