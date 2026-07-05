from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from model import (
    Constants,
    ConstantWind,
    Forcing,
    Grid,
    Method,
    State,
    StateHistory,
    StepWind,
    SWEModel,
)
from plotting import (
    plot_scenario_stream_snapshots,
    plot_summary_maps,
    plot_zeta_hovmoller,
    plot_zeta_timeseries,
)

##################################################
# Stuff that don't change ;)                     #
##################################################
CONSTANTS = Constants.from_latitude(60.0)
BASE_GRID = Grid.from_txt("data/bathymetry.txt", 10.0, 10.0)

DT_SEC = 0.1
METHOD = Method.FV_RK4
N_STEPS = 1_000

OUT_DIR = Path("out")


##################################################
# Define the scenarios!                          #
##################################################
GridBuilder = Callable[[Grid], Grid]


def same_grid(base_grid: Grid) -> Grid:
    return base_grid


def barrier_grid(*, row: int) -> GridBuilder:
    def build(base_grid: Grid) -> Grid:
        H = base_grid.H.copy()
        H[row, :] = 0.0
        return Grid(base_grid.dx, base_grid.dy, H)

    return build


@dataclass(frozen=True)
class Scenario:
    name: str
    forcing: Forcing
    grid_builder: GridBuilder = same_grid
    dt: float = DT_SEC
    method: Method = METHOD
    n_steps: int = N_STEPS
    save_every: int = 1


@dataclass
class ScenarioResult:
    scenario: Scenario
    grid: Grid
    model: SWEModel
    history: StateHistory


def define_scenarios(constants: Constants) -> tuple[Scenario, ...]:
    forcing_1 = Forcing(
        wind=ConstantWind(10.0, 0.0),
        constants=constants,
    )

    forcing_2 = Forcing(
        wind=StepWind(
            changes=(
                (0, 0.0, 10.0),
                (50, 10.0, 0.0),
                (200, 0.0, 0.0),
            )
        ),
        constants=constants,
    )

    forcing_3 = Forcing(
        wind=StepWind(
            changes=(
                (0, 10.0, 5.0),
                (50, 10.0, 0.0),
                (200, 0.0, 0.0),
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
# Running the scenarios...                       #
##################################################
def run_scenario(
    scenario: Scenario,
    *,
    base_grid: Grid,
    constants: Constants,
) -> ScenarioResult:
    grid = scenario.grid_builder(base_grid)

    model = SWEModel(
        constants=constants,
        grid=grid,
        forcing=scenario.forcing,
        state=State.init_zeros(grid),
        dt=scenario.dt,
        method=scenario.method,
    )

    history = model.run_with_history(
        scenario.n_steps,
        save_every=scenario.save_every,
    )

    return ScenarioResult(
        scenario=scenario,
        grid=grid,
        model=model,
        history=history,
    )


def run_all_scenarios(
    scenarios: Sequence[Scenario],
    *,
    base_grid: Grid,
    constants: Constants,
) -> list[ScenarioResult]:
    results = []

    for scenario in scenarios:
        print(f"Running {scenario.name}...")
        result = run_scenario(
            scenario,
            base_grid=base_grid,
            constants=constants,
        )
        results.append(result)

    return results


##################################################
# Plots                                          #
##################################################
def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")


def plot_all_results(
    results: Sequence[ScenarioResult],
    *,
    out_dir: Path,
    show: bool = True,
    save: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ##################################################
    # 1. Shared sea-level time series comparison     #
    ##################################################

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    for result in results:
        plot_zeta_timeseries(
            result.history,
            row=25,
            col=10,
            label=result.scenario.name,
            ax=ax,
        )

    ax.set_title("Sea level at row=25, col=10")
    ax.legend()

    if save:
        save_figure(fig, out_dir / "zeta_timeseries.png")

    ##################################################
    # 2. Hovmöller comparison across all scenarios   #
    ##################################################

    histories = {result.scenario.name: result.history for result in results}

    fig, axes = plot_zeta_hovmoller(
        histories,
        results[0].grid,
        y_index=25,
    )

    if save:
        save_figure(fig, out_dir / "zeta_hovmoller.png")

    ##################################################
    # 3. Per-scenario summary maps and streamplots   #
    ##################################################

    for result in results:
        safe_name = result.scenario.name.lower()
        safe_name = safe_name.replace(":", "")
        safe_name = safe_name.replace(" ", "_")

        scenario_dir = out_dir / safe_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plot_summary_maps(
            result.history,
            result.grid,
        )
        fig.suptitle(result.scenario.name)

        if save:
            save_figure(fig, scenario_dir / "summary_maps.png")

        fig, axes = plot_scenario_stream_snapshots(
            result.history,
            result.grid,
            scenario_name=result.scenario.name,
            density=1.5,
        )

        if save:
            save_figure(fig, scenario_dir / "velocity_snapshots.png")

    if show:
        plt.show()
    else:
        plt.close("all")


##################################################
# Visualizer                                     #
##################################################
def run_viz(scenario: Scenario):
    from viz import LakeVisualizer, LakeVisualizerConfig

    # Setup visualizer config
    config = LakeVisualizerConfig(show_shoreline=False)

    # Setup model
    grid = scenario.grid_builder(BASE_GRID)
    model = SWEModel(
        constants=CONSTANTS,
        grid=grid,
        forcing=scenario.forcing,
        state=State.init_zeros(grid),
        dt=scenario.dt,
        method=scenario.method,
    )

    # Run visualizer
    viz = LakeVisualizer(model, config)
    viz.run()


##################################################
# Main                                           #
##################################################
def main(viz: bool = False) -> None:
    scenarios = define_scenarios(CONSTANTS)

    if viz:
        run_viz(scenarios[3])
    else:
        results = run_all_scenarios(
            scenarios,
            base_grid=BASE_GRID,
            constants=CONSTANTS,
        )

        plot_all_results(
            results,
            out_dir=OUT_DIR,
            show=True,
            save=True,
        )


if __name__ == "__main__":
    main()
