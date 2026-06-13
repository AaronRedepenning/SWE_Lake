from model import Constants, Forcing, Grid, Method, State, SWEModel
from plots import LakeVisualizer

##################################################
# Things that don't change :)                    #
##################################################
CONSTANTS = Constants.from_latitude(60.0)
GRID = Grid.from_txt("data/bathymetry.txt", 10.0, 10.0)
FORCING = Forcing(10.0, 0.0, CONSTANTS)

DT_SEC = 0.01
METHOD = Method.RK4

N_STEPS = 1_000


##################################################
# Main                                           #
##################################################
def main():
    ## (1) MODEL SETUP
    state = State.init_zeros(GRID)
    model = SWEModel(CONSTANTS, GRID, FORCING, state, DT_SEC, METHOD)

    ## (2) VISUALIZATION SETUP
    visualizer = LakeVisualizer(
        GRID,
        state,
        factor=5,
        zeta_scale=10000.0,
        steps_per_frame=1,
        fps=1 / 20,
    )

    ## (3) RUN THE MODEL WITH INTERACTIVE VISUALIZATION
    visualizer.run_with_model(model, N_STEPS)
    visualizer.show()


if __name__ == "__main__":
    main()
