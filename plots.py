from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom
from vispy import app, scene

from model import Grid, State

# =====================================================================
# Visualization helper functions
# =====================================================================


def resample_for_visualization(
    H: np.ndarray,
    zeta: np.ndarray,
    factor: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample bathymetry and water elevation for smoother visualization."""
    water = H > 0.0

    # Nearest-neighbor keeps land/water boundary crisp.
    water_hi = zoom(water.astype(float), factor, order=0) > 0.5

    # Smooth arrays for visualization.
    H_hi = zoom(H, factor, order=1)
    zeta_hi = zoom(zeta, factor, order=3)

    # Keep land cells flat.
    H_hi = np.where(water_hi, H_hi, 0.0)
    zeta_hi = np.where(water_hi, zeta_hi, 0.0)

    return H_hi, zeta_hi, water_hi


def make_faces(nx: int, ny: int) -> np.ndarray:
    """Generate triangle faces for a grid mesh."""
    faces: list[list[int]] = []

    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = j * nx + i
            v10 = j * nx + (i + 1)
            v01 = (j + 1) * nx + i
            v11 = (j + 1) * nx + (i + 1)

            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    return np.asarray(faces, dtype=np.uint32)


def make_vertex_colors(
    zeta_hi: np.ndarray,
    water_hi: np.ndarray,
) -> np.ndarray:
    """Generate vertex colors based on water elevation and bathymetry."""
    ny, nx = zeta_hi.shape

    colors = np.zeros((ny, nx, 4), dtype=np.float32)

    water = water_hi
    land = ~water_hi

    # Land color: earthy brown-green.
    colors[land] = np.array([0.35, 0.28, 0.12, 1.0], dtype=np.float32)

    # Water color: blue, modulated by zeta.
    if np.any(water):
        z = zeta_hi[water]
        scale = max(abs(z.min()), abs(z.max()), 1e-8)

        q = 0.5 + 0.5 * zeta_hi / scale
        q = np.clip(q, 0.0, 1.0)

        colors[water, 0] = 0.05 + 0.10 * q[water]
        colors[water, 1] = 0.25 + 0.35 * q[water]
        colors[water, 2] = 0.70 + 0.25 * q[water]
        colors[water, 3] = 1.0

    return colors.reshape(-1, 4).astype(np.float32)


def make_water_mesh(
    grid: Grid,
    state: State,
    *,
    factor: int = 5,
    zeta_scale: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate water surface mesh with color mapping."""
    H_hi, zeta_hi, water_hi = resample_for_visualization(
        grid.H,
        state.z,
        factor=factor,
    )

    ny, nx = H_hi.shape

    dx_hi = grid.dx / factor
    dy_hi = grid.dy / factor

    x = (np.arange(nx) - 0.5 * (nx - 1)) * dx_hi
    y = (np.arange(ny) - 0.5 * (ny - 1)) * dy_hi
    X, Y = np.meshgrid(x, y)

    # Surface height:
    #   water = exaggerated zeta
    #   land  = flat at z = 0
    Z = np.zeros_like(H_hi, dtype=np.float32)
    Z[water_hi] = zeta_scale * zeta_hi[water_hi]
    Z[~water_hi] = 0.0

    vertices = np.column_stack(
        [
            X.ravel(),
            Y.ravel(),
            Z.ravel(),
        ]
    ).astype(np.float32)

    faces = make_faces(nx=nx, ny=ny)

    vertex_colors = make_vertex_colors(
        zeta_hi=zeta_hi,
        water_hi=water_hi,
    )

    return vertices, faces, vertex_colors


def make_bottom_mesh(
    grid: Grid,
    state: State,
    *,
    factor: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate lake bottom mesh."""
    H_hi, zeta_hi, water_hi = resample_for_visualization(
        grid.H * 10.0,
        state.z,
        factor=factor,
    )

    ny, nx = H_hi.shape
    dx_hi = grid.dx / factor
    dy_hi = grid.dy / factor

    x = (np.arange(nx) - 0.5 * (nx - 1)) * dx_hi
    y = (np.arange(ny) - 0.5 * (ny - 1)) * dy_hi
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(H_hi, dtype=np.float32)
    Z[water_hi] = -H_hi[water_hi]
    Z[~water_hi] = 1.0

    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
    faces = make_faces(nx=nx, ny=ny)

    colors = np.zeros((ny, nx, 4), dtype=np.float32)

    # Underwater bottom: dark muted blue/brown
    colors[water_hi] = np.array([0.08, 0.10, 0.12, 1.0], dtype=np.float32)

    # Land: brown/green
    colors[~water_hi] = np.array([0.30, 0.24, 0.12, 1.0], dtype=np.float32)

    return vertices, faces, colors.reshape(-1, 4)


def make_shoreline_points(
    grid: Grid,
    state: State,
    *,
    factor: int = 5,
) -> np.ndarray:
    """Generate shoreline marker points."""
    H_hi, zeta_hi, water_hi = resample_for_visualization(
        grid.H,
        state.z,
        factor=factor,
    )

    ny, nx = water_hi.shape
    dx_hi = grid.dx / factor
    dy_hi = grid.dy / factor

    x = (np.arange(nx) - 0.5 * (nx - 1)) * dx_hi
    y = (np.arange(ny) - 0.5 * (ny - 1)) * dy_hi
    X, Y = np.meshgrid(x, y)

    land = ~water_hi

    # Cells that are water and adjacent to land
    shoreline = water_hi & (
        np.roll(land, 1, axis=0)
        | np.roll(land, -1, axis=0)
        | np.roll(land, 1, axis=1)
        | np.roll(land, -1, axis=1)
    )

    points = np.column_stack(
        [
            X[shoreline],
            Y[shoreline],
            np.full(np.sum(shoreline), 1.0),
        ]
    ).astype(np.float32)

    return points


# =====================================================================
# Lake Visualizer Class
# =====================================================================


class LakeVisualizer:
    """Realistic 3D lake visualizer using VisPy."""

    def __init__(
        self,
        grid: Grid,
        state: State,
        *,
        factor: int = 5,
        zeta_scale: float = 10000.0,
        steps_per_frame: int = 5,
        fps: float = 1 / 20,
        window_size: tuple[int, int] = (1200, 900),
    ):
        """
        Initialize the lake visualizer.

        Args:
            grid: The Grid object containing bathymetry data
            state: Initial State object
            factor: Resampling factor for smoother visualization
            zeta_scale: Exaggeration factor for water elevation
            steps_per_frame: Number of model steps per visualization frame
            fps: Target frames per second for animation
            window_size: Canvas window size (width, height)
        """
        self.grid = grid
        self.state = state
        self.factor = factor
        self.zeta_scale = zeta_scale
        self.steps_per_frame = steps_per_frame
        self.fps = fps

        # Create canvas and scene
        self.canvas = scene.SceneCanvas(
            keys="interactive",
            show=True,
            bgcolor="black",
            size=window_size,
        )

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=45,
            elevation=30,
            azimuth=45,
        )

        # Create meshes
        self._create_meshes()

        # Setup text display
        self.text = scene.visuals.Text(
            text=f"t = {self.state.t:.1f} s",
            color="white",
            font_size=14,
            parent=self.canvas.scene,
        )
        self.text.pos = (80, 40)

        # Add axis
        scene.visuals.XYZAxis(parent=self.view.scene)

        # Set camera range
        self.view.camera.set_range()

        # Setup animation timer
        self.timer = app.Timer(
            interval=self.fps,
            connect=self._on_update,
            start=False,
        )

    def _create_meshes(self):
        """Create the bottom, water, and shoreline meshes."""
        # Bottom mesh
        bottom_vertices, bottom_faces, bottom_colors = make_bottom_mesh(
            self.grid,
            self.state,
            factor=self.factor,
        )

        self.bottom_mesh = scene.visuals.Mesh(
            vertices=bottom_vertices,
            faces=bottom_faces,
            vertex_colors=bottom_colors,
            shading="smooth",
            parent=self.view.scene,
        )

        # Water mesh
        vertices, faces, vertex_colors = make_water_mesh(
            self.grid,
            self.state,
            factor=self.factor,
            zeta_scale=self.zeta_scale,
        )

        self.water_mesh = scene.visuals.Mesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            shading=None,
            parent=self.view.scene,
        )

        self.water_mesh.set_gl_state(
            "translucent",
            depth_test=True,
            cull_face=False,
        )

        # Shoreline markers
        shore_points = make_shoreline_points(self.grid, self.state, factor=self.factor)

        self.shore = scene.visuals.Markers()
        self.shore.set_data(
            shore_points,
            face_color=(0.85, 0.78, 0.55, 1.0),
            size=3,
        )
        self.view.add(self.shore)

    def _on_update(self, event) -> None:
        """Internal update callback for animation timer."""
        # This will be set when run_with_model is called
        pass

    def run_with_model(self, model, n_steps: int) -> None:
        """
        Run the model and update visualization in real-time.

        Args:
            model: The SWEModel object
            n_steps: Number of steps to run
        """
        step_counter = [0]  # Use list to allow modification in nested function

        def update_loop(event) -> None:
            """Update callback for animation timer."""
            for _ in range(self.steps_per_frame):
                step_counter[0] += 1
                try:
                    # Get next state from model
                    self.state = model.state.copy()

                    # Update meshes
                    vertices, faces, vertex_colors = make_water_mesh(
                        self.grid,
                        self.state,
                        factor=self.factor,
                        zeta_scale=self.zeta_scale,
                    )

                    self.water_mesh.set_data(
                        vertices=vertices,
                        faces=faces,
                        vertex_colors=vertex_colors,
                    )

                    # Update shoreline
                    shore_points = make_shoreline_points(
                        self.grid,
                        self.state,
                        factor=self.factor,
                    )
                    self.shore.set_data(shore_points)

                    # Update text
                    self.text.text = (
                        f"t = {self.state.t:.1f} s  |  Step {step_counter[0]}/{n_steps}"
                    )

                except StopIteration:
                    self.timer.stop()
                    break

        # Replace the update callback
        self.timer.connect(update_loop)

        # Run model and visualization together
        for step, state in model.run(n_steps):
            self.state = state.copy()

            # Update meshes
            vertices, faces, vertex_colors = make_water_mesh(
                self.grid,
                self.state,
                factor=self.factor,
                zeta_scale=self.zeta_scale,
            )

            self.water_mesh.set_data(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors,
            )

            # Update shoreline
            shore_points = make_shoreline_points(
                self.grid,
                self.state,
                factor=self.factor,
            )
            self.shore.set_data(shore_points)

            # Update text
            self.text.text = f"t = {self.state.t:.1f} s  |  Step {step}/{n_steps}"

            self.canvas.update()

    def show(self) -> None:
        """Display the visualization."""
        app.run()
