from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, zoom


def _load_vispy():
    try:
        from vispy import app, scene
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LakeVisualizer needs VisPy to open an interactive 3D window. "
            "Install it with: pip install vispy"
        ) from exc
    return app, scene


@dataclass(frozen=True)
class LakeVisualizerConfig:
    """Rendering and animation settings for :class:`LakeVisualizer`.

    The visualizer is intentionally model-agnostic. It expects a model with:
      - ``model.grid.H``, ``model.grid.dx``, ``model.grid.dy``
      - ``model.state.z`` or ``model.state.zeta``
      - ``model.state.t``
      - ``model.step()``

    This matches your attached ``SWEModel`` code, where surface elevation is
    stored as ``state.z``.
    """

    # Resampling / vertical exaggeration
    factor: int = 5
    zeta_scale: float = 10_000.0
    bottom_depth_scale: float = 2.0
    water_surface_offset: float = 0.20

    # Water appearance
    water_alpha: float = 0.95
    water_alpha_depth_gain: float = 0.05
    water_overlap_cells: int = 1
    water_smooth_shading: bool = False

    # Optional visual-only small wind ripples. These do not affect the model.
    ripple_height: float = 0.0
    ripple_wavelength: float = 180.0
    ripple_speed: float = 1.3
    ripple_direction: tuple[float, float] = (1.0, 0.35)

    # Shoreline brightening / wet edge
    shoreline_foam_width_cells: int = 2
    shoreline_foam_strength: float = 0.18

    # Visual terrain apron outside the model grid. This does not affect physics.
    terrain_padding_cells: int = 10
    terrain_z: float = 1.0
    terrain_roughness: float = 0.30

    # Procedural ground material. These are visual-only vertex colors: grass,
    # darker/lighter patches, sandy shoreline, and a wet edge near the water.
    terrain_grassiness: float = 0.85
    terrain_texture_strength: float = 0.35
    terrain_patch_scale: float = 650.0
    terrain_fine_scale: float = 115.0
    shore_sand_width_cells: int = 5
    shore_wet_width_cells: int = 2
    shore_sand_strength: float = 0.70
    shore_wet_strength: float = 0.55

    # Animation
    steps_per_frame: int = 5
    interval: float = 1.0 / 20.0

    # Canvas / camera
    canvas_size: tuple[int, int] = (1000, 700)
    bgcolor: str = "black"
    camera_fov: float = 45.0
    camera_elevation: float = 30.0
    camera_azimuth: float = 45.0

    # Scene elements
    show_bottom: bool = True
    show_shoreline: bool = True
    show_axis: bool = False
    show_time: bool = True
    shoreline_size: float = 3.0
    shoreline_z: float = 1.0


class LakeVisualizer:
    """Real-time VisPy renderer for a shallow-water lake model.

    Example
    -------
    >>> visualizer = LakeVisualizer(model)
    >>> visualizer.run()

    You can also advance the model yourself and call ``refresh()`` manually.
    """

    def __init__(self, model: Any, config: LakeVisualizerConfig | None = None) -> None:
        self.model = model
        self.config = config or LakeVisualizerConfig()
        self._validate_model()

        self._app: Any | None = None
        self._scene: Any | None = None

        self.canvas: Any | None = None
        self.view: Any | None = None
        self.water_mesh: Any | None = None
        self.bottom_mesh: Any | None = None
        self.shoreline_markers: Any | None = None
        self.axis: Any | None = None
        self.time_text: Any | None = None
        self.timer: Any | None = None

        self._static_faces_cache: dict[tuple[int, int], np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Create the scene and start the VisPy app loop."""
        self.setup_scene(show=True)
        self._app.run()

    def setup_scene(self, *, show: bool = True) -> Any:
        """Build the VisPy scene without necessarily starting the app loop."""
        app, scene = _load_vispy()
        self._app = app
        self._scene = scene
        cfg = self.config

        self.canvas = scene.SceneCanvas(
            keys="interactive",
            show=show,
            bgcolor=cfg.bgcolor,
            size=cfg.canvas_size,
        )

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=cfg.camera_fov,
            elevation=cfg.camera_elevation,
            azimuth=cfg.camera_azimuth,
        )

        if cfg.show_bottom:
            vertices, faces, colors = self.make_bottom_mesh()
            self.bottom_mesh = scene.visuals.Mesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=colors,
                shading="smooth",
                parent=self.view.scene,
            )

        if cfg.show_shoreline:
            points = self.make_shoreline_points()
            self.shoreline_markers = scene.visuals.Markers(parent=self.view.scene)
            self.shoreline_markers.set_data(
                points,
                face_color=(0.85, 0.78, 0.55, 1.0),
                size=cfg.shoreline_size,
            )

        vertices, faces, colors = self.make_water_mesh()
        self.water_mesh = scene.visuals.Mesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors,
            shading="smooth" if cfg.water_smooth_shading else None,
            parent=self.view.scene,
        )
        self.water_mesh.set_gl_state(
            "translucent",
            depth_test=True,
            cull_face=False,
        )

        if cfg.show_axis:
            self.axis = scene.visuals.XYZAxis(parent=self.view.scene)

        if cfg.show_time:
            self.time_text = scene.visuals.Text(
                text=self._time_label(),
                color="white",
                font_size=14,
                parent=self.canvas.scene,
            )
            self.time_text.pos = (80, 40)

        self.view.camera.set_range()

        self.timer = app.Timer(
            interval=cfg.interval,
            connect=self._on_timer,
            start=True,
        )

        return self.canvas

    def step(self, n: int | None = None) -> None:
        """Advance the model by ``n`` time steps, then refresh the rendered water."""
        count = self.config.steps_per_frame if n is None else n
        for _ in range(count):
            self.model.step()
        self.refresh()

    def refresh(self) -> None:
        """Refresh dynamic visuals from the model's current state."""
        if self.water_mesh is None:
            return

        vertices, faces, colors = self.make_water_mesh()
        self.water_mesh.set_data(vertices=vertices, faces=faces, vertex_colors=colors)

        if self.time_text is not None:
            self.time_text.text = self._time_label()

        if self.canvas is not None:
            self.canvas.update()

    # ------------------------------------------------------------------
    # Mesh builders
    # ------------------------------------------------------------------

    def make_water_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        H_hi, z_hi, water_hi = self.resample_for_visualization(
            self.model.grid.H,
            self._surface_elevation(),
            factor=cfg.factor,
        )

        # Draw one or more high-resolution cells beyond the mathematical water
        # mask. This removes the shoreline gap caused by drawing only quads
        # whose four corners are water.
        render_water_hi = self._expanded_water_mask(water_hi, cfg.water_overlap_cells)

        X, Y = self._xy_mesh(H_hi.shape)
        visual_z = cfg.zeta_scale * z_hi

        if cfg.ripple_height > 0.0:
            visual_z = visual_z + self._procedural_ripples(X, Y)

        Z = np.full_like(H_hi, cfg.water_surface_offset, dtype=np.float32)
        Z[render_water_hi] = visual_z[render_water_hi] + cfg.water_surface_offset

        vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
        faces = self._water_faces(render_water_hi)
        colors = self._water_vertex_colors(
            H_hi=H_hi,
            z_hi=z_hi,
            real_water_hi=water_hi,
            render_water_hi=render_water_hi,
        )

        return vertices, faces, colors

    def make_bottom_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config

        # The physical model only exists on the bathymetry grid, but the visual
        # terrain can extend beyond it. Padding with zero-depth cells creates a
        # surrounding land apron without changing the simulation.
        pad = max(int(cfg.terrain_padding_cells), 0)
        H_for_terrain = np.pad(
            self.model.grid.H,
            pad_width=pad,
            mode="constant",
            constant_values=0.0,
        )
        z_for_terrain = np.zeros_like(H_for_terrain, dtype=np.float64)

        H_hi, _, water_hi = self.resample_for_visualization(
            H_for_terrain,
            z_for_terrain,
            factor=cfg.factor,
        )

        X, Y = self._xy_mesh(H_hi.shape)

        Z = np.empty_like(H_hi, dtype=np.float32)
        Z[water_hi] = -cfg.bottom_depth_scale * H_hi[water_hi]

        terrain = self._terrain_height(X, Y)
        Z[~water_hi] = terrain[~water_hi]

        vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
        faces = self._full_faces(nx=H_hi.shape[1], ny=H_hi.shape[0])
        colors = self._bottom_vertex_colors(
            H_hi=H_hi,
            water_hi=water_hi,
            X=X,
            Y=Y,
        )

        return vertices, faces, colors

    def make_shoreline_points(self) -> np.ndarray:
        cfg = self.config
        _, _, water_hi = self.resample_for_visualization(
            self.model.grid.H,
            self._surface_elevation(),
            factor=cfg.factor,
        )

        X, Y = self._xy_mesh(water_hi.shape)
        land = ~water_hi

        shoreline = water_hi & (
            np.roll(land, 1, axis=0)
            | np.roll(land, -1, axis=0)
            | np.roll(land, 1, axis=1)
            | np.roll(land, -1, axis=1)
        )

        return np.column_stack(
            [
                X[shoreline],
                Y[shoreline],
                np.full(np.sum(shoreline), cfg.shoreline_z),
            ]
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def resample_for_visualization(
        H: np.ndarray,
        z: np.ndarray,
        *,
        factor: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if factor < 1:
            raise ValueError("factor must be >= 1")
        if H.shape != z.shape:
            raise ValueError(f"H shape {H.shape} does not match z shape {z.shape}")

        water = H > 0.0

        # Nearest-neighbor resampling preserves a crisp land/water boundary.
        water_hi = zoom(water.astype(float), factor, order=0) > 0.5

        # Smooth depth and surface elevation for visualization only.
        H_hi = zoom(H, factor, order=1)
        z_hi = zoom(z, factor, order=3)

        H_hi = np.where(water_hi, H_hi, 0.0)
        z_hi = np.where(water_hi, z_hi, 0.0)

        return H_hi.astype(np.float32), z_hi.astype(np.float32), water_hi

    def _xy_mesh(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = shape
        dx_hi = self.model.grid.dx / self.config.factor
        dy_hi = self.model.grid.dy / self.config.factor

        x = (np.arange(nx) - 0.5 * (nx - 1)) * dx_hi
        y = (np.arange(ny) - 0.5 * (ny - 1)) * dy_hi
        return np.meshgrid(x, y)

    def _expanded_water_mask(self, water_hi: np.ndarray, cells: int) -> np.ndarray:
        cells = max(int(cells), 0)
        if cells == 0 or not np.any(water_hi):
            return water_hi
        return binary_dilation(water_hi, iterations=cells)

    def _full_faces(self, *, nx: int, ny: int) -> np.ndarray:
        key = (nx, ny)
        cached = self._static_faces_cache.get(key)
        if cached is not None:
            return cached

        faces: list[list[int]] = []
        for j in range(ny - 1):
            row = j * nx
            next_row = (j + 1) * nx
            for i in range(nx - 1):
                v00 = row + i
                v10 = row + i + 1
                v01 = next_row + i
                v11 = next_row + i + 1
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])

        out = np.asarray(faces, dtype=np.uint32)
        self._static_faces_cache[key] = out
        return out

    def _water_faces(self, water_hi: np.ndarray) -> np.ndarray:
        ny, nx = water_hi.shape
        faces: list[list[int]] = []

        for j in range(ny - 1):
            row = j * nx
            next_row = (j + 1) * nx
            for i in range(nx - 1):
                # Draw every quad touched by water. Requiring all four corners
                # to be water makes the lake shrink inward and creates a gap.
                if not (
                    water_hi[j, i]
                    or water_hi[j, i + 1]
                    or water_hi[j + 1, i]
                    or water_hi[j + 1, i + 1]
                ):
                    continue

                v00 = row + i
                v10 = row + i + 1
                v01 = next_row + i
                v11 = next_row + i + 1
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])

        return np.asarray(faces, dtype=np.uint32)

    def _water_vertex_colors(
        self,
        *,
        H_hi: np.ndarray,
        z_hi: np.ndarray,
        real_water_hi: np.ndarray,
        render_water_hi: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        colors = np.zeros((*H_hi.shape, 4), dtype=np.float32)

        if np.any(render_water_hi):
            depth = H_hi[render_water_hi]
            depth_scale = max(
                float(H_hi[real_water_hi].max()) if np.any(real_water_hi) else 0.0, 1e-8
            )
            depth_norm = np.clip(depth / depth_scale, 0.0, 1.0)

            z = (
                z_hi[real_water_hi]
                if np.any(real_water_hi)
                else np.array([0.0], dtype=np.float32)
            )
            z_scale = max(abs(float(z.min())), abs(float(z.max())), 1e-8)
            wave_q = np.clip(0.5 + 0.5 * z_hi[render_water_hi] / z_scale, 0.0, 1.0)

            # Shallow water is slightly greener/cyan; deep water is darker blue.
            colors[render_water_hi, 0] = 0.035 + 0.045 * wave_q - 0.020 * depth_norm
            colors[render_water_hi, 1] = 0.330 + 0.170 * wave_q - 0.145 * depth_norm
            colors[render_water_hi, 2] = 0.560 + 0.185 * wave_q + 0.175 * depth_norm

            # Less-transparent water. Deep cells become almost opaque while
            # shoreline cells remain just slightly translucent.
            colors[render_water_hi, 3] = np.clip(
                cfg.water_alpha + cfg.water_alpha_depth_gain * depth_norm,
                0.0,
                1.0,
            )

            # Subtle pale shoreline/wet-edge effect.
            if cfg.shoreline_foam_width_cells > 0 and cfg.shoreline_foam_strength > 0.0:
                shore_band = real_water_hi & binary_dilation(
                    ~real_water_hi,
                    iterations=cfg.shoreline_foam_width_cells,
                )
                if np.any(shore_band):
                    foam_rgb = np.array([0.75, 0.90, 0.95], dtype=np.float32)
                    strength = np.clip(cfg.shoreline_foam_strength, 0.0, 1.0)
                    colors[shore_band, :3] = (1.0 - strength) * colors[
                        shore_band, :3
                    ] + strength * foam_rgb
                    colors[shore_band, 3] = np.maximum(colors[shore_band, 3], 0.96)

        return colors.reshape(-1, 4).astype(np.float32)

    def _procedural_ripples(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        cfg = self.config
        t = float(getattr(self.model.state, "t", 0.0))

        dx, dy = cfg.ripple_direction
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-8)
        dx, dy = dx / norm, dy / norm

        wavelength = max(float(cfg.ripple_wavelength), 1e-8)
        phase_1 = 2.0 * np.pi * (dx * X + dy * Y) / wavelength
        phase_2 = 2.0 * np.pi * (-dy * X + dx * Y) / (0.55 * wavelength)

        ripples = cfg.ripple_height * np.sin(phase_1 + cfg.ripple_speed * t)
        ripples += (
            0.35 * cfg.ripple_height * np.sin(phase_2 + 1.7 * cfg.ripple_speed * t)
        )
        return ripples.astype(np.float32)

    def _terrain_height(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        cfg = self.config
        if cfg.terrain_roughness <= 0.0:
            return np.full_like(X, cfg.terrain_z, dtype=np.float32)

        # Deterministic low-frequency terrain variation. It is purely visual,
        # so it does not affect bathymetry or shallow-water dynamics.
        rough = (
            0.55 * np.sin(X / 750.0 + 1.7)
            + 0.35 * np.cos(Y / 620.0 - 0.8)
            + 0.25 * np.sin((X + 0.6 * Y) / 1100.0)
        )
        return (cfg.terrain_z + cfg.terrain_roughness * rough).astype(np.float32)

    def _bottom_vertex_colors(
        self,
        *,
        H_hi: np.ndarray,
        water_hi: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        colors = np.zeros((*H_hi.shape, 4), dtype=np.float32)

        if np.any(water_hi):
            depth = H_hi[water_hi]
            depth_norm = depth / max(float(depth.max()), 1e-8)

            # Lake bed: shallow areas slightly warmer, deep areas darker.
            colors[water_hi, 0] = 0.18 - 0.10 * depth_norm
            colors[water_hi, 1] = 0.15 - 0.05 * depth_norm
            colors[water_hi, 2] = 0.10 - 0.03 * depth_norm
            colors[water_hi, 3] = 1.0

        land = ~water_hi
        if np.any(land):
            texture = self._terrain_texture(X, Y)
            patch = self._terrain_patchiness(X, Y)
            fine = self._terrain_fine_noise(X, Y)

            grassiness = np.clip(cfg.terrain_grassiness, 0.0, 1.0)
            tex_strength = np.clip(cfg.terrain_texture_strength, 0.0, 1.0)

            # Base material palette, chosen to read well under VisPy lighting.
            dry_grass = np.array([0.22, 0.38, 0.13], dtype=np.float32)
            bright_grass = np.array([0.34, 0.52, 0.18], dtype=np.float32)
            scrub = np.array([0.19, 0.27, 0.12], dtype=np.float32)
            dirt = np.array([0.34, 0.25, 0.13], dtype=np.float32)
            sand = np.array([0.62, 0.54, 0.34], dtype=np.float32)
            wet_soil = np.array([0.16, 0.13, 0.09], dtype=np.float32)

            # Blend grass and dirt in slow, irregular patches. This gives the
            # padded apron a natural mottled look without requiring image files.
            grass_mix = np.clip(0.55 + 0.45 * patch, 0.0, 1.0)
            grass_rgb = (
                scrub[None, None, :] * (1.0 - grass_mix[..., None])
                + bright_grass[None, None, :] * grass_mix[..., None]
            )
            base_rgb = dirt[None, None, :] * (1.0 - grassiness) + grass_rgb * grassiness

            # Fine texture gives a grass/noise effect at vertex level.
            shade = 1.0 + tex_strength * (0.35 * texture + 0.20 * fine)
            base_rgb = np.clip(base_rgb * shade[..., None], 0.0, 1.0)

            # Add irregular bare-earth patches so it does not look uniformly green.
            bare_patch = np.clip((0.32 - patch) / 0.42, 0.0, 1.0)
            base_rgb = base_rgb * (1.0 - 0.45 * bare_patch[..., None]) + dirt[
                None, None, :
            ] * (0.45 * bare_patch[..., None])

            colors[land, :3] = base_rgb[land]
            colors[land, 3] = 1.0

            # Sandy and wet shoreline rings on land immediately around water.
            if np.any(water_hi):
                if cfg.shore_sand_width_cells > 0 and cfg.shore_sand_strength > 0.0:
                    sand_band = land & binary_dilation(
                        water_hi,
                        iterations=max(int(cfg.shore_sand_width_cells), 1),
                    )
                    if np.any(sand_band):
                        strength = np.clip(cfg.shore_sand_strength, 0.0, 1.0)
                        colors[sand_band, :3] = (1.0 - strength) * colors[
                            sand_band, :3
                        ] + strength * sand

                if cfg.shore_wet_width_cells > 0 and cfg.shore_wet_strength > 0.0:
                    wet_band = land & binary_dilation(
                        water_hi,
                        iterations=max(int(cfg.shore_wet_width_cells), 1),
                    )
                    if np.any(wet_band):
                        strength = np.clip(cfg.shore_wet_strength, 0.0, 1.0)
                        colors[wet_band, :3] = (1.0 - strength) * colors[
                            wet_band, :3
                        ] + strength * wet_soil

        return colors.reshape(-1, 4).astype(np.float32)

    def _terrain_texture(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Deterministic medium-frequency color texture in [-1, 1]."""
        scale = max(float(self.config.terrain_fine_scale), 1e-8)
        tex = (
            0.50 * np.sin(X / scale + 0.7 * np.sin(Y / (1.7 * scale)))
            + 0.35 * np.cos(Y / (1.3 * scale) - 0.4 * np.sin(X / (2.1 * scale)))
            + 0.15 * np.sin((X - 1.8 * Y) / (0.8 * scale))
        )
        return np.clip(tex, -1.0, 1.0).astype(np.float32)

    def _terrain_patchiness(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Deterministic low-frequency patch map in [0, 1]."""
        scale = max(float(self.config.terrain_patch_scale), 1e-8)
        patch = (
            0.55 * np.sin(X / scale + 1.6)
            + 0.35 * np.cos(Y / (0.75 * scale) - 0.4)
            + 0.25 * np.sin((X + 0.85 * Y) / (1.4 * scale) + 2.1)
        )
        return np.clip(0.5 + 0.5 * patch, 0.0, 1.0).astype(np.float32)

    def _terrain_fine_noise(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Small deterministic speckle-like variation in [-1, 1]."""
        # This uses only sin/cos so it remains deterministic and dependency-free.
        noise = np.sin(12.9898 * X + 78.233 * Y) * 43758.5453
        noise = noise - np.floor(noise)
        return (2.0 * noise - 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Model compatibility helpers
    # ------------------------------------------------------------------

    def _surface_elevation(self) -> np.ndarray:
        state = self.model.state
        if hasattr(state, "z"):
            return state.z
        if hasattr(state, "zeta"):
            return state.zeta
        raise AttributeError(
            "model.state must expose surface elevation as `z` or `zeta`"
        )

    def _time_label(self) -> str:
        t = getattr(self.model.state, "t", 0.0)
        return f"t = {t:.1f} s"

    def _validate_model(self) -> None:
        for attr in ("grid", "state", "step"):
            if not hasattr(self.model, attr):
                raise TypeError(f"model must have a `{attr}` attribute")

        for attr in ("H", "dx", "dy"):
            if not hasattr(self.model.grid, attr):
                raise TypeError(f"model.grid must have a `{attr}` attribute")

        z = self._surface_elevation()
        H = self.model.grid.H
        if H.shape != z.shape:
            raise ValueError(
                f"grid.H shape {H.shape} does not match surface shape {z.shape}"
            )

    def _on_timer(self, event: Any) -> None:
        self.step(self.config.steps_per_frame)
