import bisect
import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Vector = tuple[float, float]
ArrayShape = tuple[int, int]
ArrayMask = NDArray[np.bool_]


##################################################
# Integration methods                            #
##################################################
class IntegrationMethod(StrEnum):
    EULER = "Euler"
    RK4 = "RK4"


##################################################
# Grid types                                     #
##################################################
class GridType(StrEnum):
    CENTERED = "Centered"
    C_GRID = "C-Grid"


def array_shapes(
    grid: "Grid", grid_type: GridType
) -> tuple[ArrayShape, ArrayShape, ArrayShape]:
    """
    Compute grid shapes depending on the gridding method.
    """
    if grid_type == GridType.CENTERED:
        shape = grid.shape
        return shape, shape, shape

    elif grid_type == GridType.C_GRID:
        ny, nx = grid.shape
        return (ny, nx), (ny, nx + 1), (ny + 1, nx)

    else:
        raise NotImplementedError(grid_type)


def array_water_masks(
    grid: "Grid", grid_type: GridType
) -> tuple[ArrayMask, ArrayMask, ArrayMask]:
    if grid_type == GridType.CENTERED:
        water = grid.water_mask
        return water, water, water

    elif grid_type == GridType.C_GRID:
        water = grid.water_mask
        _, U_shape, V_shape = array_shapes(grid, grid_type)

        # U mask
        U_mask = np.zeros(U_shape, dtype=np.bool_)

        left_open = grid.H[:, :-1] > 0
        right_open = grid.H[:, 1:] > 0

        U_mask[:, 1:-1] = left_open & right_open

        # V mask
        V_mask = np.zeros(V_shape, dtype=np.bool_)

        top_open = grid.H[:-1, :] > 0
        bottom_open = grid.H[1:, :] > 0

        V_mask[1:-1, :] = top_open & bottom_open

        return water, U_mask, V_mask

    else:
        raise NotImplementedError(grid_type)


def interpolate_grid(
    state: "State",
    grid_type: GridType,
) -> tuple[Array, Array, Array]:
    z, U, V = state.z, state.U, state.V

    if grid_type == GridType.C_GRID:
        U = 0.5 * (U[:, 1:] + U[:, :-1])
        V = 0.5 * (V[1:, :] + V[:-1, :])

    return z, U, V


##################################################
# Physical & Model Constants                     #
##################################################
@dataclass(frozen=True)
class Constants:
    # Physical
    g: float = 9.81  # Gravity
    omega: float = 7.2921159e-5  # Earth's rotation

    # Model
    r: float = 0.003  # Frictional coefficient
    wsf: float = 3.2e-6  # Wind stress factor
    f: float = 0  # Coriolis

    @classmethod
    def from_latitude(
        cls,
        latitude_deg: float,
        *,
        g: float = 9.81,
        omega: float = 7.2921159e-5,
        r: float = 0.003,
        wsf: float = 3.3e-6,
    ) -> "Constants":
        latitude_rad = math.radians(latitude_deg)
        f = 2.0 * omega * math.sin(latitude_rad)

        return cls(
            g=g,
            omega=omega,
            r=r,
            wsf=wsf,
            f=f,
        )


##################################################
# Grid Structure                                 #
##################################################
@dataclass(frozen=True)
class Grid:
    dx: float
    dy: float
    H: Array

    @classmethod
    def from_txt(cls, path: str | Path, dx: float, dy: float) -> "Grid":
        H = np.loadtxt(
            path,
            dtype=np.float64,
        )

        if H.ndim != 2:
            raise ValueError(f"Expected H to be a 2D array, got shape {H.shape}")

        return cls(dx, dy, H)

    @property
    def water_mask(self):
        return self.H > 0.0

    @property
    def land_mask(self):
        return ~self.water_mask

    @property
    def nx(self):
        return self.H.shape[1]

    @property
    def ny(self):
        return self.H.shape[0]

    @property
    def shape(self):
        return self.H.shape


##################################################
# External Forcing                               #
##################################################
class WindModel(Protocol):
    def __call__(self, *, t: float, step: int) -> Vector: ...


@dataclass(frozen=True)
class ConstantWind:
    wx: float
    wy: float

    def __call__(self, *, t: float, step: int) -> Vector:
        return self.wx, self.wy


@dataclass(frozen=True)
class StepWind:
    changes: tuple[tuple[int, float, float], ...]

    def __call__(self, *, t: float, step: int) -> Vector:
        steps = [s for s, _, _ in self.changes]
        idx = bisect.bisect_right(steps, step) - 1
        idx = max(idx, 0)

        _, wx, wy = self.changes[idx]
        return wx, wy


@dataclass(frozen=True)
class Forcing:
    wind: WindModel
    constants: Constants

    def wind_stress(self, t: float, step: int, linear: bool = False) -> Vector:
        wx, wy = self.wind(t=t, step=step)
        wsf = self.constants.wsf

        if linear:
            return (
                wsf * wx,
                wsf * wy,
            )
        else:
            wind_speed = math.sqrt(wx**2 + wy**2)
            return (
                wsf * wx * wind_speed,
                wsf * wy * wind_speed,
            )


##################################################
# Model State                                    #
##################################################
@dataclass
class State:
    z: Array
    U: Array
    V: Array
    t: float
    step: int

    @classmethod
    def init_zeros(
        cls,
        grid: Grid,
        grid_type: GridType = GridType.CENTERED,
    ) -> "State":
        z_shape, U_shape, V_shape = array_shapes(grid, grid_type)

        return cls(
            z=np.zeros(z_shape, dtype=np.float64),
            U=np.zeros(U_shape, dtype=np.float64),
            V=np.zeros(V_shape, dtype=np.float64),
            t=0.0,
            step=0,
        )

    def copy(self) -> "State":
        return State(
            z=self.z.copy(),
            U=self.U.copy(),
            V=self.V.copy(),
            t=self.t,
            step=self.step,
        )


##################################################
# State History                                  #
##################################################
@dataclass
class StateHistory:
    steps: NDArray[np.int64]
    t: Array
    z: Array
    U: Array
    V: Array

    _i: int = field(default=0, init=False, repr=False)

    @classmethod
    def allocate(
        cls,
        *,
        n: int,
        grid: Grid,
    ) -> "StateHistory":
        shape = grid.shape

        return cls(
            steps=np.empty(n, dtype=np.int64),
            t=np.empty(n, dtype=np.float64),
            z=np.empty((n, *shape), dtype=np.float64),
            U=np.empty((n, *shape), dtype=np.float64),
            V=np.empty((n, *shape), dtype=np.float64),
        )

    def record(self, state: State, grid_type: GridType) -> None:
        if self._i >= len(self.steps):
            raise IndexError("Hey! The StateHistory is FULL! ;)")

        z, U, V = interpolate_grid(state, grid_type)

        self.steps[self._i] = state.step
        self.t[self._i] = state.t
        self.z[self._i] = z
        self.U[self._i] = U
        self.V[self._i] = V

        self._i += 1

    def save_npz(self, path: str) -> None:
        np.savez_compressed(
            path,
            steps=self.steps[: self._i],
            t=self.t[: self._i],
            z=self.z[: self._i],
            U=self.U[: self._i],
            V=self.V[: self._i],
        )


##################################################
# THE Shallow Water Equations Model!!            #
##################################################
@dataclass
class SWEModel:
    constants: Constants
    grid: Grid
    forcing: Forcing
    state: State

    dt: float
    linear: bool
    integration: IntegrationMethod
    grid_type: GridType

    def run(self, n_steps: int):
        for _ in range(n_steps):
            self.step()
            yield self.state.copy()

    def run_with_history(
        self,
        n_steps: int,
        *,
        save_every: int = 1,
    ) -> StateHistory:
        n = 1 + n_steps // save_every

        history = StateHistory.allocate(
            n=n,
            grid=self.grid,
        )
        history.record(self.state, self.grid_type)

        for state in self.run(n_steps):
            if state.step % save_every == 0:
                history.record(state, self.grid_type)

        return history

    def step(self) -> None:
        # Run numerical integration, using desired method
        if self.integration == IntegrationMethod.EULER:
            # Forward-time (Euler)
            dz, dU, dV = self.euler()
        elif self.integration == IntegrationMethod.RK4:
            # Runge-Kutta 4
            dz, dU, dV = self.rk4()
        else:
            raise NotImplementedError(self.integration)

        self.state.z += dz
        self.state.U += dU
        self.state.V += dV
        self.state.t += self.dt
        self.state.step += 1

        # For C grid we apply bottom drag term by division
        if self.grid_type == GridType.C_GRID:
            if not self.linear:
                # Compute bottom drag terms
                U, V = self.state.U, self.state.V

                V_on_U = np.zeros_like(U)
                V_on_U[:, 1:-1] = 0.25 * (
                    V[:-1, :-1] + V[1:, :-1] + V[:-1, 1:] + V[1:, 1:]
                )

                U_on_V = np.zeros_like(V)
                U_on_V[1:-1, :] = 0.25 * (
                    U[:-1, :-1] + U[:-1, 1:] + U[1:, :-1] + U[1:, 1:]
                )

                drag_u = (
                    1.0
                    + self.dt
                    * self.constants.r
                    * np.sqrt(U**2 + V_on_U**2)
                    / np.maximum(self.grid.H, 1e-3) ** 2
                )
                drag_v = (
                    1.0
                    + self.dt
                    * self.constants.r
                    * np.sqrt(U_on_V**2 + V**2)
                    / np.maximum(self.grid.H, 1e-3) ** 2
                )

                self.state.U /= drag_u
                self.state.V /= drag_v

            else:
                # Compute bottom drag terms
                drag = 1.0 + self.dt * self.constants.r

                self.state.U /= drag
                self.state.V /= drag

        # Apply closed lake boundary conditions
        z_mask, U_mask, V_mask = array_water_masks(self.grid, self.grid_type)
        self.state.z[~z_mask] = 0.0
        self.state.U[~U_mask] = 0.0
        self.state.V[~V_mask] = 0.0

    def euler(self) -> tuple[Array, Array, Array]:
        zt, Ut, Vt = self.rhs(self.state)

        dz = zt * self.dt
        dU = Ut * self.dt
        dV = Vt * self.dt

        return dz, dU, dV

    def rk4(self) -> tuple[Array, Array, Array]:
        state = self.state
        dt = self.dt
        half_dt = dt / 2.0

        # k1
        k1z, k1U, k1V = self.rhs(state)

        # k2
        k2z, k2U, k2V = self.rhs(
            State(
                state.z + k1z * half_dt,
                state.U + k1U * half_dt,
                state.V + k1V * half_dt,
                state.t + half_dt,
                state.step,
            )
        )

        # k3
        k3z, k3U, k3V = self.rhs(
            State(
                state.z + k2z * half_dt,
                state.U + k2U * half_dt,
                state.V + k2V * half_dt,
                state.t + half_dt,
                state.step,
            )
        )

        # k4
        k4z, k4U, k4V = self.rhs(
            State(
                state.z + k3z * dt,
                state.U + k3U * dt,
                state.V + k3V * dt,
                state.t + dt,
                state.step,
            )
        )

        # RK4 solution
        def rk4_step(k1: Array, k2: Array, k3: Array, k4: Array) -> Array:
            return (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        dz = rk4_step(k1z, k2z, k3z, k4z)
        dU = rk4_step(k1U, k2U, k3U, k4U)
        dV = rk4_step(k1V, k2V, k3V, k4V)

        return dz, dU, dV

    def rhs(self, state: State) -> tuple[Array, Array, Array]:
        """
        Compute the right hand side (RHS) of the linearized
        shallow water equations.
        """
        if self.grid_type == GridType.CENTERED:
            zt, Ut, Vt = self.rhs_centered(state)

        elif self.grid_type == GridType.C_GRID:
            zt, Ut, Vt = self.rhs_cgrid(state)

        else:
            raise NotImplementedError(self.grid_type)

        # Apply closed lake boundary conditions
        z_mask, U_mask, V_mask = array_water_masks(self.grid, self.grid_type)
        zt[~z_mask] = 0.0
        Ut[~U_mask] = 0.0
        Vt[~V_mask] = 0.0

        return zt, Ut, Vt

    def rhs_centered(self, state: State) -> tuple[Array, Array, Array]:
        """
        Compute the right hand side (RHS) for a centered grid.
        """
        H = self.grid.H

        z = state.z
        U = state.U
        V = state.V

        dx, dy = self.grid.dx, self.grid.dy

        # Compute zeta centered space gradients
        zx = np.zeros_like(z)
        zy = np.zeros_like(z)

        zx[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / (2.0 * dx)
        zy[1:-1, :] = (z[2:, :] - z[:-2, :]) / (2.0 * dy)

        zx[self.grid.land_mask] = 0.0
        zy[self.grid.land_mask] = 0.0

        # Compute U, V centered space gradients
        Ux = np.zeros_like(U)
        Vy = np.zeros_like(V)

        Ux[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / (2.0 * dx)
        Vy[1:-1, :] = (V[2:, :] - V[:-2, :]) / (2.0 * dy)

        Ux[self.grid.land_mask] = 0.0
        Vy[self.grid.land_mask] = 0.0

        # Compute pressure gradient terms
        pressure_u = self.constants.g * H * zx
        pressure_v = self.constants.g * H * zy

        # Compute bottom drag terms
        if self.linear:
            # Compute bottom drag terms
            drag_u = self.constants.r * U
            drag_v = self.constants.r * V
        else:
            speed_term = np.sqrt(U**2 + V**2) / np.maximum(self.grid.H, 1e-3) ** 2
            drag_u = self.constants.r * U * speed_term
            drag_v = self.constants.r * V * speed_term

        # Compute wind stress terms
        wind_u, wind_v = self.forcing.wind_stress(
            state.t,
            state.step,
            self.linear,
        )

        # Compute coriolis terms
        coriolis_u = self.constants.f * V
        coriolis_v = self.constants.f * U

        # Compute diffusion terms
        Ah = 1000.0
        dx, dy = self.grid.dx, self.grid.dy

        lap_U = np.zeros_like(U)
        lap_V = np.zeros_like(V)

        lap_U[1:-1, 1:-1] = (
            U[1:-1, 2:] - 2 * U[1:-1, 1:-1] + U[1:-1, :-2]
        ) / dx**2 + (U[2:, 1:-1] - 2 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / dy**2

        lap_V[1:-1, 1:-1] = (
            V[1:-1, 2:] - 2 * V[1:-1, 1:-1] + V[1:-1, :-2]
        ) / dx**2 + (V[2:, 1:-1] - 2 * V[1:-1, 1:-1] + V[:-2, 1:-1]) / dy**2

        # Compute RHS
        zt = -(Ux + Vy)
        Ut = -pressure_u + wind_u + coriolis_u - drag_u + Ah * lap_U
        Vt = -pressure_v + wind_v - coriolis_v - drag_v + Ah * lap_V

        return zt, Ut, Vt

    def rhs_cgrid(self, state: State) -> tuple[Array, Array, Array]:
        """
        Compute the right hand side (RHS) for a C grid.
        """
        H = self.grid.H

        z = state.z
        U = state.U
        V = state.V

        dx, dy = self.grid.dx, self.grid.dy

        # Compute zeta spatial gradients
        zx = np.zeros_like(U)
        zy = np.zeros_like(V)

        zx[:, 1:-1] = (z[:, 1:] - z[:, :-1]) / dx
        zy[1:-1, :] = (z[1:, :] - z[:-1, :]) / dy

        # Compute U, V spatial gradients
        Ux = (U[:, 1:] - U[:, :-1]) / dx
        Vy = (V[1:, :] - V[:-1, :]) / dy

        # Compute pressure gradient terms
        HU = np.zeros_like(U)
        HV = np.zeros_like(V)

        HU[:, 1:-1] = 0.5 * (H[:, 1:] + H[:, :-1])
        HV[1:-1, :] = 0.5 * (H[1:, :] + H[:-1, :])

        _, U_mask, V_mask = array_water_masks(self.grid, self.grid_type)
        HU[~U_mask] = 0.0
        HV[~V_mask] = 0.0

        pressure_u = self.constants.g * HU * zx
        pressure_v = self.constants.g * HV * zy

        # Compute wind stress terms
        wind_u, wind_v = self.forcing.wind_stress(
            state.t,
            state.step,
            self.linear,
        )

        # Compute coriolis terms
        V_on_U = np.zeros_like(U)
        V_on_U[:, 1:-1] = 0.25 * (V[:-1, :-1] + V[1:, :-1] + V[:-1, 1:] + V[1:, 1:])

        U_on_V = np.zeros_like(V)
        U_on_V[1:-1, :] = 0.25 * (U[:-1, :-1] + U[:-1, 1:] + U[1:, :-1] + U[1:, 1:])

        coriolis_u = self.constants.f * V_on_U
        coriolis_v = self.constants.f * U_on_V

        # Compute RHS
        zt = -(Ux + Vy)
        Ut = -pressure_u + wind_u + coriolis_u
        Vt = -pressure_v + wind_v - coriolis_v

        return zt, Ut, Vt

    def grad_xy(self, a: Array) -> tuple[Array, Array]:
        return self.grad_x(a), self.grad_y(a)

    def grad_x(self, a: Array) -> Array:
        dx = self.grid.dx

        ax = np.zeros_like(a)
        ax[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / (2.0 * dx)
        ax[self.grid.land_mask] = 0.0

        return ax

    def grad_y(self, a: Array) -> Array:
        dy = self.grid.dy

        ay = np.zeros_like(a)
        ay[1:-1, :] = (a[2:, :] - a[:-2, :]) / (2.0 * dy)
        ay[self.grid.land_mask] = 0.0

        return ay
