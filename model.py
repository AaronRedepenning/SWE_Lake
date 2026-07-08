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


##################################################
# Integration methods                            #
##################################################
class Method(StrEnum):
    ANDY_METHOD = "Andy's Method"
    AARON_METHOD = "Aaron's Method"


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

    def wind_stress(self, t: float, step: int, method: Method) -> Vector:
        wx, wy = self.wind(t=t, step=step)
        wsf = self.constants.wsf

        if method == Method.ANDY_METHOD:
            wind_speed = math.sqrt(wx**2 + wy**2)

            return (
                wsf * wx * wind_speed,
                wsf * wy * wind_speed,
            )
        elif method == Method.AARON_METHOD:
            wind_speed = math.sqrt(wx**2 + wy**2)

            return (
                wsf * wx * wind_speed,
                wsf * wy * wind_speed,
            )
        else:
            raise NotImplementedError(f"No method == {self.method}!")


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
    def init_zeros(cls, grid: Grid) -> "State":
        shape = grid.shape
        return cls(
            z=np.zeros(shape, dtype=np.float64),
            U=np.zeros(shape, dtype=np.float64),
            V=np.zeros(shape, dtype=np.float64),
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
    def allocate(cls, *, n: int, shape: tuple[int, int]) -> "StateHistory":
        return cls(
            steps=np.empty(n, dtype=np.int64),
            t=np.empty(n, dtype=np.float64),
            z=np.empty((n, *shape), dtype=np.float64),
            U=np.empty((n, *shape), dtype=np.float64),
            V=np.empty((n, *shape), dtype=np.float64),
        )

    def record(self, state: State) -> None:
        if self._i >= len(self.steps):
            raise IndexError("Hey! The StateHistory is FULL! ;)")

        self.steps[self._i] = state.step
        self.t[self._i] = state.t
        self.z[self._i] = state.z
        self.U[self._i] = state.U
        self.V[self._i] = state.V

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
    method: Method

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
            shape=self.grid.shape,
        )
        history.record(self.state)

        for state in self.run(n_steps):
            if state.step % save_every == 0:
                history.record(state)

        return history

    def step(self) -> None:
        # Run numerical integration, using desired method
        if self.method == Method.ANDY_METHOD:
            # Forward-time (Euler)
            dz, dU, dV = self.euler()
        elif self.method == Method.AARON_METHOD:
            # Runge-Kutta 4
            dz, dU, dV = self.rk4()
        else:
            raise NotImplementedError(f"No method == {self.method}!")

        self.state.z += dz
        self.state.U += dU
        self.state.V += dV
        self.state.t += self.dt
        self.state.step += 1

        # Apply closed lake boundary conditions
        land_mask = self.grid.land_mask
        self.state.z[land_mask] = 0.0
        self.state.U[land_mask] = 0.0
        self.state.V[land_mask] = 0.0

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
        H = self.grid.H

        z = state.z
        U = state.U
        V = state.V

        # Compute zeta spatial gradients
        zx, zy = self.grad_xy(z)

        # Compute U, V spatial gradients
        Ux = self.grad_x(U)
        Vy = self.grad_y(V)

        # Compute pressure gradient terms
        pressure_u = self.constants.g * H * zx
        pressure_v = self.constants.g * H * zy

        # Compute bottom drag terms
        if self.method == Method.ANDY_METHOD:
            # Compute bottom drag terms
            drag_u = self.constants.r * U
            drag_v = self.constants.r * V
        elif self.method == Method.AARON_METHOD:
            speed_term = np.sqrt(U**2 + V**2) / np.maximum(self.grid.H, 1e-3) ** 2
            drag_u = self.constants.r * U * speed_term
            drag_v = self.constants.r * V * speed_term
        else:
            raise NotImplementedError(f"No method == {self.method}!")

        # Compute wind stress terms
        wind_u, wind_v = self.forcing.wind_stress(
            t=state.t,
            step=state.step,
            method=self.method,
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

        # Mask land
        land_mask = self.grid.land_mask
        zt[land_mask] = 0.0
        Ut[land_mask] = 0.0
        Vt[land_mask] = 0.0

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
