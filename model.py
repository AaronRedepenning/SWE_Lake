import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


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
@dataclass(frozen=True)
class Forcing:
    # Wind
    wx: float
    wy: float

    constants: Constants

    @property
    def wind_speed(self) -> float:
        return math.sqrt(self.wx**2 + self.wy**2)

    @property
    def wind_stress(self) -> tuple[float, float]:
        wsf = self.constants.wsf
        wind_speed = self.wind_speed

        return (
            wsf * self.wx * wind_speed,
            wsf * self.wy * wind_speed,
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

    @classmethod
    def init_zeros(cls, grid: Grid) -> "State":
        shape = grid.shape
        return cls(
            z=np.zeros(shape, dtype=np.float64),
            U=np.zeros(shape, dtype=np.float64),
            V=np.zeros(shape, dtype=np.float64),
            t=0.0,
        )

    def copy(self) -> "State":
        return State(
            z=self.z.copy(),
            U=self.U.copy(),
            V=self.V.copy(),
            t=self.t,
        )


##################################################
# THE Shallow Water Equations Model!!            #
##################################################
class Method(StrEnum):
    EULER = "Euler"
    RK4 = "Runge-Kutta 4"


@dataclass
class SWEModel:
    constants: Constants
    grid: Grid
    forcing: Forcing
    state: State

    dt: float
    method: Method

    def run(self, n_steps: int):
        for step in range(1, n_steps + 1):
            self.step()
            yield step, self.state.copy()

    def step(self) -> None:
        # Run numerical integration, using desired method
        if self.method == Method.EULER:
            dz, dU, dV = self.euler()
        elif self.method == Method.RK4:
            dz, dU, dV = self.rk4()

        self.state.z += dz
        self.state.U += dU
        self.state.V += dV
        self.state.t += self.dt

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
        s1 = self.state
        dt = self.dt
        half_dt = dt / 2.0

        # k1
        k1z, k1U, k1V = self.rhs(s1)

        # k2
        s2 = State(
            s1.z + k1z * half_dt,
            s1.U + k1U * half_dt,
            s1.V + k1V * half_dt,
            s1.t + half_dt,
        )
        k2z, k2U, k2V = self.rhs(s2)

        # k3
        s3 = State(
            s2.z + k2z * half_dt,
            s2.U + k2U * half_dt,
            s2.V + k2V * half_dt,
            s2.t + half_dt,
        )
        k3z, k3U, k3V = self.rhs(s3)

        # k4
        s4 = State(
            s3.z + k2z * dt,
            s3.U + k2U * dt,
            s3.V + k2V * dt,
            s3.t + dt,
        )
        k4z, k4U, k4V = self.rhs(s4)

        # RK4 solution
        def rk4_solve(k1: Array, k2: Array, k3: Array, k4: Array) -> Array:
            return (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        dz = rk4_solve(k1z, k2z, k3z, k4z)
        dU = rk4_solve(k1U, k2U, k3U, k4U)
        dV = rk4_solve(k1V, k2V, k3V, k4V)

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

        # Compute spatial gradients
        zx = self.grad_x(z)
        zy = self.grad_y(z)

        # Compute derivatives of transport compoenent fluxes
        Ux, Vy = self.flux_derivatives(U, V)

        # Compute pressure gradient terms
        pressure_u = self.constants.g * H * zx
        pressure_v = self.constants.g * H * zy

        # Compute bottom drag terms
        speed_term = np.sqrt(U**2 + V**2) / np.maximum(self.grid.H, 1e-3) ** 2
        drag_u = self.constants.r * U * speed_term
        drag_v = self.constants.r * V * speed_term

        # Compute wind stress terms
        wind_u, wind_v = self.forcing.wind_stress

        # Compute coriolis terms
        coriolis_u = self.constants.f * V
        coriolis_v = self.constants.f * U

        # Compute RHS
        zt = -(Ux + Vy)
        Ut = -pressure_u - drag_u + wind_u + coriolis_u
        Vt = -pressure_v - drag_v + wind_v - coriolis_v

        # Mask land
        land_mask = self.grid.land_mask
        zt[land_mask] = 0.0
        Ut[land_mask] = 0.0
        Vt[land_mask] = 0.0

        return zt, Ut, Vt

    def grad_x(self, a: Array) -> Array:
        dx = self.grid.dx
        water_mask = self.grid.water_mask

        water_left = np.zeros_like(a, dtype=bool)
        water_right = np.zeros_like(a, dtype=bool)
        water_left[:, 1:] = water_mask[:, :-1]
        water_right[:, :-1] = water_mask[:, 1:]

        both = water_mask & water_left & water_right
        left_only = water_mask & water_left & ~water_right
        right_only = water_mask & ~water_left & water_right

        a_left = np.zeros_like(a)
        a_right = np.zeros_like(a)
        a_left[:, 1:] = a[:, :-1]
        a_right[:, :-1] = a[:, 1:]

        ax = np.zeros_like(a)
        ax[both] = (a_right[both] - a_left[both]) / (2.0 * dx)  # Centered
        ax[left_only] = (a[left_only] - a_left[left_only]) / dx  # Forward
        ax[right_only] = (a_right[right_only] - a[right_only]) / dx  # Backward

        return ax

    def grad_y(self, a: Array) -> Array:
        dy = self.grid.dy
        water_mask = self.grid.water_mask

        water_below = np.zeros_like(a, dtype=bool)
        water_above = np.zeros_like(a, dtype=bool)
        water_below[:-1, :] = water_mask[1:, :]
        water_above[1:, :] = water_mask[:-1, :]

        both = water_mask & water_above & water_below
        below_only = water_mask & ~water_above & water_below
        above_only = water_mask & water_above & ~water_below

        a_below = np.zeros_like(a)
        a_above = np.zeros_like(a)
        a_below[:-1, :] = a[1:, :]
        a_above[1:, :] = a[:-1, :]

        ay = np.zeros_like(a)
        ay[both] = (a_above[both] - a_below[both]) / (2.0 * dy)  # Centered
        ay[below_only] = (a[below_only] - a_below[below_only]) / dy  # Forward
        ay[above_only] = (a_above[above_only] - a[above_only]) / dy  # Backward

        return ay

    def flux_derivatives(self, U: Array, V: Array) -> tuple[Array, Array]:
        dx, dy = self.grid.dx, self.grid.dy
        ny, nx = self.grid.shape
        land_mask = self.grid.land_mask
        water_mask = self.grid.water_mask

        # Grid faces are only open if both sides have water
        open_U = water_mask[:, :-1] & water_mask[:, 1:]
        open_V = water_mask[:-1, :] & water_mask[1:, :]

        # Compute face fluxes by averaging transport components across the grid face
        U_face = np.zeros((ny, nx + 1), dtype=np.float64)
        V_face = np.zeros((ny + 1, nx), dtype=np.float64)

        U_face[:, 1:-1][open_U] = 0.5 * (U[:, :-1][open_U] + U[:, 1:][open_U])
        V_face[1:-1, :][open_V] = 0.5 * (V[:-1, :][open_V] + V[1:, :][open_V])

        # Compute flux derivatives
        Ux = (U_face[:, 1:] - U_face[:, :-1]) / dx
        Vy = (V_face[:-1, :] - V_face[1:, :]) / dy
        Ux[land_mask] = 0.0
        Vy[land_mask] = 0.0

        return Ux, Vy
