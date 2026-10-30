import json
import sys
from pathlib import Path

import numpy as np
import pyshtools
import plotly.graph_objects as go
from joblib import Parallel, delayed
from ai import cs
from matplotlib import pyplot as plt
from scipy.integrate import RK45

from coronal_diffusion import utils
import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INNER_BOUNDARY = 1.00
OUTER_BOUNDARY = 2.50


class SHInterpolator:
    def __init__(self, real, imag):
        real_coeffs = utils.complex_to_real(real, imag)
        
        # transforms (csphase=False) and the physical WSA field.
        self.coeffs = pyshtools.SHMagCoeffs.from_array(
            real_coeffs,
            normalization="ortho",
            r0=1,
        )

    def interpolate(self, r, lat, lon):
        result = np.array(
            self.coeffs.expand(r=[r], lat=[lat], lon=[lon], degrees=False)
        ).flatten()
        Br, Btheta, Bphi = result

        return Br, Btheta, Bphi


class Visualizer:

    def __init__(self, real=None, imag=None):
        if real is not None and imag is not None:
            self.itp = SHInterpolator(real, imag)
        else:
            raise ValueError("Need to specify real and imag")

    def trace_field_line(
        self, start_point, step_size, max_steps=100_000_000, closed_only=False
    ):
        field_line = [start_point]
        color = "black"

        for _ in range(max_steps):
            r, lat, lon = cs.cart2sp(*field_line[-1])

            # Expand spherical harmonics to get field components
            try:
                Br, Btheta, Bphi = self.itp.interpolate(r, lat, lon)
            except ValueError:
                break

            # Convert spherical field components to Cartesian coordinates
            Bx, By, Bz = spherical_to_cartesian_vector(Br, Btheta, Bphi, lat, lon)

            Bnorm = np.sqrt(Bx**2 + By**2 + Bz**2)
            Bx /= Bnorm
            By /= Bnorm
            Bz /= Bnorm

            x = field_line[-1][0] + Bx * step_size
            y = field_line[-1][1] + By * step_size
            z = field_line[-1][2] + Bz * step_size
            field_line.append((x, y, z))

            r = np.sqrt(x**2 + y**2 + z**2)
            if r > OUTER_BOUNDARY:
                color = "blue" if Br > 0 else "red"
                break
            if r < INNER_BOUNDARY:
                break

        return field_line, color

    def _get_field_lines_on_grid(self, r, grid_density, closed_only):
        step_size = 0.01  # Step size
        max_steps = 1000  # Maximum number of steps to trace

        lat_values = np.linspace(-89.9, 89.9, grid_density // 2, endpoint=False)
        lon_values = np.linspace(0, 360, grid_density, endpoint=False)

        field_lines = []
        colors = []
        for lat_deg in lat_values:
            for lon_deg in lon_values:
                for dir in [-1, 1]:
                    start_point = cs.sp2cart(
                        r, np.deg2rad(lat_deg), np.deg2rad(lon_deg)
                    )
                    field_line, color = self.trace_field_line(
                        start_point, dir * step_size, max_steps, closed_only=closed_only
                    )

                    if field_line is not None:
                        field_lines.append(np.array(field_line))
                        colors.append(color)

        return field_lines, colors

    def get_coronal_holes(self, grid_density, r=1.1):
        step_size = 0.01  # Step size
        max_steps = 100000  # Maximum number of steps to trace

        lat_values = np.linspace(-89, 89, grid_density // 2, endpoint=False)
        lon_values = np.linspace(0, 360, grid_density, endpoint=False)

        open_fl = np.zeros((lat_values.size, lon_values.size))

        for i, lat_deg in enumerate(lat_values):
            for j, lon_deg in enumerate(lon_values):
                for dir in [-1, 1]:
                    start_point = cs.sp2cart(
                        r, np.deg2rad(lat_deg), np.deg2rad(lon_deg)
                    )
                    field_line, color = self.trace_field_line(
                        start_point, dir * step_size, max_steps, closed_only=True
                    )

                    last_r, last_lat, last_lon = cs.cart2sp(*field_line[-1])

                    if last_r > 2.49:
                        open_fl[i, j] = 1

        return lat_values, lon_values, open_fl

    def plot_coronal_holes(self, grid_density=10, r=1.01):
        lat_values, lon_values, ch_map = self.get_coronal_holes(grid_density, r=r)
        plt.figure(figsize=(10, 5))
        plt.pcolor(lon_values, lat_values, ch_map)
        plt.xlabel('Carrington Longitude')
        plt.ylabel('Latitude')
        plt.gca().set_aspect('equal')
        cbar = plt.colorbar()
        cbar.set_ticks([0, 1])
        cbar.set_label("Coronal Hole")

    def visualize_field_lines(
        self, r=1.1, grid_density=10, closed_only=False, lim=OUTER_BOUNDARY
    ):
        field_lines, colors = self._get_field_lines_on_grid(
            r,
            grid_density,
            closed_only=closed_only,
        )

        fig = self.plot(field_lines, colors, lim)
        fig.update_layout(showlegend=False)

        return fig

    def plot(self, field_lines, colors, lim):
        fig = go.Figure()

        # Add field lines
        for field_line, color in zip(field_lines, colors):
            fig.add_trace(
                go.Scatter3d(
                    x=field_line[:, 0],
                    y=field_line[:, 1],
                    z=field_line[:, 2],
                    mode="lines",
                    line=dict(width=2, color=color),
                )
            )

        # Add yellow sphere of radius 1
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                colorscale=[[0, "yellow"], [1, "yellow"]],
                opacity=1,
                showscale=False,
            )
        )

        fig.update_layout(
            scene=dict(
                aspectmode="cube",  # Ensures equal aspect ratio
                xaxis=dict(range=[-lim, lim]),
                yaxis=dict(range=[-lim, lim]),
                zaxis=dict(range=[-lim, lim]),
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
            ),
            margin=dict(l=0, r=0, b=0, t=0),
        )
        return fig

    def plot_magnetogram(self, r=1.05, lim=2.5, sign=False, ax=None):
        result, lat, lon = self.get_magnetogram(r)
        Br = result[:, 0].reshape(lat.shape) * 1e-5
        if sign:
            Br = np.sign(Br)
            lim = 1

        if not ax:
            plt.figure(figsize=(10, 5))
            ax = plt.gca()

        ax.pcolor(lon, lat, Br, cmap="Grays", vmax=lim, vmin=-lim)

        if sign and not ax:
            plt.title(f"Polarity at r={r} $R_\\odot$", fontsize=18)
            plt.colorbar().set_label("Polarity")
        elif not ax:
            plt.colorbar().set_label("Br (T)")
            plt.title(f"$B_r$ at r={r} $R_\\odot$", fontsize=18)

    def plot_current_sheet(self, r=2.5, ax=None):
        self.plot_magnetogram(r=r, sign=True, ax=ax)

    def get_magnetogram(self, r):
        # Get magnetogram data
        lat_axis = np.arange(-89, 90, 1)
        lon_axis = np.arange(0, 360, 1)

        lat, lon = np.meshgrid(lat_axis, lon_axis, indexing="ij")

        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        result = np.zeros((lat.size, 3))

        for i in range(lat.size):
            result[i, :] = self.itp.interpolate(
                r, lat_rad.flatten()[i], lon_rad.flatten()[i]
            )

        return result, lat, lon


class VisualizerRK45(Visualizer):

    def __init__(self, real=None, imag=None, n_jobs=-1):
        super().__init__(real=real, imag=imag)
        self.real = np.array(real, copy=True)
        self.imag = np.array(imag, copy=True)
        self.n_jobs = n_jobs

    def trace_field_line(
        self, start_point, step_size, max_steps=100_000_000, closed_only=False
    ):
        del closed_only

        start_point = np.asarray(start_point, dtype=float)
        field_line = [tuple(start_point)]
        color = "black"

        step_size = float(step_size)
        if step_size == 0:
            return field_line, color

        direction = 1.0 if step_size > 0 else -1.0
        max_step = abs(step_size)
        solver = RK45(
            self._field_line_rhs(direction),
            0.0,
            start_point,
            max_steps * max_step,
            max_step=max_step,
        )

        for _ in range(max_steps):
            r, lat, lon = cs.cart2sp(*solver.y)

            try:
                Br, _, _ = self.itp.interpolate(r, lat, lon)
            except ValueError:
                break

            try:
                solver.step()
            except ValueError:
                break

            field_line.append(tuple(solver.y))

            r = np.linalg.norm(solver.y)
            if r > OUTER_BOUNDARY:
                color = "blue" if Br > 0 else "red"
                break
            if r < INNER_BOUNDARY:
                break

            if solver.status != "running":
                break

        return field_line, color

    def _field_line_rhs(self, direction):
        def rhs(_, point):
            r, lat, lon = cs.cart2sp(*point)

            Br, Btheta, Bphi = self.itp.interpolate(r, lat, lon)
            Bx, By, Bz = spherical_to_cartesian_vector(Br, Btheta, Bphi, lat, lon)
            Bnorm = np.sqrt(Bx**2 + By**2 + Bz**2)

            if not np.isfinite(Bnorm) or Bnorm == 0:
                return np.zeros(3)

            return direction * np.array([Bx, By, Bz]) / Bnorm

        return rhs

    def _get_field_lines_on_grid(self, r, grid_density, closed_only):
        step_size = 0.01
        max_steps = 1000

        lat_values = np.linspace(-89.9, 89.9, grid_density // 2, endpoint=False)
        lon_values = np.linspace(0, 360, grid_density, endpoint=False)

        seeds = []
        for lat_deg in lat_values:
            for lon_deg in lon_values:
                for direction in (-1, 1):
                    start_point = cs.sp2cart(
                        r, np.deg2rad(lat_deg), np.deg2rad(lon_deg)
                    )
                    seeds.append((start_point, direction * step_size, max_steps, closed_only))

        results = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
            delayed(_trace_field_line_rk45)(
                start_point,
                step_size,
                max_steps,
                self.real,
                self.imag,
                closed_only,
            )
            for start_point, step_size, max_steps, closed_only in seeds
        )

        field_lines = []
        colors = []
        for field_line, color in results:
            if field_line is not None:
                field_lines.append(np.array(field_line))
                colors.append(color)

        return field_lines, colors

    def get_coronal_holes(self, grid_density, r=1.1):
        step_size = 0.01
        max_steps = 100000

        lat_values = np.linspace(-88, 88, grid_density // 2, endpoint=False)
        lon_values = np.linspace(0, 360, grid_density, endpoint=False)

        seeds = []
        for i, lat_deg in enumerate(lat_values):
            for j, lon_deg in enumerate(lon_values):
                for direction in (-1, 1):
                    start_point = cs.sp2cart(
                        r, np.deg2rad(lat_deg), np.deg2rad(lon_deg)
                    )
                    seeds.append((i, j, start_point, direction * step_size, max_steps))

        results = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
            delayed(_trace_field_line_rk45)(
                start_point,
                step_size,
                max_steps,
                self.real,
                self.imag,
                True,
            )
            for _, _, start_point, step_size, max_steps in seeds
        )

        open_fl = np.zeros((lat_values.size, lon_values.size))
        for (i, j, _, _, _), (field_line, _) in zip(seeds, results):
            last_r, _, _ = cs.cart2sp(*field_line[-1])
            if last_r > 2.49:
                open_fl[i, j] = 1

        return lat_values, lon_values, open_fl


def spherical_to_cartesian_vector(Br, Btheta, Bphi, lat, lon):
    colat = np.pi / 2 - lat

    sinθ = np.sin(colat)
    cosθ = np.cos(colat)
    sinφ = np.sin(lon)
    cosφ = np.cos(lon)

    Bx = Br * sinθ * cosφ + Btheta * cosθ * cosφ - Bphi * sinφ
    By = Br * sinθ * sinφ + Btheta * cosθ * sinφ + Bphi * cosφ
    Bz = Br * cosθ - Btheta * sinθ

    return Bx, By, Bz


def _trace_field_line_rk45(
    start_point, step_size, max_steps, real, imag, closed_only=False
):
    del closed_only

    itp = SHInterpolator(real, imag)

    start_point = np.asarray(start_point, dtype=float)
    field_line = [tuple(start_point)]
    color = "black"

    step_size = float(step_size)
    if step_size == 0:
        return field_line, color

    direction = 1.0 if step_size > 0 else -1.0
    max_step = abs(step_size)
    solver = RK45(
        _field_line_rhs_rk45(itp, direction),
        0.0,
        start_point,
        max_steps * max_step,
        max_step=max_step,
    )

    for _ in range(max_steps):
        r, lat, lon = cs.cart2sp(*solver.y)

        try:
            Br, _, _ = itp.interpolate(r, lat, lon)
        except ValueError:
            break

        try:
            solver.step()
        except ValueError:
            break

        field_line.append(tuple(solver.y))

        r = np.linalg.norm(solver.y)
        if r > OUTER_BOUNDARY:
            color = "blue" if Br > 0 else "red"
            break
        if r < INNER_BOUNDARY:
            break

        if solver.status != "running":
            break

    return field_line, color


def _field_line_rhs_rk45(itp, direction):
    def rhs(_, point):
        r, lat, lon = cs.cart2sp(*point)

        Br, Btheta, Bphi = itp.interpolate(r, lat, lon)
        Bx, By, Bz = spherical_to_cartesian_vector(Br, Btheta, Bphi, lat, lon)
        Bnorm = np.sqrt(Bx**2 + By**2 + Bz**2)

        if not np.isfinite(Bnorm) or Bnorm == 0:
            return np.zeros(3)

        return direction * np.array([Bx, By, Bz]) / Bnorm

    return rhs
