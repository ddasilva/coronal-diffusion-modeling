import numpy as np
import pyshtools
import plotly.graph_objects as go
from ai import cs
from matplotlib import pyplot as plt

OUTER_BOUNDARY = 2.5


class SHVisualizer:

    def __init__(self, G, H, normalization="ortho"):
        coeffs_array = np.array([G, H])
        self.coeffs = pyshtools.SHMagCoeffs.from_array(
            coeffs_array, normalization=normalization, r0=1,
        )

    def trace_field_line(self, start_point, step_size, max_steps=100_000_000, closed_only=False):
        field_line = [start_point]
        color = "black"

        for _ in range(max_steps):
            r, lat, lon = cs.cart2sp(*field_line[-1])

            # Expand spherical harmonics to get field components
            result = np.array(
                self.coeffs.expand(r=[r], lat=[lat], lon=[lon], degrees=False)
            ).flatten()
            Br, Btheta, Bphi = result

            # Convert spherical field components to Cartesian coordinates
            Bx, By, Bz = spherical_to_cartesian_vector(Br, Btheta, Bphi, lat, lon)

            Bnorm = np.sqrt(Bx**2 + By**2 + Bz**2)
            Bx /= Bnorm
            By /= Bnorm
            Bz /= Bnorm

            x = field_line[-1][0] + Bx * step_size
            y = field_line[-1][1] + By * step_size
            z = field_line[-1][2] + Bz * step_size

            r = np.sqrt(x**2 + y**2 + z**2)
            if r > OUTER_BOUNDARY:
                if closed_only:
                    return None, None
                else:
                    color = "blue" if Br > 0 else "red"
                    break
            if r < 1:
                break

            field_line.append((x, y, z))

        return field_line, color

    def _get_field_lines_on_grid(self, r, grid_density, closed_only):
        step_size = 0.01  # Step size
        max_steps = 1000  # Maximum number of steps to trace

        lat_values = np.linspace(-89.9, 89.9, grid_density, endpoint=False)
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
        max_steps = 1000  # Maximum number of steps to trace

        lat_values = np.linspace(-89, 89, grid_density)
        lon_values = np.linspace(0, 359, grid_density)

        ch_map = np.ones((lat_values.size, lon_values.size), dtype=bool)
        
        for i, lat_deg in enumerate(lat_values):
            for j, lon_deg in enumerate(lon_values):
                for dir in [-1, 1]:
                    start_point = cs.sp2cart(
                        r, np.deg2rad(lat_deg), np.deg2rad(lon_deg)
                    )
                    field_line, color = self.trace_field_line(
                        start_point, dir * step_size, max_steps, closed_only=True
                    )

                    if field_line is None:
                        ch_map[i, j] = 0

        return lat_values, lon_values, ch_map

    def plot_coronal_holes(self, grid_density=10):
        lat_values, lon_values, ch_map = self.get_coronal_holes(grid_density)
        plt.pcolor(lon_values, lat_values, ch_map)
        plt.colorbar().set_label('Coronal Hole')
        
    
    def visualize_field_lines(
        self, r=1.1, grid_density=10, closed_only=False, lim=OUTER_BOUNDARY
    ):
        field_lines, colors = self._get_field_lines_on_grid(
            r, grid_density, closed_only=closed_only,
        )

        fig = self.plot(field_lines, colors, lim)
    
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

    def plot_magnetogram(self, r=1.05, lim=2.5, sign=False):
        result, lat, lon = self.get_magnetogram(r)
        Br = result[:, 0].reshape(lat.shape) * 1e-5
        if sign:
            Br = np.sign(Br)
            lim = 1
        
        plt.figure(figsize=(10, 5))
        plt.pcolor(lon, lat, Br, cmap="Grays", vmax=lim, vmin=-lim)

        if sign:
            plt.title(f'Polarity at r={r} $R_\\odot$', fontsize=18)
            plt.colorbar().set_label('Polarity')
        else:
            plt.colorbar().set_label('Br (T)')
            plt.title(f'$B_r$ at r={r} $R_\\odot$', fontsize=18)

    def plot_current_sheet(self, r=2.5):
        self.plot_magnetogram(r=r, sign=True)
        
        
    
    def get_magnetogram(self, r):
        # Get magnetogram data
        lat_axis = np.arange(-89, 90, 1)
        lon_axis = np.arange(0, 360, 1)

        lat, lon = np.meshgrid(lat_axis, lon_axis, indexing="ij")

        result = np.array(
            self.coeffs.expand(r=[r] * lat.size, lat=lat.flatten(), lon=lon.flatten(), degrees=True)
        ).squeeze()
 
        return result, lat, lon


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
