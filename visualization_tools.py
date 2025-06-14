import numpy as np
import pyshtools
import plotly.graph_objects as go
from ai import cs

OUTER_BOUNDARY = 2.5


class SHVisualizer:

    def __init__(self, G, H):
        coeffs_array = np.array([G, H])
        self.coeffs = pyshtools.SHMagCoeffs.from_array(
            coeffs_array, normalization="schmidt", r0=1
        )

    def trace_field_line(self, start_point, step_size, max_steps, closed_only=False):
        field_line = [start_point]

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
                    return None
                else:
                    break
            if r < 1:
                break

            field_line.append((x, y, z))

        return field_line

    def _get_field_lines_on_grid(self, grid_density, closed_only):
        step_size = 0.01  # Step size
        max_steps = 1000  # Maximum number of steps to trace

        r = 1.1
        lat_values = np.linspace(-89, 89, grid_density)
        lon_values = np.linspace(0, 360, grid_density)

        field_lines = []
        for lat_deg in lat_values:
            for lon_deg in lon_values:
                for dir in [-1, 1]:
                    start_point = cs.sp2cart(
                        r, np.deg2rad(lat_deg), np.deg2rad(lon_deg)
                    )
                    field_line = self.trace_field_line(
                        start_point, dir * step_size, max_steps, closed_only=closed_only
                    )

                    if field_line is not None:
                        field_lines.append(np.array(field_line))

        return field_lines

    def visualize_field_lines(
        self, lim=OUTER_BOUNDARY, grid_density=10, closed_only=False
    ):
        field_lines = self._get_field_lines_on_grid(
            grid_density, closed_only=closed_only
        )

        fig = go.Figure()

        # Add field lines
        for field_line in field_lines:
            fig.add_trace(
                go.Scatter3d(
                    x=field_line[:, 0],
                    y=field_line[:, 1],
                    z=field_line[:, 2],
                    mode="lines",
                    line=dict(width=2, color="black"),
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
        fig.show()


def spherical_to_cartesian_vector(Br, Btheta, Bphi, theta_deg, phi_deg):
    theta = np.pi / 2 - np.radians(theta_deg)
    phi = np.radians(phi_deg)

    sinθ = np.sin(theta)
    cosθ = np.cos(theta)
    sinφ = np.sin(phi)
    cosφ = np.cos(phi)

    Bx = Br * sinθ * cosφ + Btheta * cosθ * cosφ - Bphi * sinφ
    By = Br * sinθ * sinφ + Btheta * cosθ * sinφ + Bphi * cosφ
    Bz = Br * cosθ - Btheta * sinθ

    return Bx, By, Bz
