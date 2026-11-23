
"""Interactive 3D visualization of strawberry positions in the frame."""

from pathlib import Path

import plotly.graph_objects as go

from harvestai_perception.geometry.frame_transform import DEFAULT_FRAME, inside_working_volume


def make_3d_plot(points):
    cfg = DEFAULT_FRAME

    hx = cfg.inner_half_width
    hy = cfg.inner_half_length
    h = cfg.height

    # Plot area = working volume plus a margin on every side, so berries just
    # outside the frame are still visible (drawn as diamonds, labelled OUT).
    margin = 0.06
    x_lo, x_hi = -hx - margin, hx + margin
    y_lo, y_hi = -hy - margin, hy + margin
    z_lo, z_hi = -margin, h + margin

    fig = go.Figure()

    # Draw the working-volume wireframe (16 x 16 x 44 cm by default).
    xs, ys, zs = [], [], []

    def add_edge(a, b):
        xs.extend([a[0], b[0], None])
        ys.extend([a[1], b[1], None])
        zs.extend([a[2], b[2], None])

    for z in (0, h):
        for y in (-hy, hy):
            add_edge((-hx, y, z), (hx, y, z))
        for x in (-hx, hx):
            add_edge((x, -hy, z), (x, hy, z))

    for x in (-hx, hx):
        for y in (-hy, hy):
            add_edge((x, y, 0), (x, y, h))

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color="#aaaaaa", width=4),
        name="Working volume",
        hoverinfo="skip",
    ))

    # Draw the outer visualization boundary (working volume + margin).
    outer_xs, outer_ys, outer_zs = [], [], []

    def add_outer_edge(a, b):
        outer_xs.extend([a[0], b[0], None])
        outer_ys.extend([a[1], b[1], None])
        outer_zs.extend([a[2], b[2], None])

    # Top and bottom faces
    for z in (z_lo, z_hi):
        for y in (y_lo, y_hi):
            add_outer_edge((x_lo, y, z), (x_hi, y, z))
        for x in (x_lo, x_hi):
            add_outer_edge((x, y_lo, z), (x, y_hi, z))

    # Four vertical edges
    for x in (x_lo, x_hi):
        for y in (y_lo, y_hi):
            add_outer_edge((x, y, z_lo), (x, y, z_hi))

    fig.add_trace(go.Scatter3d(
        x=outer_xs,
        y=outer_ys,
        z=outer_zs,
        mode="lines",
        line=dict(
            color="rgba(80, 150, 255, 0.5)",
            width=3,
        ),
        name="Visualization boundary",
        hoverinfo="skip",
    ))

    # Direction labels near the top of the outer cube
    dx = x_hi - 0.025
    dy = y_hi - 0.025
    label_z = z_hi - 0.015

    fig.add_trace(go.Scatter3d(
        x=[-dx, dx, 0, 0],
        y=[0, 0, dy, -dy],
        z=[label_z] * 4,

        mode="text",
        text=[
            "L (-X)",
            "R (+X)",
            "F (+Y)",
            "B (-Y)",
        ],

        textfont=dict(
            size=12,
            color="#FFD166",
        ),

        showlegend=False,
        hoverinfo="skip",
    ))

    # Add strawberry positions.
    colors = {
        "ripe": "red",
        "turning": "orange",
        "unripe": "limegreen",
        "unknown": "gray",
    }

    for ripeness, color in colors.items():
        selected = [
            p for p in points
            if p["ripeness"] == ripeness
        ]

        if not selected:
            continue

        fig.add_trace(go.Scatter3d(
            x=[p["x"] for p in selected],
            y=[p["y"] for p in selected],
            z=[p["z"] for p in selected],
            text=[
                f"#{p['id']}"
                if inside_working_volume(p["x"], p["y"], p["z"])
                else f"#{p['id']} OUT"
                for p in selected
            ],
            mode="markers+text",
            marker=dict(
                size=11,
                color=color,
                symbol=[
                    "circle" if inside_working_volume(
                        p["x"], p["y"], p["z"]
                    ) else "diamond"
                    for p in selected
                ],
            ),
            textfont=dict(
                size=16,
                color="white",
            ),
            textposition="top center",
            name=ripeness,
            hovertemplate=(
                "Strawberry %{text}"
                "<br>X: %{x:.3f} m"
                "<br>Y: %{y:.3f} m"
                "<br>Z: %{z:.3f} m"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=0, r=0, t=20, b=0),
        uirevision="harvestai-3d",
        scene=dict(
            xaxis=dict(
                title="X (m)",
                range=[x_lo, x_hi],
            ),
            yaxis=dict(
                title="Y (m)",
                range=[y_lo, y_hi],
            ),
            zaxis=dict(
                title="Z (m)",
                range=[z_lo, z_hi],
            ),
            # True proportions, so the tall frame looks tall.
            aspectmode="manual",
            aspectratio=dict(
                x=(x_hi - x_lo) / (z_hi - z_lo),
                y=(y_hi - y_lo) / (z_hi - z_lo),
                z=1,
            ),
        ),
    )

    return fig


if __name__ == "__main__":
    test_points = [
        {"id": 1, "x": 0.016, "y": -0.004, "z": 0.023,
         "ripeness": "ripe"},
        {"id": 2, "x": -0.047, "y": 0.006, "z": 0.020,
         "ripeness": "unripe"},
        {"id": 3, "x": -0.018, "y": 0.022, "z": 0.020,
         "ripeness": "turning"},
        {
            "id": 4,
            "x": 0.12,
            "y": 0.03,
            "z": 0.30,
            "ripeness": "ripe",
            "inside": False,
        },
    ]

    output = Path("/tmp/harvest_3d_test.html")
    make_3d_plot(test_points).write_html(str(output))
    print(f"3D visualizer saved to: {output}")
