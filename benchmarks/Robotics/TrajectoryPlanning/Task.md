# TrajectoryPlanning — plan a collision-free robot path

## Scientific background
Motion planning for robots in environments with obstacles is fundamental to autonomous systems.
Given a 2D workspace with circular obstacles, find a path from start to goal that avoids all
obstacles while minimizing path length and curvature (for kinematic feasibility).

## Your task
```python
def plan_trajectory(start, goal, obstacles, bounds):
    """Return list of (x,y) waypoints from start to goal avoiding obstacles.
    obstacles: list of (cx, cy, r). bounds: (xmin, ymin, xmax, ymax)."""
```

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
