# CIPHER Planning Visualizer

Visualizes the step-by-step execution of the CIPHER multi-robot motion planning algorithm. Events are read sequentially from a YAML log file and displayed one at a time; use the arrow keys to step forward and backward.

```
python viz.py run.yaml                    # arrow keys to navigate
python viz.py run.yaml --auto             # auto-advance (1.5s between events)
python viz.py run.yaml --auto --delay 1.0
python viz.py run.yaml --anim-speed 2.0   # animation speed multiplier
```

After the last event, robots animate along their final low-level paths. Press any key to skip.

---

## Log File Format

A single YAML file with two top-level keys: `header` and `events`.

### Header

```yaml
header:
  dimensions: 2        # 2 or 3 (default: 3)

  robots:
    - id: r0
      dynamics: first_order_unicycle
      geometry: {type: sphere, radius: 0.15}
      start: [x, y, z]
      goal:  [x, y, z]

  grid:
    cells:
      - id: c0
        bounds:
          min: [x, y, z]
          max: [x, y, z]
```

- **`dimensions`** controls whether the visualizer uses a 3D axes or a flat 2D axes. In 2D mode only the first two components of state vectors are used for position.
- **`start`** and **`goal`** always have three components even in 2D (the third is ignored by the visualizer).

### Events

A list of typed event dicts processed in order. Events may repeat and may appear in any order (within physical reason). The visualizer updates its state incrementally — a repeated event overwrites only the data it contains.

---

#### `mapf`

High-level path for each robot as an ordered list of cell ids.

```yaml
- type: mapf
  paths:
    r0: [c0, c2, c3]
    r1: [c1, c3, c2]
```

Cells along each robot's path are shaded in the robot's color. A dashed line connects cell centers in sequence.

---

#### `low_level_paths`

Continuous trajectory for each robot as a sequence of waypoints. Only robots listed in the event are updated; others keep their existing paths.

```yaml
- type: low_level_paths
  paths:
    r0:
      - state:    [x, y, z, ...]   # state vector; first 2 or 3 components are position
        control:  [v, omega]       # control input (ignored by visualizer)
        duration: 1.0              # time to travel from this waypoint to the next
      - state:    [x, y, z, ...]
        control:  [v, omega]
        duration: 0.0              # 0.0 marks the terminal waypoint
```

Paths are drawn as colored polylines. Any previously displayed collision markers for robots that receive new paths here are automatically cleared.

---

#### `segments`

Divides each robot's current low-level path into segments by waypoint index. Only robots listed are updated.

```yaml
- type: segments
  segments:
    r0:
      - [0, 2]   # segment from waypoint index 0 to 2
      - [2, 4]
```

Segment boundary waypoints are marked with filled circles on the path.

---

#### `collision`

Reports a collision between robots at a specific time. Collision markers accumulate on screen and persist until new paths are found for **all** involved robots.

```yaml
- type: collision
  robots: [r0, r1]
  time: 1.8        # continuous time used to interpolate robot positions
```

Each involved robot is shown with a red marker at its interpolated position at the collision time.

---

#### `grid_update`

Modifies the cell decomposition. Typically used to split a cell into sub-cells after a collision is detected in that region.

```yaml
- type: grid_update
  removed: [c2]          # optional; cell ids that no longer exist
  cells:                 # new or replacement cells
    - id: c2a
      bounds:
        min: [x, y, z]
        max: [x, y, z]
```

The grid is fully redrawn after this event. `removed` is optional; if absent, only additions/replacements occur.

---

#### `coupled_planning`

Indicates that a set of robots will be planned jointly through a set of cells. Used to highlight the region being resolved.

```yaml
- type: coupled_planning
  groups:
    - robots: [r0, r1]
      cells:  [c2a, c2b, c3]
```

Cells involved in coupled planning are highlighted in orange with a thicker outline. Multiple groups can appear in one event.

---

## Notes

- **Partial updates**: `low_level_paths` and `segments` events only need to include the robots that changed. Robots not listed keep their existing data.
- **Collision clearing**: a collision's visual markers are removed automatically when `low_level_paths` provides new paths for every robot listed in that collision. If only a subset of robots get new paths, the markers persist until the rest do too.
- **Back-navigation**: stepping backward replays all events from the beginning up to the target index, so the visual state is always consistent.
