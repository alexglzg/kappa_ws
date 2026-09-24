# Kappa real-robot experiment logs

The JSON runs in this directory are grouped by identical planner inputs:

- scenario;
- corridor sequence/geometry;
- predefined initial pose `[x, y, yaw]`;
- final pose `[x, y, yaw]`.

Measured placement variation (`start_pose_used` / `measured_pose_at_plan`) is deliberately
not used to split batches, because it is run-to-run experimental variation around the same
predefined input.

## Real-robot batches

| Directory | Input start pose | Input goal pose | JSON runs | Sessions represented |
|---|---|---|---:|---|
| `01_right_angle_turn/input_01_original` | `[-0.15, -0.35, -2.356194490192345]` | `[2.1, 3.1, -0.5235987755982988]` | 10 | `s10sept` (10) |
| `02_right_angle_overlapping_circles/input_01_original` | `[1.33, -0.34, 2.356194490192345]` | `[1.76, 0.22, -0.5235987755982988]` | 21 | `s10sept` (10), `s10septs2` (10), `labsept10` (1) |
| `02_right_angle_overlapping_circles/input_02_updated_start_and_goal` | `[1.33, -0.54, 1.5707963267948966]` | `[1.82, 0.22, -0.5235987755982988]` | 10 | `s10septs2_mod` (10) |
| `02_right_angle_overlapping_circles/input_03_final_start_and_goal` | `[1.33, -0.54, 1.5707963267948966]` | `[1.92, 0.62, 3.141592653589793]` | 10 | `s10septs2_last` (10) |
| `03_multiple_corridors/input_01` | `[-0.28, -0.7, -1.5707963267948966]` | `[1.205, 3.8320508075688773, -1.5707963267948966]` | 10 | `s10septs3` (10) |
| `04_multiple_narrow_corridors/input_01` | `[-0.23, -0.7, 0.0]` | `[1.8451150572225252, 2.29, 0.5235987755982988]` | 10 | `s10septs4` (10) |

No JSON log with an updated input pose was found for scenario 1. All 10 retained scenario-1
JSON files contain the same predefined initial pose, final pose, and corridor geometry.

The `s10septs1` and `labsept10` scenario-1 warm-up runs, and the `labsept10`
scenario-3/4 warm-up runs, were removed from the retained analysis set.

## CSV summaries

A CSV belonging entirely to one input batch is stored with that batch. The files in
`_mixed_session_summaries` contain rows from more than one scenario and therefore cannot
be assigned to one input directory without splitting or changing their contents:

- `s10sept.csv`: scenarios 1 and 2;
- `labsept10.csv`: scenarios 1, 2, 3, and 4.

Some CSVs contain more rows than there are retained JSON files with the matching session
name (notably `labsept10.csv`). The CSVs do not contain corridor geometry or input poses,
so those extra rows cannot be assigned reliably to an input batch from the available data.

`_simulation_reference` contains the `sim` session and is intentionally kept separate from
the real-robot batches.
