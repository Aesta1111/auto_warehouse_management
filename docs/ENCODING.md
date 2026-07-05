# Observation and Action Encoding

This document defines the model-facing vector contracts for the pinned tiny
TA-RWARE environment. Changing their order or meaning requires retraining
models and recording a new experiment configuration.

## Observation encoding

`WarehouseObservation.encode()` returns a one-dimensional NumPy array with
`float32` dtype and shape:

```text
(7 * number_of_agvs + 4 * number_of_pickers + 2 * number_of_shelf_locations,)
```

For the primary tiny environment:

```text
3 AGVs, 2 pickers, 48 shelf locations
7 * 3 + 4 * 2 + 2 * 48 = 125
shape = (125,)
```

Agents use ascending TA-RWARE simulator ID. Shelf locations use ascending
destination ID. `observation.feature_names` gives the name for every vector
index and `observation.encoded_shape` gives the expected shape.

### Per-AGV features

Each AGV contributes seven values:

| Offset | Feature | Encoding |
| ---: | --- | --- |
| 0 | carrying shelf | `0.0` or `1.0` |
| 1 | carrying a requested shelf | `0.0` or `1.0` |
| 2 | performing load/unload toggle | `0.0` or `1.0` |
| 3 | current row | unnormalized grid coordinate |
| 4 | current column | unnormalized grid coordinate |
| 5 | target row | coordinate, or `0.0` without a target |
| 6 | target column | coordinate, or `0.0` without a target |

### Per-picker features

Each picker contributes four values:

| Offset | Feature | Encoding |
| ---: | --- | --- |
| 0 | current row | unnormalized grid coordinate |
| 1 | current column | unnormalized grid coordinate |
| 2 | target row | coordinate, or `0.0` without a target |
| 3 | target column | coordinate, or `0.0` without a target |

### Per-shelf-location features

Each fixed shelf location contributes two values:

| Offset | Feature | Encoding |
| ---: | --- | --- |
| 0 | shelf present | `0.0` or `1.0` |
| 1 | present shelf is requested | `0.0` or `1.0` |

The valid-action mask is supplied separately because masking is part of action
selection, not warehouse-state encoding.

## Action decoding

`TarwareEnvWrapper.decode(model_output)` accepts a one-dimensional integer
vector with shape:

```text
(number_of_agvs + number_of_pickers,)
```

Element `i` selects the destination for observation agent `i`. For the primary
tiny environment the shape is `(5,)`:

```text
[agv-0, agv-1, agv-2, picker-0, picker-1]
```

Each value is a TA-RWARE macro-destination ID:

- `0`: no new destination; required while an agent is busy.
- `1..number_of_goals`: goal locations.
- Remaining IDs: fixed shelf locations.

The tiny environment has 55 choices, so each value must be in `[0, 54]`.
`decode` expects selected integer IDs, not logits or probabilities. Batched
arrays, floating-point arrays, invalid or masked IDs, and conflicting
same-role assignments are rejected. It returns a validated `JointAction`,
which can be passed directly to `TarwareEnvWrapper.step()`.

```python
raw_action = np.asarray([12, 0, 21, 12, 0], dtype=np.int64)
action = environment.decode(raw_action)
observation, rewards, terminated, truncated, info = environment.step(action)
```

An AGV and picker may select the same shelf for cooperation. Two AGVs or two
pickers may not select the same nonzero destination in one joint action.

## Collection types

Pydantic accepts lists as constructor input and converts canonical ordered
collections to tuples. Tuples are retained for agents, shelves, goals, and
joint actions because their position determines model feature or action
meaning. A mutable list could change that meaning after validation. Temporary
collections used while building an observation remain ordinary lists.
