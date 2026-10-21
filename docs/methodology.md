# Methodology

How each feature is built and how the models are trained and evaluated. See the [README](../README.md) for results and the [data dictionary](data_dictionary.md) for columns.

## 1. Problem formulation

For each retardant drop, predict which side is protected (`pt = 1`) and which is the fire-origin side (`pt = 0`). Each drop has two buffer polygons (`buff_id` 1 and 2); each side is one row with target `pt`.

## 2. Burn timing relative to the drop

A DOB raster stores the day-of-year each pixel burned (`65535` = never burned). The drop date is converted to day-of-year (`drop_doy`). For each region, pixels are classified:

| Pixel type | Rule | Meaning |
|---|---|---|
| early | burn day `<` `drop_doy` | the fire was already there (origin side) |
| late | burn day `>` `drop_doy` | the fire arrived later (protected side) |
| unburned | value `== 65535` | protected side |

A pixel that burned on exactly `drop_doy` is neither early nor late.

In the local-buffer features, mean `early_pixels` is 16.0 for `pt = 0` vs 4.4 for `pt = 1`, and mean `unburned_pixels` is 181.3 vs 211.7.

## 3. The drop line

1. `drop_line = geom_1.intersection(geom_2)`, the shared boundary of the two buffer polygons.
2. Empty intersection: the drop is skipped.
3. `GeometryCollection`: keep the line components and merge them with `linemerge`; if there are none, skip.
4. Anything that is not a `LineString` or `MultiLineString`: skip.
5. `extend_line`: a `MultiLineString` is merged, and if it is still multi-part the longest segment is used. The line then runs from the first coordinate to the last, extended by 500,000 m (500 km) at both ends along that direction.

## 4. Spatial scales

| Scale | Region | Rationale |
|---|---|---|
| Local | The buffer polygon | Direct signal at the drop, but a small area; models using only these features plateau around 81% |
| Raster split | Each whole half of the fire raster on either side of the extended line | Thousands of pixels per side; the largest driver of improvement (about 81% to about 90%) |
| Expanded polygon | Each half intersected with a circle centered on the drop-line centroid, radius = 25% of the raster diagonal | Keeps large context while avoiding unrelated fire arms on irregularly shaped fires |
| Distance to unburned | One value per polygon centroid | The protected side sits closer to unburned area; the origin side is more deeply surrounded by fire |

### Matching halves to sides
`shapely.ops.split(raster_bbox, extended_line)` returns two polygons in unspecified order. If half A contains polygon 1's centroid, A is side 1; otherwise B is. If the split doesn't produce exactly two geometries, the drop is skipped. The split was checked visually on sample drops.

### Distance to unburned
`burned_mask = (raster != 65535)`, then `distance_transform_edt(burned_mask) * pixel_size`, sampled at each polygon centroid (indices clipped to the raster bounds). Unburned pixels have distance 0.

## 5. Paired-difference features

For each key count, `x_pair_diff = x - mean(x over the two sides of the drop)`. Comparing the sides matters more than absolute values, and the paired difference is self-normalizing across drops of very different sizes. For two sides it equals half the difference between them, with opposite signs.

Random Forest with local features only: raw about 68.4%, paired-difference only about 80.0%, both 81.1%.

## 6. Wind features

Fire spread direction is heavily wind-driven. `extract_wind_features` (Notebook 2, Part 4) describes each side of a drop relative to the wind.

### Geometry
1. Build the drop line, the two split halves and the 25%-of-diagonal circle as in sections 3-4.
2. Unit direction of the extended line: `d = (end - start) / |end - start|`. Normal: `n1 = (-d_y, d_x)`.
3. `to_side1` is the vector from the drop-line center to the centroid of side 1's clipped region. If `n1 . to_side1 < 0`, flip `n1`. The normals are `{1: n1, 2: -n1}`.

### Wind
1. The drop-line center is converted from EPSG:5070 to lon/lat (`rasterio.warp.transform`).
2. `sample_grid` finds the nearest grid cell (`argmin` of the absolute lat and lon differences) and the index of the drop date in the grid's `day` coordinate.
3. `th` is `wind_from_direction` in degrees (the direction the wind comes **from**, clockwise from north), converted to radians. `speed` is `wind_speed` in m/s.
4. The wind blows **toward** `(-sin th, -cos th)` in (east, north).
5. `wind_align = wind_to . normal_of_side`, `wind_align_spd = wind_align * speed`, `wind_speed = speed`.

### When the function returns `None`
- The line geometry is invalid or the split isn't exactly two halves.
- Either clipped region is empty.
- The lon/lat transform fails or returns a non-finite value.
- The date is not in the wind data (a `KeyError`, `IndexError` or `ValueError` is caught).
- The sampled direction or speed is NaN.

### Visual check
`plot_wind_drop(pair)` plots the DOB raster, both side regions, each side's normal labeled with `wind_align`, and the wind vector labeled with direction and speed.

### Status
The wind features are not in the 30-feature final model and are not saved to a feature CSV. The wind direction files cover 2017-2019.

## 7. Modeling setup

| Step | Detail |
|---|---|
| Merge | The `dob`, `expanded`, `split` and `dist` CSVs are merged on `(event_id, buff_id, drop_id)` |
| Clean pairs | Keep drops with exactly two rows; drops with 1, 3, 4 or 6 rows (about 2.4%) are ambiguous pairs and are excluded |
| Missing values | `mean_dob`, `median_dob` and `split_mean_dob` are filled with 0, because a polygon with no burned pixels is a valid example (`burned_pixels = 0` already encodes it). Rows with NaN in any other feature are dropped, leaving 20,022 rows from 420 fires |
| Cross-validation | `GroupKFold(n_splits=5)` with `groups = event_id`, so a fire's drops never appear in both train and test |
| Random Forest | `n_estimators=200`, `random_state=42` |
| XGBoost | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `random_state=42` |
| Metric | `accuracy_score`, mean ± std across the 5 folds |
| Baseline | Within each drop, the side with more `unburned_pixels` is predicted protected; accuracy 68.9% on 20,022 rows |
| Final model | XGBoost fit on all 20,022 rows with the 30 features. The Random Forest fit on the same data is also saved in `models/` |
