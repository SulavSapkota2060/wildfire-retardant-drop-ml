# Data Dictionary

Definitions of the datasets, identifiers and feature columns used in the pipeline. All raw data was provided by the professor.

## Conventions

| Term | Meaning |
|---|---|
| `buff_id` | `1` or `2`: which side of the drop. Unrelated to the protected/origin label. |
| `pt` | Target. `1` = protected side, `0` = origin side. Copied from the shapefile column `new_pt`. |
| DOB | Day of Burning raster. Pixel values are day-of-year; `65535` is nodata and marks pixels that never burned. |
| `drop_doy` | Drop date as day-of-year, for comparison with DOB values. |
| CRS | EPSG:5070. The rasters are in this CRS and the drops are reprojected to it. |
| Clean pair | A `drop_id` with exactly two rows. Modeling keeps only these. |

## Identifiers

| Column | Description |
|---|---|
| `Evnt_ID` / `event_id` | Fire incident ID; also the DOB file name (`<Evnt_ID>.dob.tif`). Used as the group key for `GroupKFold`. |
| `drop_id` | Unique drop identifier built in Notebook 1 as `Evnt_ID_Latitd1(5 dp)_Longtd1(5 dp)_UTCdtTm`. Pairs the two sides of one drop. |
| `buff_id` | See above. |

## Source shapefile: `data/raw/buffered_drops/cleaned_drop_buffers.shp`

Columns:

`irwinid, IncNum1, OBJECTI, Drp_ID1, Vendor1, FalsPs1, SpltDr1, Latitd1, Longtd1, LoclDt1, LoclHr1, Knots1, HdngTr1, AltFet1, LoclTZ1, Gallns1, Tal_Nmb, UTCdtTm, lclMnth, loclDy1, loclYr1, LinLnM1, IncNam1, IADrop1, AC_Typ1, Fluid1, NONFIRE, Shp_Lng, DropMnt, DrpScnd, Shp_L_1, X, cntn_dt, Evnt_ID, buff_id, pst_trt, new_pt, Shape_Leng, Shape_Area, Shape_Le_1, Shape_Le_2, Field`

Columns the pipeline uses:

| Column | Used for |
|---|---|
| `Evnt_ID` | Fire ID; matched to the DOB file name |
| `buff_id` | Side of the drop |
| `new_pt` | Target label (`pt`) |
| `Latitd1`, `Longtd1`, `UTCdtTm` | Build `drop_id` |
| `loclYr1`, `lclMnth`, `loclDy1` | Build `local_date` |
| `AltFet1` | Aircraft altitude at the time of the drop (saved as `altitude`) |
| `Knots1` | Aircraft speed at the time of the drop (saved as `knots`) |
| `Gallns1` | Volume of retardant dropped (saved as `gallons`) |
| `DrpScnd` | Duration of the drop in seconds (saved as `drop_seconds`) |

The remaining columns are not used.

## `data/interim/drops_clean.geojson` (Notebook 1 output)

The shapefile columns plus: `Evnt_ID_str` (event ID as text), `tif_path` (path to the fire's DOB raster), `local_date` (from `loclYr1`, `lclMnth`, `loclDy1`), `drop_doy` (day-of-year of `local_date`) and `drop_id`. Geometry is in EPSG:5070.

## Feature files: `data/features/train/`

All are keyed by `(event_id, buff_id, drop_id)`.

### `training_features_dob.csv`: local buffer (Notebook 2, Part 1)
| Column | Description |
|---|---|
| `mean_dob`, `median_dob` | Mean / median burn day of the burned pixels under the polygon. NaN when nothing burned (filled with 0 in Notebook 3). |
| `early_pixels` | Burned pixels with burn day < `drop_doy`: the fire was already there (origin side). |
| `late_pixels` | Burned pixels with burn day > `drop_doy`: the fire arrived later (protected side). |
| `unburned_pixels` | Pixels equal to `65535`. |
| `total_pixels` | All pixels in the cropped bounding box (burned + unburned + boundary). |
| `burned_pixels` | Pixels not equal to `65535`. |
| `altitude`, `knots`, `gallons`, `drop_seconds` | Drop metadata from the shapefile. Not model features. |
| `pt` | Target label. |

### `training_features_split.csv`: whole-raster split (Notebook 2, Part 2)
The raster is split by the extended drop line; statistics are computed over each whole half.

`split_mean_dob`, `split_early_pixels`, `split_late_pixels`, `split_unburned_pixels`, `split_burned_pixels`, `split_total_pixels`: same definitions as the local columns, over the half.

### `training_features_expanded.csv`: expanded polygon (Notebook 2, Part 3b)
Each half is intersected with a circle around the drop-line centroid; radius = 25% of the raster diagonal.

`exp_mean_dob`, `exp_burned_pixels`, `exp_unburned_pixels`, `exp_early_pixels`, `exp_late_pixels`, `exp_total_pixels`: same definitions, over the clipped region.

### `training_features_dist.csv`: distance to unburned (Notebook 2, Part 3a)
| Column | Description |
|---|---|
| `dist_to_unburned` | Distance in meters from the polygon centroid to the nearest unburned pixel: `distance_transform_edt` of the burned mask times the pixel size. 0 if the centroid is on unburned ground. |

## Paired-difference columns (built in Notebook 3)

`feature - mean(feature over the two rows of the same drop_id)`.

| Column | Built from |
|---|---|
| `unburned_pair_diff`, `early_pair_diff`, `burned_pair_diff` | `unburned_pixels`, `early_pixels`, `burned_pixels` |
| `split_unburned_pair_diff`, `split_early_pair_diff`, `split_burned_pair_diff` | the `split_` equivalents |
| `exp_unburned_pair_diff`, `exp_early_pair_diff`, `exp_burned_pair_diff` | the `exp_` equivalents |
| `dist_pair_diff` | `dist_to_unburned` |

**Final model feature list (30):** `mean_dob, median_dob, early_pixels, late_pixels, unburned_pixels, burned_pixels, total_pixels, unburned_pair_diff, early_pair_diff, burned_pair_diff, dist_to_unburned, dist_pair_diff, split_mean_dob, split_early_pixels, split_late_pixels, split_unburned_pixels, split_burned_pixels, split_total_pixels, split_burned_pair_diff, split_unburned_pair_diff, split_early_pair_diff, exp_mean_dob, exp_burned_pixels, exp_unburned_pixels, exp_early_pixels, exp_late_pixels, exp_total_pixels, exp_burned_pair_diff, exp_unburned_pair_diff, exp_early_pair_diff`

## Wind features (Notebook 2, Part 4)

Returned by `extract_wind_features(pair)` as one dict per side. They are not saved to any CSV and are not in the 30-feature list.

| Field | Description |
|---|---|
| `event_id`, `buff_id`, `drop_id` | Identifiers |
| `wind_align` | Dot product of the wind's blowing-toward vector and the side's unit normal: `+1` = the side is straight downwind of the line, `-1` = straight upwind, `0` = wind parallel to the line. |
| `wind_align_spd` | `wind_align x wind_speed` |
| `wind_speed` | Wind speed on the drop day, in m/s |

Inputs: `wind_from_direction` (the direction the wind comes from) from `data/raw/th/th_<year>.nc`, and `wind_speed` (m/s) from `data/raw/vs/vs_<year>.nc`, with coordinates `lat`, `lon`, `day`. The pipeline loads 2017-2019. The `th/` files cover 2017-2019 and the `vs/` files cover 2017-2021.

## `data/raw/` inventory

| Folder | Contents |
|---|---|
| `buffered_drops/` | Drop polygons (shapefile) |
| `DOBs/` | 444 `<Evnt_ID>.dob.tif` rasters |
| `th/` | Wind direction, `th_2017-2019.nc` |
| `vs/` | Wind speed, `vs_2017-2021.nc`, plus `vs_mtbs17/18/19_long.csv` |
| `vpd/` | `vpd_2017-2021.nc`, plus `vpd_mtbs17/18/19/20_long.csv`. Unused. |
| `LANDFIRE_fuel/`, `mtbs_mosaic/`, `dem_tiles/` | Used in `notebooks/exploratory/` only |
| `validation_2021/` | 2020-2021 validation drops |

## Outputs

| File | Description |
|---|---|
| `outputs/failed_predictions.csv` | 317 misclassified rows from `notebooks/exploratory/03_modeling.ipynb` (the DOB + split model), with columns `event_id, drop_id, buff_id, pt, predicted, mean_dob, split_burned_pixels, split_unburned_pixels, unburned_pixels, burned_pixels`. |
| `outputs/failed_predictions.geojson` | Geometries of the failed drops for QGIS inspection (358 geometries). |
| `models/model_final_xgboost.pkl` | Final model (XGBoost). |
| `models/model_final_rf.pkl` | Random Forest trained on the same features, kept for comparison. |
| `models/model_features_xgboost.pkl`, `models/model_features_rf.pkl` | The feature lists saved with them. |
