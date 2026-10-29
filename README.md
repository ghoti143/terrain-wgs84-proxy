# terrain-wgs84-proxy

Proxies Esri's [WorldElevation3D/Terrain3D ImageServer](https://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer)
and converts its tiles on the fly from **EGM96 orthometric heights** to **WGS84 ellipsoidal heights**:

    h_ellipsoid = H_orthometric + N_EGM96(lat, lon)

## How it works

- `GET …/ImageServer/tile/{level}/{row}/{col}`: fetches the upstream LERC1 (`CntZImage`) tile, decodes it,
  adds the geoid undulation N for every one of the 257×257 vertex samples, and re-encodes it as LERC1
  (the format SceneView's elevation decoder expects). The nodata mask is passed through unchanged.
  Converted tiles are cached in memory (LRU).
- `GET …/ImageServer?f=json`: upstream service JSON, with `minValues`/`maxValues` widened by the
  geoid range and a note added to the description.
- Everything else under `…/ImageServer/` (e.g. `tilemap`) is passed through as-is.
- N comes from PROJ's EGM96 15′ grid (`us_nga_egm96_15.tif`, fetched from cdn.proj.org on first start),
  sampled to a global 0.25° grid, cached to `~/.cache/terrain-wgs84-proxy/egm96_0p25.npy`, and
  bilinearly interpolated per sample.

Accuracy: values match upstream + exact PROJ N to within the LERC quantization error (`maxZError`, 0.1 m by
default, the same as upstream).

## Run

```sh
uv run terrain-wgs84-proxy            # http://127.0.0.1:8000
HOST=0.0.0.0 PORT=8080 uv run terrain-wgs84-proxy
```

Env vars: `HOST`, `PORT`, `UPSTREAM_URL`, `SERVICE_PATH`, `GEOID_CACHE`, `TILE_CACHE_SIZE`,
`MAX_Z_ERROR` (override re-encode tolerance in metres). `GET /healthz` reports N at (0, 0) — should be ≈ 17.16.

## SceneView

```js
const elevation = new ElevationLayer({
  url: "http://localhost:8000/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer",
});
const map = new Map({ basemap: "satellite", ground: new Ground({ layers: [elevation] }) });
```

## Caveats

- Only tiles are converted. `identify`, `exportImage`, `getSamples` etc. are passed through and still
  return orthometric heights.
- Ocean areas become N instead of 0 (correct for ellipsoidal heights, but sea level is no longer flat at 0).
- The spatial reference in the service JSON still has no vertical CRS; SceneView doesn't need one.
- Terrain3D's terms of use still apply to the proxied data.

## Tests

```sh
uv run pytest
```
