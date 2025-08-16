"""
Raster-based high-order topography approximant

Provides a fast High-Order local polynomial approximant over a DEM raster (rasterio)

This module provides:

-

    Loads a raster DEM via rasterio, builds a RegularGridInterpolator for fast z queries
    and exposes convenience methods (bounds checking, pixel spacing, etc).

- RasterHighOrderApproximant

    local polynomial least-squares approximant over raster neighborhoods with:

        -   optional parallel precompute of coefficients using joblib + multiprocessing
        -   optional memmap-backed tiled precompute to avoid large memory peaks
        -   GeoTIFF export / import of coefficient stacks (multiband)
        -   optional numba acceleration for hot kernels
        -   precompute_all_coeffs supports memmap_path and tile_size to perform out-of-core
            tiled precomputation writing coefficients to disk as a NumPy memmap.
        -   export_coeffs_geotiff and load_coeffs_geotiff provide a portable multiband GeoTIFF
            for coefficients (one band per coefficient) with metadata describing the monomial basis.


Notes
    -   When using memmap-based precompute, tiles are processed sequentially; within a tile
        the coefficient computations are performed in parallel and results are gathered in the
        main process and written to the memmap to avoid concurrent writes to the same region.
    -   memmap dtype is float32 by default to reduce disk usage; change via dtype param.


Deviations from paper
    -   The paper's HighOrderApproximant is point-cloud/triangle-based. This implementation
        instead exploits raster regularity to gather neighborhoods much faster and to allow
        optional precomputation on cell centers. This speeds up metric computation and
        mesh generation pipelines when the input is a DEM raster (very common).
    -   The polynomial basis is identical (monomials up to degree p), and derivative
        formulas follow the same algebraic rules; therefore the approximant is consistent
        with the paper's intent. Any deviation is documented via the `precompute` option and
        the fallback behavior (bilinear finite-diff), which is a pragmatic robustness choice.

"""

from __future__ import annotations

import datetime
import json
import os
from collections.abc import Iterable

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import lstsq

# Optional dependencies
try:
    import rasterio
    from rasterio.transform import Affine
except Exception:
    rasterio = None
    Affine = None

# Parallel precompute
try:
    from joblib import Parallel, delayed

    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

# tqdm for progress
try:
    from tqdm import tqdm

    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# optional numba
try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

# -----------------------
# Optional numba kernels
# -----------------------
if _HAS_NUMBA:

    @njit(inline="always")
    def _vandermonde_numba(us, vs, monomials_i, monomials_j):
        n = us.size
        m = monomials_i.size
        A = np.empty((n, m), dtype=np.float64)
        for k in range(m):
            i = monomials_i[k]
            j = monomials_j[k]
            for t in range(n):
                A[t, k] = (us[t] ** i) * (vs[t] ** j)
        return A

    @njit(inline="always")
    def _eval_poly_numba(coeffs, u, v, monomials_i, monomials_j):
        val = 0.0
        for k in range(coeffs.size):
            val += coeffs[k] * (u ** monomials_i[k]) * (v ** monomials_j[k])
        return val
else:

    def _vandermonde_numba(us, vs, monomials_i, monomials_j):
        us = np.asarray(us)
        vs = np.asarray(vs)
        m = len(monomials_i)
        A = np.empty((us.size, m), dtype=float)
        for k, (i, j) in enumerate(zip(monomials_i, monomials_j, strict=False)):
            A[:, k] = (us ** int(i)) * (vs ** int(j))
        return A

    def _eval_poly_numba(coeffs, u, v, monomials_i, monomials_j):
        val = 0.0
        for k in range(len(coeffs)):
            i = monomials_i[k]
            j = monomials_j[k]
            val += coeffs[k] * (u**i) * (v**j)
        return float(val)


# -----------------------
# Worker (picklable) for multiprocessing
# -----------------------
def _compute_coeff_cell(
    r: int,
    c: int,
    data: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    height: int,
    width: int,
    monomials_i: np.ndarray,
    monomials_j: np.ndarray,
    degree: int,
    support_pixels: int,
    min_samples: int,
):
    """
    Compute polynomial coefficients for single raster cell at (r,c).
    Implemented at module level so it is picklable by multiprocessing backends.

    Returns (r, c, coeffs_array or None)
    """
    try:
        zcenter = data[r, c]
        if not np.isfinite(zcenter):
            return (r, c, None)
        half = support_pixels
        row_min = max(0, r - half)
        row_max = min(height - 1, r + half)
        col_min = max(0, c - half)
        col_max = min(width - 1, c + half)
        rows = np.arange(row_min, row_max + 1, dtype=int)
        cols = np.arange(col_min, col_max + 1, dtype=int)
        CC, RR = np.meshgrid(cols, rows)
        CCf = CC.ravel()
        RRf = RR.ravel()
        xs_pts = xs[CCf]
        ys_pts = ys[RRf]
        zs = data[RRf, CCf]
        mask = np.isfinite(zs)
        if np.count_nonzero(mask) < min_samples:
            return (r, c, None)
        xs_s = xs_pts[mask]
        ys_s = ys_pts[mask]
        zs_s = zs[mask].astype(float)
        x0 = xs[c]
        y0 = ys[r]
        us = xs_s - x0
        vs = ys_s - y0
        A = _vandermonde_numba(us, vs, monomials_i, monomials_j)
        cvec, *_ = lstsq(A, zs_s)
        return (r, c, cvec.astype(float))
    except Exception:
        return (r, c, None)


class RasterTopography:
    """Wrapper to load a single-band DEM raster via rasterio and provide fast interpolation queries.

    Args:
        raster_path: Path to the raster file (GeoTIFF or similar).
        band: Band index to read (1-based).
        nodata_fill: Optional replacement for nodata (default: None -> NaN).
        verbosity: Verbosity level.
    """

    def __init__(
        self, raster_path: str, band: int = 1, nodata_fill: float | None = None, verbosity: int = 1
    ):
        if rasterio is None:
            raise RuntimeError(
                "rasterio is required to use RasterTopography (pip install rasterio)"
            )

        self.raster_path = raster_path
        self.band = int(band)
        self.verbosity = verbosity

        with rasterio.open(raster_path) as src:
            self.width = src.width
            self.height = src.height
            self.crs = src.crs
            self.transform = src.transform  # Affine
            arr = src.read(self.band, masked=True)
            self._nodata = src.nodata

        if hasattr(arr, "mask"):
            data = arr.filled(np.nan).astype(float)
        else:
            data = np.asarray(arr, dtype=float)
            if self._nodata is not None:
                data[data == self._nodata] = np.nan

        self.data = data  # shape (rows, cols) = (height, width)

        a = self.transform.a
        b = self.transform.b
        c = self.transform.c
        d = self.transform.d
        e = self.transform.e
        f = self.transform.f

        cols = np.arange(self.width)
        rows = np.arange(self.height)
        xs = c + cols * a + 0.0 * b
        ys = f + 0.0 * d + rows * e

        self.xs = xs
        self.ys = ys

        self.interpolator = RegularGridInterpolator(
            (self.ys, self.xs), self.data, bounds_error=False, fill_value=np.nan, method="linear"
        )

        if self.verbosity:
            print(f"[RasterTopography] loaded {raster_path}, size={self.width}x{self.height}")

    def bounds(self) -> tuple[float, float, float, float]:
        """Return (xmin, xmax, ymin, ymax)."""
        xmin = self.xs.min()
        xmax = self.xs.max()
        ymin = float(min(self.ys))
        ymax = float(max(self.ys))
        return float(xmin), float(xmax), ymin, ymax

    def contains(self, xy: tuple[float, float]) -> bool:
        """Return True if (x,y) inside raster bounding box."""
        x, y = float(xy[0]), float(xy[1])
        xmin, xmax, ymin, ymax = self.bounds()
        return (x >= xmin) and (x <= xmax) and (y >= ymin) and (y <= ymax)

    def sample(self, xy: tuple[float, float]) -> float:
        """Sample bilinear interpolated height at (x,y)."""
        x, y = float(xy[0]), float(xy[1])
        val = self.interpolator((y, x))
        return float(val)

    def pixel_size(self) -> tuple[float, float]:
        """Return pixel spacing (dx, dy)."""
        dx = abs(self.transform.a)
        dy = abs(self.transform.e)
        return float(dx), float(dy)

    def xy_to_colrow(self, x: float, y: float) -> tuple[float, float]:
        """Return fractional (col, row) for (x,y)."""
        if Affine is not None:
            inv = ~self.transform
            colf, rowf = inv * (x, y)
            return float(colf), float(rowf)
        else:
            a = self.transform.a
            b = self.transform.b
            d = self.transform.d
            e = self.transform.e
            c = self.transform.c
            f = self.transform.f
            A = np.array([[a, b], [d, e]], dtype=float)
            rhs = np.array([x - c, y - f], dtype=float)
            colf, rowf = np.linalg.solve(A, rhs)
            return float(colf), float(rowf)

    def colrow_to_xy(self, col: float, row: float) -> tuple[float, float]:
        """Map (col,row) back to (x,y)."""
        x = self.transform.c + col * self.transform.a + row * self.transform.b
        y = self.transform.f + col * self.transform.d + row * self.transform.e
        return float(x), float(y)


class RasterHighOrderApproximant:
    """Local polynomial least-squares approximant over raster neighborhoods.

    Args:
        raster_topo: RasterTopography instance.
        degree: polynomial degree (default 3).
        support_pixels: half-width of stencil in pixels (default = max(3, 3*degree)).
        min_samples: minimum valid samples required for LS fit (default num_coeffs + 3).
        verbosity: verbosity level.
        precompute: if True will call precompute_all_coeffs during init.
    """

    def __init__(
        self,
        raster_topo: RasterTopography,
        degree: int = 3,
        support_pixels: int | None = None,
        min_samples: int | None = None,
        verbosity: int = 1,
        precompute: bool = False,
        precompute_chunksize: int = 20000,
    ):
        self.rtopo = raster_topo
        self.degree = int(degree)
        self.verbosity = verbosity
        if support_pixels is None:
            self.support_pixels = max(3, 3 * self.degree)
        else:
            self.support_pixels = int(support_pixels)
        self.monomials = [
            (i, j) for i in range(self.degree + 1) for j in range(self.degree + 1 - i)
        ]
        self.num_coeffs = len(self.monomials)
        self.monomials_i = np.array([mi for mi, mj in self.monomials], dtype=np.int64)
        self.monomials_j = np.array([mj for mi, mj in self.monomials], dtype=np.int64)
        self.min_samples = min_samples if min_samples is not None else (self.num_coeffs + 3)
        self._precomputed = None
        self._precompute_mask = None
        self._precompute_chunksize = int(precompute_chunksize)
        if precompute:
            if self.verbosity:
                print(
                    "[RasterHO] Starting precompute of polynomial coeffs (parallel, may be memory-heavy)..."
                )
            # default precompute writes in-memory (no memmap)
            self.precompute_all_coeffs(n_jobs=-1, use_tqdm=True)

    def _build_vandermonde(self, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
        """Build Vandermonde matrix for centered monomials."""
        return _vandermonde_numba(us, vs, self.monomials_i, self.monomials_j)

    def _eval_poly(
        self, coeffs: np.ndarray, xy: tuple[float, float], center: tuple[float, float]
    ) -> float:
        """Evaluate polynomial at xy given centered coefficients and center xy."""
        u = float(xy[0]) - float(center[0])
        v = float(xy[1]) - float(center[1])
        return _eval_poly_numba(coeffs, u, v, self.monomials_i, self.monomials_j)

    def _grad_hess_from_coeffs(self, coeffs: np.ndarray, center: tuple[float, float]):
        """Return two callables eval_grad(xy) and eval_hess(xy) given coefficients."""
        mon_i = self.monomials_i
        mon_j = self.monomials_j

        def eval_grad(xy):
            u = float(xy[0]) - float(center[0])
            v = float(xy[1]) - float(center[1])
            gx = 0.0
            gy = 0.0
            for k in range(coeffs.size):
                i = mon_i[k]
                j = mon_j[k]
                if i > 0:
                    gx += coeffs[k] * i * (u ** (i - 1)) * (v**j)
                if j > 0:
                    gy += coeffs[k] * j * (u**i) * (v ** (j - 1))
            return np.array([gx, gy], dtype=float)

        def eval_hess(xy):
            u = float(xy[0]) - float(center[0])
            v = float(xy[1]) - float(center[1])
            dxx = 0.0
            dxy = 0.0
            dyy = 0.0
            for k in range(coeffs.size):
                i = mon_i[k]
                j = mon_j[k]
                if i > 1:
                    dxx += coeffs[k] * i * (i - 1) * (u ** (i - 2)) * (v**j)
                if i > 0 and j > 0:
                    dxy += coeffs[k] * i * j * (u ** (i - 1)) * (v ** (j - 1))
                if j > 1:
                    dyy += coeffs[k] * j * (j - 1) * (u**i) * (v ** (j - 2))
            return np.array([[dxx, dxy], [dxy, dyy]], dtype=float)

        return eval_grad, eval_hess

    def fit_at_point(self, xy: tuple[float, float]) -> tuple[np.ndarray, tuple[float, float]]:
        """Fit polynomial at xy using local raster pixel centers.

        Returns:
            coeffs (num_coeffs,), center (x0, y0)
        Raises:
            RuntimeError if insufficient valid samples exist.
        """
        x0, y0 = float(xy[0]), float(xy[1])
        colf, rowf = self.rtopo.xy_to_colrow(x0, y0)
        col0 = int(round(colf))
        row0 = int(round(rowf))
        half = int(self.support_pixels)
        row_min = max(0, row0 - half)
        row_max = min(self.rtopo.height - 1, row0 + half)
        col_min = max(0, col0 - half)
        col_max = min(self.rtopo.width - 1, col0 + half)

        rows = np.arange(row_min, row_max + 1, dtype=int)
        cols = np.arange(col_min, col_max + 1, dtype=int)
        CC, RR = np.meshgrid(cols, rows)
        CCf = CC.ravel()
        RRf = RR.ravel()
        if CCf.size == 0:
            raise RuntimeError("Empty stencil for fitting (point outside raster?)")

        xs = self.rtopo.xs[CCf]
        ys = self.rtopo.ys[RRf]
        zs = self.rtopo.data[RRf, CCf]
        mask = np.isfinite(zs)
        if np.count_nonzero(mask) < self.min_samples:
            raise RuntimeError("Insufficient valid samples for polynomial fit")

        xs = xs[mask]
        ys = ys[mask]
        zs = zs[mask].astype(float)

        us = xs - x0
        vs = ys - y0

        A = self._build_vandermonde(us, vs)
        c, *_ = lstsq(A, zs)
        return c.astype(float), (x0, y0)

    def query_at(self, xy: tuple[float, float]) -> tuple[float, np.ndarray, np.ndarray]:
        """Query smooth z, gradient and Hessian at (x, y).

        Returns:
            z, grad (2,), hess (2,2)
        """
        xq, yq = float(xy[0]), float(xy[1])
        z_bilin = self.rtopo.sample((xq, yq))

        if self._precomputed is not None:
            colf, rowf = self.rtopo.xy_to_colrow(xq, yq)
            coli = int(round(colf))
            rowi = int(round(rowf))
            if 0 <= coli < self.rtopo.width and 0 <= rowi < self.rtopo.height:
                if self._precompute_mask is not None and not self._precompute_mask[rowi, coli]:
                    pass
                else:
                    coeffs = self._precomputed[rowi, coli, :].astype(float)
                    center_xy = self.rtopo.colrow_to_xy(coli, rowi)
                    z = self._eval_poly(coeffs, (xq, yq), center_xy)
                    eval_grad, eval_hess = self._grad_hess_from_coeffs(coeffs, center_xy)
                    grad = eval_grad((xq, yq))
                    hess = eval_hess((xq, yq))
                    return float(z), grad, hess

        try:
            coeffs, center = self.fit_at_point((xq, yq))
            z = self._eval_poly(coeffs, (xq, yq), center)
            eval_grad, eval_hess = self._grad_hess_from_coeffs(coeffs, center)
            grad = eval_grad((xq, yq))
            hess = eval_hess((xq, yq))
            return float(z), grad, hess
        except Exception as e:
            if self.verbosity:
                print(
                    f"[RasterHO] warning: local fit failed at ({xq:.3f},{yq:.3f}): {e}; using fallback finite-diff"
                )
            # fallback quadratic fit on small neighborhood
            colf, rowf = self.rtopo.xy_to_colrow(xq, yq)
            col0 = int(round(colf))
            row0 = int(round(rowf))
            rows = np.arange(row0 - 1, row0 + 2)
            cols = np.arange(col0 - 1, col0 + 2)
            pts = []
            vals = []
            for r in rows:
                for c in cols:
                    if 0 <= r < self.rtopo.height and 0 <= c < self.rtopo.width:
                        v = self.rtopo.data[r, c]
                        if np.isfinite(v):
                            x_c, y_c = self.rtopo.colrow_to_xy(c, r)
                            pts.append((x_c, y_c))
                            vals.append(float(v))
            if len(vals) < 3:
                return (
                    float(z_bilin),
                    np.array([0.0, 0.0], dtype=float),
                    np.zeros((2, 2), dtype=float),
                )
            try:
                pts = np.array(pts)
                vals = np.array(vals)
                u = pts[:, 0] - xq
                v = pts[:, 1] - yq
                B = np.column_stack([np.ones(len(u)), u, v, u**2, u * v, v**2])
                sol, *_ = lstsq(B, vals)
                gz = float(sol[1])
                gy = float(sol[2])
                hxx = 2.0 * float(sol[3])
                hxy = float(sol[4])
                hyy = 2.0 * float(sol[5])
                return (
                    float(z_bilin),
                    np.array([gz, gy], dtype=float),
                    np.array([[hxx, hxy], [hxy, hyy]], dtype=float),
                )
            except Exception:
                return (
                    float(z_bilin),
                    np.array([0.0, 0.0], dtype=float),
                    np.zeros((2, 2), dtype=float),
                )

    # -----------------------
    # Precompute coefficients (with memmap tiled option)
    # -----------------------
    def precompute_all_coeffs(
        self,
        rows: Iterable[int] | None = None,
        cols: Iterable[int] | None = None,
        n_jobs: int = -1,
        use_tqdm: bool = True,
        memmap_path: str | None = None,
        memmap_dtype: str = "float32",
        tile_size: tuple[int, int] = (256, 256),
    ):
        """Precompute polynomial coefficients at raster cell centers in parallel.

        This function supports an out-of-core mode: if memmap_path is provided, the
        coefficients are written to a NumPy memmap on disk tile-by-tile to avoid
        high memory usage.

        Args:
            rows: Iterable of row indices to precompute (default all).
            cols: Iterable of column indices to precompute (default all).
            n_jobs: joblib n_jobs (multiprocessing). -1 uses all cores.
            use_tqdm: show progress bar when available.
            memmap_path: if provided, a NumPy memmap file path is created and used to store
                         the (height, width, num_coeffs) array on disk. Existing file will be overwritten.
            memmap_dtype: dtype used for memmap (default float32).
            tile_size: tuple (tile_rows, tile_cols) controlling size of tiles processed sequentially.

        After call:
            self._precomputed and self._precompute_mask are set. If memmap was used, self._precomputed
            is a np.memmap object backed by memmap_path.
        """
        h = self.rtopo.height
        w = self.rtopo.width
        row_idx = np.arange(h, dtype=int) if rows is None else np.asarray(list(rows), dtype=int)
        col_idx = np.arange(w, dtype=int) if cols is None else np.asarray(list(cols), dtype=int)

        total_cells = int(row_idx.size * col_idx.size)
        if self.verbosity:
            print(
                f"[RasterHO] precompute: preparing {total_cells} cells (rows {row_idx.size} x cols {col_idx.size})"
            )

        num_coeffs = self.num_coeffs
        # Prepare memmap or in-memory array
        use_memmap = memmap_path is not None
        if use_memmap:
            # Ensure parent dir exists
            os.makedirs(os.path.dirname(os.path.abspath(memmap_path)) or ".", exist_ok=True)
            # create memmap (w+ will create file)
            precomp = np.memmap(
                memmap_path, dtype=memmap_dtype, mode="w+", shape=(h, w, num_coeffs)
            )
            # initialize with NaNs
            precomp[:] = np.nan
            # mask will be saved as a small .npy next to memmap
            mask_path = memmap_path + ".mask.npy"
            mask_arr = np.zeros((h, w), dtype=np.bool_)
        else:
            precomp = np.full((h, w, num_coeffs), np.nan, dtype=float)
            mask_arr = np.zeros((h, w), dtype=bool)

        # prepare local variables for worker
        data = self.rtopo.data
        xs = self.rtopo.xs
        ys = self.rtopo.ys
        mon_i = self.monomials_i
        mon_j = self.monomials_j
        degree = self.degree
        support_pixels = self.support_pixels
        min_samples = self.min_samples
        height = h
        width = w

        tile_r, tile_c = int(tile_size[0]), int(tile_size[1])
        # compute tile ranges
        row_tiles = list(range(0, row_idx.size, tile_r))
        col_tiles = list(range(0, col_idx.size, tile_c))

        tasks_total = 0
        # iterate tiles sequentially (safe memmap writes)
        for r_off in tqdm(row_tiles, desc="row tiles") if use_tqdm and _HAS_TQDM else row_tiles:
            for c_off in tqdm(col_tiles, desc="col tiles") if use_tqdm and _HAS_TQDM else col_tiles:
                # compute tile cell indices (in original row_idx/col_idx)
                r_slice = row_idx[r_off : r_off + tile_r]
                c_slice = col_idx[c_off : c_off + tile_c]
                tasks = [(int(r), int(c)) for r in r_slice for c in c_slice]
                tasks_total += len(tasks)
                if len(tasks) == 0:
                    continue

                # evaluate tasks in parallel (workers only compute and return results)
                if not _HAS_JOBLIB or n_jobs == 1:
                    iterator = tasks
                    if use_tqdm and _HAS_TQDM:
                        iterator = tqdm(tasks, desc=f"tile [{r_off},{c_off}]")
                    results = []
                    for r, c in iterator:
                        res = _compute_coeff_cell(
                            r,
                            c,
                            data,
                            xs,
                            ys,
                            height,
                            width,
                            mon_i,
                            mon_j,
                            degree,
                            support_pixels,
                            min_samples,
                        )
                        results.append(res)
                else:
                    parallel = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=0)
                    calls = (
                        delayed(_compute_coeff_cell)(
                            r,
                            c,
                            data,
                            xs,
                            ys,
                            height,
                            width,
                            mon_i,
                            mon_j,
                            degree,
                            support_pixels,
                            min_samples,
                        )
                        for (r, c) in tasks
                    )
                    if use_tqdm and _HAS_TQDM:
                        # joblib + tqdm: run then approximate progress per tile
                        results = parallel(calls)
                    else:
                        results = parallel(calls)

                # gather results and write them into precomp and mask (main process)
                for rr, cc, coeff in results:
                    if coeff is not None:
                        precomp[rr, cc, :] = coeff.astype(precomp.dtype)
                        mask_arr[rr, cc] = True

                # end of tile
                if use_memmap:
                    # flush memmap to disk for safety
                    try:
                        precomp.flush()
                    except Exception:
                        pass

        # attach arrays to object
        self._precomputed = precomp
        self._precompute_mask = mask_arr

        # if memmap mode, save mask to disk
        if use_memmap:
            np.save(mask_path, mask_arr)
            if self.verbosity:
                print(
                    f"[RasterHO] memmap coefficients written to {os.path.abspath(memmap_path)} (mask={mask_path})"
                )
        if self.verbosity:
            nvalid = int(mask_arr.sum())
            print(
                f"[RasterHO] precompute finished: valid coeffs at {nvalid}/{h * w} cells (tiles processed: {len(row_tiles) * len(col_tiles)})"
            )

    # -----------------------
    # GeoTIFF export / import
    # -----------------------
    def export_coeffs_geotiff(
        self,
        path: str,
        nodata: float = -9999.0,
        dtype: str = "float32",
        extra_tags: dict | None = None,
    ):
        """Export precomputed coefficients as a multiband GeoTIFF.

        Each coefficient occupies a band; band order matches self.monomials ordering.
        """
        if rasterio is None:
            raise RuntimeError("rasterio is required to export GeoTIFFs (pip install rasterio)")

        if self._precomputed is None:
            raise RuntimeError(
                "No precomputed coefficients available. Run precompute_all_coeffs() first or load a coefficients file."
            )

        h, w, nc = self._precomputed.shape
        if nc != self.num_coeffs:
            raise RuntimeError(
                "Precomputed coefficients shape mismatch with expected number of monomials"
            )

        transform = self.rtopo.transform
        crs = self.rtopo.crs
        profile = {
            "driver": "GTiff",
            "height": int(h),
            "width": int(w),
            "count": int(nc),
            "dtype": dtype,
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
        }

        meta = {
            "created_by": "abl_mesh RasterHighOrderApproximant",
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "degree": int(self.degree),
            "num_coeffs": int(self.num_coeffs),
            "support_pixels": int(self.support_pixels),
            "min_samples": int(self.min_samples),
            "monomials": json.dumps(self.monomials),
        }
        if extra_tags:
            for k, v in extra_tags.items() if isinstance(extra_tags, dict) else []:
                meta[str(k)] = str(v)

        with rasterio.open(path, "w", **profile) as dst:
            for b in range(nc):
                band_arr = self._precomputed[:, :, b]
                write_arr = np.asarray(band_arr, dtype=dtype)
                nan_mask = ~np.isfinite(write_arr)
                if nan_mask.any():
                    write_arr = write_arr.astype(dtype)
                    write_arr[nan_mask] = nodata
                dst.write(write_arr, b + 1)
                try:
                    dst.set_band_description(b + 1, f"coeff_{b}")
                except Exception:
                    pass
                try:
                    dst.update_tags(
                        b + 1,
                        monomial=json.dumps(
                            {"i": int(self.monomials[b][0]), "j": int(self.monomials[b][1])}
                        ),
                    )
                except Exception:
                    pass

            try:
                dst.update_tags(**meta)
                dst.update_tags(
                    description="HO polynomial coefficients (bands correspond to monomials)."
                )
            except Exception:
                pass

            for b in range(nc):
                try:
                    dst.update_tags(b + 1, nodata=str(nodata))
                except Exception:
                    pass

        if self.verbosity:
            print(
                f"[RasterHO] exported precomputed coefficients to {path} (bands={nc}, nodata={nodata})"
            )

    def load_coeffs_geotiff(self, path: str):
        """Load polynomial coefficients from a multiband GeoTIFF into memory.

        Populates self._precomputed and self._precompute_mask accordingly.
        """
        if rasterio is None:
            raise RuntimeError("rasterio is required to load GeoTIFFs (pip install rasterio)")

        with rasterio.open(path, "r") as src:
            nb = src.count
            h = src.height
            w = src.width
            data = src.read().astype(float)  # (nb, h, w)
            tags = src.tags()
            meta_monomials = None
            try:
                if "monomials" in tags:
                    meta_monomials = json.loads(tags["monomials"])
            except Exception:
                meta_monomials = None

            if nb != self.num_coeffs:
                if self.verbosity:
                    print(
                        f"[RasterHO] warning: GeoTIFF has {nb} bands but approximant expects {self.num_coeffs} coefficients."
                    )
                if meta_monomials is not None:
                    try:
                        monoms = meta_monomials
                        self.monomials = [(int(a), int(b)) for (a, b) in monoms]
                        self.num_coeffs = len(self.monomials)
                        self.monomials_i = np.array(
                            [mi for mi, mj in self.monomials], dtype=np.int64
                        )
                        self.monomials_j = np.array(
                            [mj for mi, mj in self.monomials], dtype=np.int64
                        )
                        if self.verbosity:
                            print("[RasterHO] updated monomials from GeoTIFF metadata")
                    except Exception:
                        pass

            coeffs_all = np.transpose(data, (1, 2, 0))
            nodata_vals = []
            for b in range(1, nb + 1):
                try:
                    btags = src.tags(b)
                    if "nodata" in btags:
                        nodata_vals.append(float(btags["nodata"]))
                    else:
                        nodata_vals.append(np.nan)
                except Exception:
                    nodata_vals.append(np.nan)
            nodata_unique = None
            try:
                nset = set([v for v in nodata_vals if not np.isnan(v)])
                if len(nset) == 1:
                    nodata_unique = nset.pop()
            except Exception:
                nodata_unique = None

            if nodata_unique is not None:
                mask = coeffs_all == nodata_unique
                coeffs_all = coeffs_all.astype(float)
                coeffs_all[mask] = np.nan
            else:
                for b in range(nb):
                    nd = nodata_vals[b]
                    if not np.isnan(nd):
                        band_arr = coeffs_all[:, :, b]
                        maskb = band_arr == nd
                        band_arr = band_arr.astype(float)
                        band_arr[maskb] = np.nan
                        coeffs_all[:, :, b] = band_arr

            self._precomputed = coeffs_all.astype(float)
            self._precompute_mask = np.isfinite(self._precomputed).any(axis=2)
            if self.verbosity:
                valid = int(self._precompute_mask.sum())
                print(f"[RasterHO] loaded coefficients from {path}: valid cells {valid}/{h * w}")

        return


# End of file
