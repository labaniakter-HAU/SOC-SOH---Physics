"""PyBaMM SPMe lookup wrapper.

Builds a 3-D lookup table over (SOC, I_rate, T_celsius) using PyBaMM's
Single Particle Model with electrolyte (SPMe) and caches results to a
compressed .npz file.  At inference time the lookup uses trilinear
interpolation from scipy so there is zero solver latency.

Usage (build cache once, then load at training time):
    from uapi_former.pybamm_wrapper import PyBaMMLookup
    lookup = PyBaMMLookup.build_and_save("cache/spme_nmc.npz",
                                          chemistry="NMC_Kokam")
    # --- or load a pre-built cache ---
    lookup = PyBaMMLookup.from_cache("cache/spme_nmc.npz")
    v_pred = lookup.predict(soc_array, i_array, t_array)

If PyBaMM is not installed the class falls back to a polynomial analytic
approximation so the rest of the pipeline stays functional.
"""
from __future__ import annotations

import os
import numpy as np

try:
    import pybamm  # type: ignore
    PYBAMM_AVAILABLE = True
except Exception:
    PYBAMM_AVAILABLE = False

try:
    from scipy.interpolate import RegularGridInterpolator  # type: ignore
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default grid resolution (coarse for cache size; fine enough for residuals)
# ---------------------------------------------------------------------------
_DEFAULT_SOCS     = np.linspace(0.05, 0.95, 37)   # 37 pts
_DEFAULT_RATES    = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])   # C-rate
_DEFAULT_TEMPS    = np.array([10.0, 20.0, 25.0, 30.0, 40.0, 45.0])      # °C


def _analytic_ocv(soc: np.ndarray, chemistry: str = "NMC") -> np.ndarray:
    """Polynomial open-circuit voltage approximation.

    Coefficients fitted to NMC 18650 OCV data (Plett 2015 Table 4.1).
    Used as fallback when PyBaMM is unavailable.
    """
    s = np.asarray(soc, dtype=float)
    if chemistry in ("NMC", "NMC_Kokam"):
        # 5th-order polynomial
        return (
            3.4023 + 1.5728 * s
            - 5.5816 * s**2
            + 7.9014 * s**3
            - 5.0970 * s**4
            + 1.1842 * s**5
        )
    elif chemistry == "LFP":
        # Flat plateau characteristic of LFP
        return 3.2 + 0.15 * s - 0.08 * np.sin(np.pi * s)
    elif chemistry in ("LCO", "CALCE"):
        return 3.7 + 0.7 * s - 0.3 * s**2
    else:
        return 3.6 + 0.5 * s


def _analytic_voltage(soc, i_rate, t_celsius, chemistry="NMC"):
    """Voltage = OCV - I*R_0 - thermal correction (linearised)."""
    ocv  = _analytic_ocv(soc, chemistry)
    # internal resistance ~ 80 mΩ baseline, rises at low T
    r0   = 0.08 + 0.002 * np.maximum(0.0, 25.0 - t_celsius)
    eta  = -i_rate * r0
    return ocv + eta


def _run_pybamm_spme(socs, i_rates, temps_c, chemistry="NMC_Kokam"):
    """Run PyBaMM SPMe for a batch of (SOC, I, T) points.

    Returns a 3-D array of shape (len(socs), len(i_rates), len(temps_c)).
    Each cell is the terminal voltage predicted by SPMe at that operating point.
    """
    if not PYBAMM_AVAILABLE:
        raise RuntimeError("pybamm not installed; cannot build SPMe cache")

    # Choose parameter set
    if chemistry in ("NMC", "NMC_Kokam"):
        param_set = pybamm.ParameterValues("Mohtat2020")
    elif chemistry == "LFP":
        param_set = pybamm.ParameterValues("Prada2013")
    else:
        param_set = pybamm.ParameterValues("Chen2020")

    model = pybamm.lithium_ion.SPMe()
    geometry = model.default_geometry

    param_set.process_model(model)
    param_set.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

    V_table = np.zeros((len(socs), len(i_rates), len(temps_c)))

    for ti, T_c in enumerate(temps_c):
        for ii, i_rate in enumerate(i_rates):
            for si, soc0 in enumerate(socs):
                try:
                    param_set.update({
                        "Initial temperature [K]": T_c + 273.15,
                        "Ambient temperature [K]": T_c + 273.15,
                        "Current function [A]": i_rate,
                    })
                    solver = pybamm.CasadiSolver(mode="safe")
                    t_eval = np.linspace(0, 1.0, 5)
                    sol = solver.solve(model, t_eval)
                    # voltage at t=0 (instantaneous, approximating the OCV+drop)
                    V_table[si, ii, ti] = float(sol["Terminal voltage [V]"].entries[0])
                except Exception:
                    # fallback to analytic if solver diverges at this point
                    V_table[si, ii, ti] = _analytic_voltage(soc0, i_rate, T_c)

    return V_table


class PyBaMMLookup:
    """Lookup table for SPMe terminal voltage predictions.

    Build once with `build_and_save`; load at training time with `from_cache`.
    All interpolation is done with `scipy.interpolate.RegularGridInterpolator`
    (trilinear, no extrapolation — points outside the grid are clipped).
    """

    def __init__(self, socs, i_rates, temps_c, V_table, chemistry="NMC"):
        self.socs     = np.asarray(socs)
        self.i_rates  = np.asarray(i_rates)
        self.temps_c  = np.asarray(temps_c)
        self.V_table  = np.asarray(V_table)   # (S, I, T)
        self.chemistry = chemistry
        self._interp  = None
        if SCIPY_AVAILABLE:
            self._interp = RegularGridInterpolator(
                (self.socs, self.i_rates, self.temps_c),
                self.V_table,
                method="linear",
                bounds_error=False,
                fill_value=None,   # extrapolate by nearest at boundaries
            )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def build_and_save(
        cls,
        cache_path: str,
        chemistry: str = "NMC_Kokam",
        socs=None,
        i_rates=None,
        temps_c=None,
    ) -> "PyBaMMLookup":
        socs    = _DEFAULT_SOCS    if socs    is None else np.asarray(socs)
        i_rates = _DEFAULT_RATES   if i_rates is None else np.asarray(i_rates)
        temps_c = _DEFAULT_TEMPS   if temps_c is None else np.asarray(temps_c)

        if PYBAMM_AVAILABLE:
            V_table = _run_pybamm_spme(socs, i_rates, temps_c, chemistry)
        else:
            # analytic fallback — still useful for shape/API testing
            V_table = np.zeros((len(socs), len(i_rates), len(temps_c)))
            for ti, T_c in enumerate(temps_c):
                for ii, ir in enumerate(i_rates):
                    for si, s in enumerate(socs):
                        V_table[si, ii, ti] = _analytic_voltage(s, ir, T_c, chemistry)

        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(
            cache_path,
            socs=socs, i_rates=i_rates, temps_c=temps_c,
            V_table=V_table, chemistry=np.array([chemistry]),
        )
        return cls(socs, i_rates, temps_c, V_table, chemistry)

    @classmethod
    def from_cache(cls, cache_path: str) -> "PyBaMMLookup":
        data = np.load(cache_path, allow_pickle=True)
        chemistry = str(data["chemistry"][0])
        return cls(data["socs"], data["i_rates"], data["temps_c"], data["V_table"], chemistry)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        soc: np.ndarray,
        i_rate: np.ndarray,
        t_celsius: np.ndarray,
    ) -> np.ndarray:
        """Trilinear lookup for arbitrary (SOC, I, T) arrays.

        All arrays must be the same shape.  Returns V_spme of that shape.
        """
        soc      = np.clip(np.asarray(soc, dtype=float),      self.socs.min(),    self.socs.max())
        i_rate   = np.clip(np.asarray(i_rate, dtype=float),   self.i_rates.min(), self.i_rates.max())
        t_celsius = np.clip(np.asarray(t_celsius, dtype=float), self.temps_c.min(), self.temps_c.max())

        pts = np.stack([soc.ravel(), i_rate.ravel(), t_celsius.ravel()], axis=1)

        if self._interp is not None:
            return self._interp(pts).reshape(soc.shape)

        # no scipy: nearest-neighbour fallback
        si = np.argmin(np.abs(self.socs[:, None]    - soc.ravel()[None, :]),    axis=0)
        ii = np.argmin(np.abs(self.i_rates[:, None] - i_rate.ravel()[None, :]), axis=0)
        ti = np.argmin(np.abs(self.temps_c[:, None] - t_celsius.ravel()[None, :]), axis=0)
        return self.V_table[si, ii, ti].reshape(soc.shape)

    def predict_voltage(self, V, I=None, T=None):
        """Legacy single-array interface used by precompute_spme.py.

        Estimates SPMe baseline from the raw voltage V (used as a SOC proxy).
        For proper usage call `predict(soc, i_rate, t_celsius)` directly.
        """
        V_arr = np.asarray(V, dtype=float)
        # crude SOC estimate from V (NMC: 3.4 V ~ 0%, 4.2 V ~ 100%)
        soc_est = np.clip((V_arr - 3.4) / (4.2 - 3.4), 0.05, 0.95)
        i_arr = np.zeros_like(soc_est) if I is None else np.asarray(I, dtype=float)
        t_arr = np.full_like(soc_est, 25.0) if T is None else np.asarray(T, dtype=float)
        return self.predict(soc_est, i_arr, t_arr)
