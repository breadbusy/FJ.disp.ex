"""FJ Dispersion Extractor (Step 2 GUI).

Load subarray NPZ files, interactively run FJ computation and dispersion curve extraction.
Supports interactive mouse picking on the dispersion energy spectrum:
  - Left-click:  add single dispersion point
  - Right-click: auto-trace ridge from clicked point
  - Dialog popup with frequency range filter and Extract/Keep/Skip options
  - Multi-mode manual & automatic extraction (fundamental / 1st / 2nd ...)
  - Auto-advance to next subarray after confirming extraction

Usage:
    python -m gui.dispersion_extractor
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure ccfj is on Python path
_script_dir = Path(__file__).resolve().parent
_project_dir = _script_dir.parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from matplotlib.backend_bases import MouseEvent

import ccfj

from .common import EmbeddedPlot, FileSelector, ParamGroup, StatusBar, run_in_thread
from .fj_utils import aggregate_fj_dispersion, auto_extract_fj_modes, trace_fj_ridge


# ── Pick Dialog ────────────────────────────────────────

class _PickDialog(tk.Toplevel):
    """Modal dialog that shows a tracked ridge and lets the user choose an action."""

    def __init__(self, parent, freqs, vels, energy, picked_vels, tracked_vels, freq_idx_start, mode_idx=0):
        super().__init__(parent)
        mode_name = {0: "Fundamental", 1: "1st", 2: "2nd", 3: "3rd"}.get(mode_idx, f"Mode {mode_idx}")
        self.title(f"Confirm Dispersion Extraction - {mode_name}")
        self.geometry("420x340")
        self.resizable(False, False)
        self.transient(parent)
        self.wait_visibility()
        self.grab_set()

        self._freqs = freqs
        self._vels = vels
        self._energy = energy
        self._picked_vels = picked_vels    # original auto-search output for all freqs
        self._tracked_vels = list(tracked_vels)  # mutable copy
        self._freq_idx_start = freq_idx_start
        self._mode_idx = mode_idx

        self.result = None  # "extract", "keep", "skip", "cancel"

        self._build_ui()
        # 用实际频率范围初始化频率过滤控件
        self._fmin_var.set(f"{np.min(freqs):.1f}")
        self._fmax_var.set(f"{np.max(freqs):.1f}")
        # 绑定 trace 以响应用户手动输入（Spinbox command 只对箭头按钮有效）
        self._fmin_var.trace_add("write", lambda *a: self._on_filter_changed())
        self._fmax_var.trace_add("write", lambda *a: self._on_filter_changed())
        self._info_label.config(text=f"{len(self._tracked_vels)} points tracked")

    def _build_ui(self):
        # Top: filter controls
        filter_frame = tk.LabelFrame(self, text="Frequency Range Filter", padx=8, pady=5)
        filter_frame.pack(fill=tk.X, padx=8, pady=(8, 4), ipady=3)

        frow = tk.Frame(filter_frame)
        frow.pack(fill=tk.X, pady=2)

        tk.Label(frow, text="F min (Hz):", width=10, anchor="w").pack(side=tk.LEFT)
        self._fmin_var = tk.StringVar(value="0.0")
        tk.Spinbox(frow, textvariable=self._fmin_var, from_=0, to=20, increment=0.1,
                   format="%.1f", width=8, command=self._on_filter_changed).pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(frow, text="F max (Hz):", width=10, anchor="w").pack(side=tk.LEFT)
        self._fmax_var = tk.StringVar(value="20.0")
        tk.Spinbox(frow, textvariable=self._fmax_var, from_=0, to=20, increment=0.1,
                   format="%.1f", width=8, command=self._on_filter_changed).pack(side=tk.LEFT)

        # Smoothing
        srow = tk.Frame(filter_frame)
        srow.pack(fill=tk.X, pady=(4, 0))
        tk.Label(srow, text="Smooth window:", width=13, anchor="w").pack(side=tk.LEFT)
        self._smooth_var = tk.IntVar(value=3)
        tk.Spinbox(srow, textvariable=self._smooth_var, from_=0, to=15, increment=1,
                   width=5, command=self._on_filter_changed).pack(side=tk.LEFT)
        self._smooth_var.trace_add("write", lambda *a: self._on_filter_changed())

        # Info label
        self._info_label = tk.Label(
            self, text=f"{len(self._tracked_vels)} points tracked",
            font=("", 9, "bold"),
        )
        self._info_label.pack(pady=(5, 2))

        # Buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=(8, 8))

        tk.Button(btn_frame, text="Extract & Next", command=lambda: self._done("extract"),
                  bg="#27AE60", fg="white", font=("", 9, "bold"), width=14).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Keep & Continue", command=lambda: self._done("keep"),
                  bg="#2980B9", fg="white", width=14).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Skip", command=lambda: self._done("skip"),
                  bg="#E67E22", fg="white", width=14).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Cancel", command=lambda: self._done("cancel"),
                  width=14).pack(side=tk.LEFT, padx=3)

        self.protocol("WM_DELETE_WINDOW", lambda: self._done("cancel"))
        self.bind("<Escape>", lambda e: self._done("cancel"))

    def _on_filter_changed(self, *args):
        try:
            fmin = float(self._fmin_var.get())
            fmax = float(self._fmax_var.get())
            sw = int(self._smooth_var.get())
            mask = (self._freqs >= fmin) & (self._freqs <= fmax)
            n_in_range = int(np.sum(mask))
            print(f"[DEBUG _on_filter_changed] fmin={fmin}, fmax={fmax}, "
                  f"total freqs={len(self._freqs)}, in_range={n_in_range}, "
                  f"called_from={'spinbox' if args else 'manual'}")
            self._tracked_vels = list(self._picked_vels.copy())
            for i in range(len(self._tracked_vels)):
                if not mask[i]:
                    self._tracked_vels[i] = 0.0
            if sw > 0:
                import scipy.ndimage as ndi
                nonzero = np.array(self._tracked_vels) > 0
                print(f"[DEBUG _on_filter_changed] before median_filter: nonzero={int(np.sum(nonzero))}")
                if np.any(nonzero):
                    self._tracked_vels = ndi.median_filter(self._tracked_vels, size=sw).tolist()
                    new_nonzero = sum(1 for v in self._tracked_vels if v > 0)
                    print(f"[DEBUG _on_filter_changed] after median_filter: nonzero={new_nonzero}")
            n_valid = sum(1 for v in self._tracked_vels if v > 0)
            self._info_label.config(text=f"{n_valid} valid points (filtered)")
            print(f"[DEBUG _on_filter_changed] final valid points: {n_valid}")
        except (ValueError, tk.TclError) as e:
            print(f"[DEBUG _on_filter_changed] ERROR: {e}")
            self._info_label.config(text="Invalid filter values")

    def _done(self, result):
        self.result = result
        self.destroy()

    def _get_filtered_curve(self):
        """返回过滤后的 (freqs, vels) 数组。

        如果没有有效点（所有 tracked_vels <= 0），返回空数组。
        """
        n_total = len(self._tracked_vels)
        valid = np.array(self._tracked_vels) > 0
        n_valid = int(np.sum(valid))
        print(f"[DEBUG _get_filtered_curve] total={n_total}, valid={n_valid}")
        if not np.any(valid):
            print(f"[DEBUG _get_filtered_curve] NO valid points, returning empty")
            return np.array([]), np.array([])
        result_f = self._freqs[valid]
        result_v = np.array(self._tracked_vels)[valid]
        print(f"[DEBUG _get_filtered_curve] returning {len(result_f)} points, "
              f"freqs[{len(self._freqs)}], vels[{n_total}]")
        return result_f, result_v


# ── Main Application ────────────────────────────────────

class DispersionExtractorApp:
    """FJ Dispersion Extractor GUI application."""

    MODE_NAMES = {0: "Fundamental", 1: "1st", 2: "2nd", 3: "3rd",
                  4: "4th", 5: "5th", 6: "6th", 7: "7th"}
    MODE_COLORS = {0: "cyan", 1: "magenta", 2: "orange", 3: "lime",
                   4: "pink", 5: "yellow", 6: "brown", 7: "purple"}

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FJ Dispersion Extractor")
        self.root.geometry("1280x900")
        self.root.minsize(1024, 700)

        # State
        self._metadata: List[Dict] = []
        self._subarray_dir: str = ""
        self._current_idx: int = 0
        self._current_data: Optional[Dict[str, np.ndarray]] = None

        # Results
        self._frequencies: Optional[np.ndarray] = None
        self._velocities: Optional[np.ndarray] = None
        self._energy: Optional[np.ndarray] = None
        self._disp_result: Optional[Dict] = None

        # Manual picks: list of (freq, vel) tuples (current mode temporary picks)
        self._manual_picks: List[Tuple[float, float]] = []

        # Multi-mode state
        self._mode_picks: Dict[int, List] = {}    # {mode_idx: [(f,v),...]}
        self._current_mode: int = 0               # 0=fundamental, 1=1st, ...

        # Batch results
        self._batch_results: List[Dict] = []

        # Processing flag
        self._processing: bool = False

        # Mouse crosshair
        self._crosshair_lines: List = []

        # Pick artist references for redraw
        self._pick_artists: List = []

        self._build_ui()
        self._update_display()

    # ── UI Construction ──────────────────────────────────

    def _build_ui(self):
        # Main PanedWindow: left controls + right plot
        pw = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=3)
        pw.pack(fill=tk.BOTH, expand=True)

        # Left panel (scrollable)
        left_frame = tk.Frame(pw)
        pw.add(left_frame, width=380)

        canvas = tk.Canvas(left_frame, highlightthickness=0)
        scrollbar = tk.Scrollbar(left_frame, orient=tk.VERTICAL, command=canvas.yview)
        self._left_scrollable = tk.Frame(canvas)
        self._left_scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._left_scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # ── Top bar (directory + navigation) ──
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=5, pady=(5, 2))

        dir_frame = tk.Frame(top_frame)
        dir_frame.pack(fill=tk.X)

        default_sub = os.path.join(str(_project_dir), "subarrays")
        if not os.path.isabs(default_sub):
            default_sub = os.path.abspath(default_sub)

        self._dir_selector = FileSelector(
            dir_frame, "Subarray Dir:", default=default_sub, mode="directory",
        )
        self._dir_selector.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        nav_frame = tk.Frame(top_frame)
        nav_frame.pack(fill=tk.X, pady=(2, 0))

        self._load_btn = tk.Button(nav_frame, text="Load", command=self._on_load,
                                   bg="#4A90D9", fg="white", font=("", 9, "bold"))
        self._load_btn.pack(side=tk.LEFT, padx=(0, 5))

        self._prev_btn = tk.Button(nav_frame, text="< Prev", command=self._on_prev,
                                    state=tk.DISABLED)
        self._prev_btn.pack(side=tk.LEFT, padx=1)

        self._subarray_label = tk.Label(nav_frame, text="-", width=16, font=("", 9, "bold"))
        self._subarray_label.pack(side=tk.LEFT, padx=3)

        self._next_btn = tk.Button(nav_frame, text="Next >", command=self._on_next,
                                    state=tk.DISABLED)
        self._next_btn.pack(side=tk.LEFT, padx=1)

        tk.Label(nav_frame, text="Go to:").pack(side=tk.LEFT, padx=(10, 2))
        self._goto_var = tk.StringVar(value="0")
        self._goto_entry = tk.Entry(nav_frame, textvariable=self._goto_var, width=6, state=tk.DISABLED)
        self._goto_entry.pack(side=tk.LEFT, padx=2)
        self._goto_entry.bind("<Return>", lambda e: self._on_goto())

        self._goto_btn = tk.Button(nav_frame, text="Go", command=self._on_goto, state=tk.DISABLED)
        self._goto_btn.pack(side=tk.LEFT)

        # Info label
        self._info_label = tk.Label(nav_frame, text="No data loaded", anchor="w",
                                     font=("", 8), fg="gray")
        self._info_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(15, 0))

        # ── FJ Parameters ──
        self._fj_group = ParamGroup(self._left_scrollable, "FJ Parameters")
        self._fj_group.pack(fill=tk.X, padx=5, pady=3)

        self._fj_group.add_spinbox("freq_low", "Freq Low (Hz):", 0.1, 0.01, 10.0, 0.1, decimal_places=2)
        self._fj_group.add_spinbox("freq_high", "Freq High (Hz):", 3.0, 0.1, 20.0, 0.1, decimal_places=2)
        self._fj_group.add_spinbox("vel_low", "Vel Low (km/s):", 0.5, 0.1, 10.0, 0.1, decimal_places=2)
        self._fj_group.add_spinbox("vel_high", "Vel High (km/s):", 5.0, 0.1, 10.0, 0.1, decimal_places=2)
        self._fj_group.add_spinbox("n_vels", "N Velocity Pts:", 200, 10, 1000, 10, integer=True)
        self._fj_group.add_spinbox("itype", "itype:", 0, 0, 10, 1, integer=True)
        self._fj_group.add_spinbox("func", "func:", 1, 0, 10, 1, integer=True)

        fj_btn_frame = tk.Frame(self._left_scrollable)
        fj_btn_frame.pack(fill=tk.X, padx=5, pady=2)

        self._fj_btn = tk.Button(
            fj_btn_frame, text="Compute FJ", command=self._on_compute_fj,
            bg="#27AE60", fg="white", font=("", 9, "bold"), state=tk.DISABLED,
        )
        self._fj_btn.pack(side=tk.LEFT, padx=(0, 3))

        self._auto_fj_var = tk.BooleanVar(value=False)
        self._auto_fj_cb = tk.Checkbutton(
            fj_btn_frame, text="Auto Compute", variable=self._auto_fj_var,
        )
        self._auto_fj_cb.pack(side=tk.LEFT)

        # ── Mouse Pick Controls ──
        pick_frame = tk.LabelFrame(self._left_scrollable, text="Mouse Pick Controls",
                                   font=("", 10, "bold"), padx=8, pady=5)
        pick_frame.pack(fill=tk.X, padx=5, pady=3)

        tk.Label(
            pick_frame,
            text="Left-click: add point\nRight-click: auto-trace ridge\nMiddle-click: undo last",
            font=("", 8), fg="gray", justify=tk.LEFT,
        ).pack(anchor="w")

        # Mode label
        self._mode_label = tk.Label(
            pick_frame, text="Current Mode: Fundamental",
            font=("", 9, "bold"), fg="#E67E22",
        )
        self._mode_label.pack(anchor="w", pady=(2, 0))

        pick_btn_frame = tk.Frame(pick_frame)
        pick_btn_frame.pack(fill=tk.X, pady=(4, 0))

        self._clear_picks_btn = tk.Button(
            pick_btn_frame, text="Clear Picks", command=self._on_clear_picks,
            font=("", 9), state=tk.DISABLED,
        )
        self._clear_picks_btn.pack(side=tk.LEFT, padx=(0, 3))

        self._undo_btn = tk.Button(
            pick_btn_frame, text="Undo Last", command=self._on_undo_pick,
            font=("", 9), state=tk.DISABLED,
        )
        self._undo_btn.pack(side=tk.LEFT)

        # Mode management button row
        mode_btn_frame = tk.Frame(pick_frame)
        mode_btn_frame.pack(fill=tk.X, pady=(4, 0))

        self._save_mode_btn = tk.Button(
            mode_btn_frame, text="Save Current Mode", command=self._on_save_current_mode,
            font=("", 9), state=tk.DISABLED,
            bg="#E67E22", fg="white",
        )
        self._save_mode_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self._prev_mode_btn = tk.Button(
            mode_btn_frame, text="Prev Mode", command=self._on_prev_mode,
            font=("", 9), state=tk.DISABLED,
        )
        self._prev_mode_btn.pack(side=tk.LEFT, padx=(0, 2))

        self._next_mode_btn = tk.Button(
            mode_btn_frame, text="Next Mode", command=self._on_next_mode,
            font=("", 9), state=tk.DISABLED,
        )
        self._next_mode_btn.pack(side=tk.LEFT)

        # ── Dispersion Parameters ──
        self._disp_group = ParamGroup(self._left_scrollable, "Dispersion Parameters")
        self._disp_group.pack(fill=tk.X, padx=5, pady=3)

        self._disp_group.add_spinbox("min_snr", "Min SNR:", 2.5, 0.1, 100.0, 0.5, decimal_places=1)
        self._disp_group.add_spinbox("max_vel_jump", "Max Vel Jump:", 1.5, 1.0, 10.0, 0.1, decimal_places=1)
        self._disp_group.add_spinbox("n_modes", "N Modes:", 3, 1, 10, 1, integer=True)
        self._disp_group.add_spinbox("min_continuous", "Min Continuous:", 4, 2, 50, 1, integer=True)

        disp_btn_frame = tk.Frame(self._left_scrollable)
        disp_btn_frame.pack(fill=tk.X, padx=5, pady=2)

        self._disp_btn = tk.Button(
            disp_btn_frame, text="Auto Extract", command=self._on_auto_extract,
            bg="#8E44AD", fg="white", font=("", 9, "bold"), state=tk.DISABLED,
        )
        self._disp_btn.pack(side=tk.LEFT, padx=(0, 3))

        self._save_current_picks_btn = tk.Button(
            disp_btn_frame, text="Save Picks", command=self._on_save_picks,
            font=("", 9), state=tk.DISABLED,
        )
        self._save_current_picks_btn.pack(side=tk.LEFT)

        # ── Right Plot ──
        right_frame = tk.Frame(pw)
        pw.add(right_frame, width=800)

        self._plot = EmbeddedPlot(right_frame, figsize=(8, 6), dpi=90)
        self._plot.pack(fill=tk.BOTH, expand=True)
        self._hover_text = self._plot.ax.text(
            0.02, 0.98, "", transform=self._plot.ax.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        self._plot.fig.canvas.mpl_connect("button_press_event", self._on_mouse_click)
        self._plot.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        # ── Bottom bar ──
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=5, pady=(2, 5))

        self._summary_label = tk.Label(bottom_frame, text="Loaded 0 subarrays",
                                        font=("", 9), fg="gray")
        self._summary_label.pack(side=tk.LEFT)

        self._batch_btn = tk.Button(
            bottom_frame, text="Batch All", command=self._on_batch,
            bg="#D35400", fg="white", font=("", 9, "bold"), state=tk.DISABLED,
        )
        self._batch_btn.pack(side=tk.RIGHT, padx=2)

        self._stats_btn = tk.Button(
            bottom_frame, text="Statistics", command=self._on_statistics, state=tk.DISABLED,
        )
        self._stats_btn.pack(side=tk.RIGHT, padx=2)

        self._export_btn = tk.Button(
            bottom_frame, text="Export JSON/NPZ", command=self._on_export, state=tk.DISABLED,
        )
        self._export_btn.pack(side=tk.RIGHT, padx=2)

        # ── Status Bar ──
        self._status = StatusBar(self.root)
        self._status.pack(side=tk.BOTTOM, fill=tk.X)

    # ── Data Loading ──────────────────────────────────────

    def _on_load(self) -> None:
        """Load metadata and prepare data for current subarray."""
        subarray_dir = self._dir_selector.value
        if not subarray_dir or not os.path.isdir(subarray_dir):
            messagebox.showerror("Error", "Please select a valid subarray directory")
            return

        meta_path = os.path.join(subarray_dir, "subarrays_metadata.json")
        if not os.path.isfile(meta_path):
            messagebox.showerror("Error", f"subarrays_metadata.json not found in:\n{subarray_dir}")
            return

        with open(meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        if not self._metadata:
            messagebox.showerror("Error", "No subarrays found in metadata")
            return

        self._subarray_dir = subarray_dir
        self._batch_results = []
        self._current_idx = 0
        self._current_data = None
        self._frequencies = None
        self._velocities = None
        self._energy = None
        self._disp_result = None
        self._manual_picks = []
        self._mode_picks = {}
        self._current_mode = 0

        self._goto_entry.config(state=tk.NORMAL)
        self._goto_btn.config(state=tk.NORMAL)
        self._prev_btn.config(state=tk.NORMAL)
        self._next_btn.config(state=tk.NORMAL)
        self._fj_btn.config(state=tk.NORMAL)
        self._disp_btn.config(state=tk.DISABLED)
        self._save_current_picks_btn.config(state=tk.DISABLED)
        self._clear_picks_btn.config(state=tk.DISABLED)
        self._undo_btn.config(state=tk.DISABLED)
        self._save_mode_btn.config(state=tk.DISABLED)
        self._prev_mode_btn.config(state=tk.DISABLED)
        self._next_mode_btn.config(state=tk.DISABLED)

        self._summary_label.config(text=f"Loaded {len(self._metadata)} subarrays")
        self._batch_btn.config(state=tk.NORMAL)
        self._load_current_subarray()

    def _load_current_subarray(self) -> None:
        """Load NPZ data for the current subarray."""
        if not self._metadata or self._current_idx < 0:
            return

        idx = min(self._current_idx, len(self._metadata) - 1)
        self._current_idx = idx

        npz_path = os.path.join(self._subarray_dir, f"subarray_{idx:03d}.npz")
        if not os.path.isfile(npz_path):
            self._info_label.config(text=f"File not found: subarray_{idx:03d}.npz")
            self._current_data = None
            return

        self._current_data = dict(np.load(npz_path, allow_pickle=True))
        data = self._current_data

        # Update info
        first_sta = str(data.get("first_station", "--"))
        last_sta = str(data.get("last_station", "--"))
        n_pairs = data["ncfs"].shape[0] if "ncfs" in data else 0

        if "r_meters" in data:
            dist_min = np.min(data["r_meters"]) / 1000.0
            dist_max = np.max(data["r_meters"]) / 1000.0
        else:
            dist_min, dist_max = 0, 0

        self._info_label.config(
            text=f"Stations: {first_sta}-{last_sta}    Pairs: {n_pairs}    "
                 f"Dist: {dist_min:.1f}-{dist_max:.1f} km    Picks: 0",
        )

        total = len(self._metadata)
        self._subarray_label.config(text=f"subarray_{idx:03d} / {total:03d}")

        # Reset results and picks
        self._frequencies = None
        self._velocities = None
        self._energy = None
        self._disp_result = None
        self._manual_picks = []
        self._current_mode = 0
        self._mode_picks = {}

        self._update_mode_label()
        self._fj_btn.config(state=tk.NORMAL)
        self._disp_btn.config(state=tk.DISABLED)
        self._save_current_picks_btn.config(state=tk.DISABLED)
        self._clear_picks_btn.config(state=tk.DISABLED)
        self._undo_btn.config(state=tk.DISABLED)
        self._save_mode_btn.config(state=tk.DISABLED)
        self._prev_mode_btn.config(state=tk.DISABLED)
        self._next_mode_btn.config(state=tk.DISABLED)

        self._plot.clear()
        self._plot.ax.set_title(
            'Click "Compute FJ" to start, then left/right-click on the plot',
            fontsize=11,
        )
        self._plot.ax.set_xlabel("Frequency (Hz)")
        self._plot.ax.set_ylabel("Phase Velocity (km/s)")
        self._plot.draw()

        if self._auto_fj_var.get():
            self._on_compute_fj()

    # ── Navigation ───────────────────────────────────────

    def _on_prev(self) -> None:
        if self._current_idx > 0:
            self._current_idx -= 1
            self._load_current_subarray()

    def _on_next(self) -> None:
        if self._current_idx < len(self._metadata) - 1:
            self._current_idx += 1
            self._load_current_subarray()

    def _on_goto(self) -> None:
        try:
            idx = int(self._goto_var.get())
            if 0 <= idx < len(self._metadata):
                self._current_idx = idx
                self._load_current_subarray()
            else:
                messagebox.showwarning("Range Error",
                                       f"Please enter a number between 0 and {len(self._metadata) - 1}")
        except (ValueError, tk.TclError):
            messagebox.showwarning("Format Error", "Please enter an integer index")

    # ── FJ Computation ──────────────────────────────────

    def _on_compute_fj(self) -> None:
        """Compute FJ dispersion energy spectrum for current subarray."""
        if self._current_data is None or self._processing:
            return

        self._processing = True
        self._fj_btn.config(state=tk.DISABLED, text="Computing...")
        self._status.set_text("Computing FJ dispersion spectrum...")

        vel_low_km = self._fj_group.get("vel_low")
        vel_high_km = self._fj_group.get("vel_high")
        n_vels = int(self._fj_group.get("n_vels"))
        freq_low = self._fj_group.get("freq_low")
        freq_high = self._fj_group.get("freq_high")
        itype_val = int(self._fj_group.get("itype"))
        func_val = int(self._fj_group.get("func"))

        vels_grid = np.linspace(vel_low_km * 1000.0, vel_high_km * 1000.0, n_vels)

        def _compute(q):
            data = self._current_data
            ncfs = data["ncfs"]
            r_meters = data["r_meters"]
            freqs_input = data["freqs"]

            # 全频率计算 FJ（避免 Hilbert 变换因截取导致失真）
            energy_full = ccfj.fj_noise(
                np.real(ncfs), r_meters, vels_grid, freqs_input,
                fstride=1, itype=itype_val, func=func_val,
            )
            # 计算后再按需求频率范围裁剪显示
            f_mask = (freqs_input >= freq_low) & (freqs_input <= freq_high)
            f_selected = freqs_input[f_mask]
            energy = energy_full[:, f_mask]
            return vels_grid, f_selected, energy

        def _on_done(result):
            self._velocities, self._frequencies, self._energy = result
            self._disp_result = None
            self._manual_picks = []

            self._draw_energy_plot()
            self._disp_btn.config(state=tk.NORMAL)
            self._clear_picks_btn.config(state=tk.NORMAL)
            self._undo_btn.config(state=tk.NORMAL)
            self._save_current_picks_btn.config(state=tk.NORMAL)
            self._save_mode_btn.config(state=tk.NORMAL)
            self._prev_mode_btn.config(state=tk.NORMAL)
            self._next_mode_btn.config(state=tk.NORMAL)
            self._fj_btn.config(state=tk.NORMAL, text="Compute FJ")
            self._status.set_text(
                f"FJ complete: {len(self._frequencies)} freqs x {len(self._velocities)} vels. "
                f"Left-click to add points, right-click to trace ridge."
            )
            self._processing = False

        def _on_error(exc):
            self._fj_btn.config(state=tk.NORMAL, text="Compute FJ")
            self._status.set_text(f"FJ computation failed: {exc}")
            self._processing = False
            messagebox.showerror("Computation Failed", str(exc))

        run_in_thread(
            self.root,
            lambda q: _compute(q),
            on_done=_on_done,
            on_error=_on_error,
        )

    def _draw_energy_plot(self) -> None:
        """Draw dispersion energy spectrum with multi-mode pick overlays."""
        ax = self._plot.ax
        ax.clear()

        if self._energy is None or self._frequencies is None:
            self._plot.draw()
            return

        # energy shape: (n_vels, n_freqs)
        vels_disp = self._velocities / 1000.0

        X, Y = np.meshgrid(self._frequencies, vels_disp)
        col_max = self._energy.max(axis=1, keepdims=True)
        col_max[col_max == 0] = 1
        energy_norm = self._energy / col_max

        ax.pcolormesh(
            X, Y, energy_norm, cmap="jet", shading="auto",
            vmin=0, vmax=0.95, rasterized=True,
        )

        # Draw saved modes with distinct colors
        for mode_idx, picks in self._mode_picks.items():
            if picks:
                mp = np.array(picks)
                color = self.MODE_COLORS.get(mode_idx, "gray")
                mode_name = self.MODE_NAMES.get(mode_idx, f"Mode {mode_idx}")
                if mp.shape[1] >= 2:
                    ax.scatter(mp[:, 0], mp[:, 1], c=color, s=30,
                               marker="s", edgecolors="black", linewidth=0.5,
                               zorder=6, label=f"{mode_name} ({len(picks)})")

        # Current manual picks (white circles)
        if self._manual_picks:
            mp = np.array(self._manual_picks)
            ax.scatter(mp[:, 0], mp[:, 1], c="white", s=50,
                       edgecolors="black", linewidth=1.5, zorder=7,
                       label=f"Current picks ({len(self._manual_picks)})")

        # Last tracked ridge
        if hasattr(self, '_last_tracked_vels') and self._last_tracked_vels is not None:
            tracked_disp = self._last_tracked_vels / 1000.0
            ax.plot(self._frequencies, tracked_disp,
                    "w--", lw=1.5, alpha=0.6, zorder=5, label="Tracked ridge")

        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_ylabel("Phase Velocity (km/s)", fontsize=10)

        mode_name = self.MODE_NAMES.get(self._current_mode, f"Mode {self._current_mode}")
        title_parts = [f"FJ Dispersion Spectrum - Mode: {mode_name}"]
        if self._manual_picks:
            title_parts.append(f"[{len(self._manual_picks)} picks]")
        ax.set_title("  ".join(title_parts), fontsize=12)

        ax.grid(True, alpha=0.2)
        if len(ax.get_legend_handles_labels()[0]) > 0:
            ax.legend(loc="upper right", fontsize=7)

        self._plot.fig.tight_layout()
        self._plot.draw()

    # ── Mouse Interaction ────────────────────────────────

    def _on_mouse_click(self, event: MouseEvent) -> None:
        """Handle mouse clicks on the energy plot.

        Left-click:  add single dispersion point
        Right-click: auto-trace ridge from clicked point
        Middle-click: undo last pick
        """
        if event.inaxes != self._plot.ax:
            return
        if self._energy is None or self._frequencies is None:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        fi = np.argmin(np.abs(self._frequencies - x))
        vi = np.argmin(np.abs(self._velocities - y * 1000.0))

        f_val = float(self._frequencies[fi])
        v_val_kms = float(self._velocities[vi] / 1000.0)

        if event.button == 1:
            self._manual_picks.append((f_val, v_val_kms))
            self._update_pick_info()
            self._draw_energy_plot()
        elif event.button == 3:
            if not self._manual_picks:
                self._manual_picks.append((f_val, v_val_kms))
            self._on_trace_ridge(vi, fi)
        elif event.button == 2:
            self._on_undo_pick()

    def _on_trace_ridge(self, vel_idx: int, freq_idx: int) -> None:
        """Trace ridge from clicked point, then show confirmation dialog."""
        if self._processing:
            return

        self._status.set_text("Tracing ridge...")
        self._processing = True

        def _trace(q):
            tracked_indices = trace_fj_ridge(
                self._energy, self._velocities,
                int(vel_idx), int(freq_idx),
            )
            n_valid = int(np.sum(tracked_indices >= 0))
            n_total = len(tracked_indices)
            print(f"[DEBUG trace] tracked_indices: total={n_total}, valid={n_valid}")
            print(f"[DEBUG trace] tracked_indices[:5]: {tracked_indices[:5]}")
            print(f"[DEBUG trace] tracked_indices[-5:]: {tracked_indices[-5:]}")
            return tracked_indices

        def _on_done(tracked_indices):
            tracked_vels_ms = self._velocities[tracked_indices]
            print(f"[DEBUG _on_done] tracked_vels_ms.shape={tracked_vels_ms.shape}, "
                  f"min={np.min(tracked_vels_ms):.1f}, max={np.max(tracked_vels_ms):.1f}")
            self._last_tracked_vels = tracked_vels_ms
            self._processing = False

            self._draw_energy_plot()
            self._status.set_text("Ridge traced. Choose action in the dialog.")

            dlg = _PickDialog(
                self.root,
                self._frequencies.copy(),
                self._velocities.copy(),
                self._energy.copy(),
                tracked_vels_ms.copy(),
                tracked_vels_ms.copy(),
                freq_idx,
                mode_idx=self._current_mode,
            )

            # 等待用户关闭对话框
            self.root.wait_window(dlg)
            self._last_tracked_vels = None

            print(f"[DEBUG _on_done] dialog result = {dlg.result!r}")

            if dlg.result == "extract":
                filtered_f, filtered_v = dlg._get_filtered_curve()
                print(f"[DEBUG extract] _get_filtered_curve returned {len(filtered_f)} points")
                n_saved = len(filtered_f)
                self._finalize_picks(filtered_f, filtered_v)
                mode_name = self.MODE_NAMES.get(self._current_mode, f"Mode {self._current_mode}")
                # 自动切换到下一模态
                self._current_mode += 1
                self._manual_picks = list(self._mode_picks.get(self._current_mode, []))
                self._update_mode_label()
                self._update_pick_info()
                self._status.set_text(
                    f"{mode_name}: {n_saved} points saved. "
                    f"Now picking {self.MODE_NAMES.get(self._current_mode, f'Mode {self._current_mode}')}."
                )
            elif dlg.result == "keep":
                filtered_f, filtered_v = dlg._get_filtered_curve()
                print(f"[DEBUG keep] _get_filtered_curve returned {len(filtered_f)} points")
                n_saved = len(filtered_f)
                self._finalize_picks(filtered_f, filtered_v)
                self._status.set_text(
                    f"Mode {self._current_mode}: {n_saved} points saved. "
                    f"Continue picking or switch mode."
                )
            elif dlg.result == "skip":
                self._manual_picks = []
                self._update_pick_info()
                self._status.set_text("Skipped. Continue or navigate manually.")
            else:
                self._draw_energy_plot()
                self._status.set_text("Cancelled. Continue picking.")

            self._draw_energy_plot()

        def _on_error(exc):
            self._processing = False
            self._status.set_text(f"Trace failed: {exc}")
            messagebox.showerror("Trace Failed", str(exc))

        run_in_thread(
            self.root, lambda q: _trace(q),
            on_done=_on_done, on_error=_on_error,
        )

    def _on_next_and_compute(self) -> None:
        """Advance to next subarray and auto-compute FJ."""
        if self._current_idx < len(self._metadata) - 1:
            self._current_idx += 1
            self._load_current_subarray()
            if self._current_data is not None:
                self._on_compute_fj()

    def _finalize_picks(self, freqs: np.ndarray, vels: np.ndarray) -> None:
        """Save current picks to the active mode."""
        print(f"[DEBUG _finalize_picks] mode={self._current_mode}, freqs.shape={freqs.shape}, vels.shape={vels.shape}")
        if len(freqs) == 0:
            print(f"[DEBUG _finalize_picks] empty freqs, returning")
            return

        vels_kms = vels / 1000.0
        picks_list = [(float(freqs[i]), float(vels_kms[i])) for i in range(len(freqs))]
        print(f"[DEBUG _finalize_picks] saving {len(picks_list)} points to mode {self._current_mode}")

        self._mode_picks[self._current_mode] = picks_list
        self._manual_picks = []
        self._update_pick_info()
        self._update_batch_result()

    def _update_batch_result(self) -> None:
        """Update batch results with all modes for the current subarray."""
        modes = {}
        for mi, picks in self._mode_picks.items():
            if picks:
                freqs_arr = np.array([p[0] for p in picks])
                vels_arr = np.array([p[1] for p in picks])
                idx_sort = np.argsort(freqs_arr)
                modes[str(mi)] = {
                    "frequencies": freqs_arr[idx_sort].tolist(),
                    "velocities": vels_arr[idx_sort].tolist(),
                }

        meta = self._metadata[self._current_idx]
        station_ids = meta.get("station_ids", [])
        station_lons = meta.get("station_lons", [])
        station_lats = meta.get("station_lats", [])

        if not station_ids and self._current_data is not None:
            if "station_ids" in self._current_data:
                raw = self._current_data["station_ids"]
                station_ids = raw.tolist() if hasattr(raw, "tolist") else list(raw)
            if "station_lons" in self._current_data:
                raw = self._current_data["station_lons"]
                station_lons = raw.tolist() if hasattr(raw, "tolist") else list(raw)
            if "station_lats" in self._current_data:
                raw = self._current_data["station_lats"]
                station_lats = raw.tolist() if hasattr(raw, "tolist") else list(raw)

        result_entry = {
            "subarray_id": self._current_idx,
            "first_station": meta.get("first_station", ""),
            "last_station": meta.get("last_station", ""),
            "n_stations": meta.get("n_stations", 0),
            "station_ids": station_ids,
            "station_lons": station_lons,
            "station_lats": station_lats,
            "modes": modes,
        }

        existing = [i for i, r in enumerate(self._batch_results)
                     if r.get("subarray_id") == self._current_idx]
        if existing:
            self._batch_results[existing[0]] = result_entry
        else:
            self._batch_results.append(result_entry)

        succeeded = sum(1 for r in self._batch_results if "error" not in r)
        self._stats_btn.config(state=tk.NORMAL if succeeded >= 2 else tk.DISABLED)
        self._export_btn.config(state=tk.NORMAL if succeeded > 0 else tk.DISABLED)
        self._summary_label.config(
            text=f"Loaded {len(self._metadata)} subarrays | Picked {succeeded}/{len(self._metadata)}"
        )
        self._auto_save_results()

    def _auto_save_results(self) -> None:
        """Auto-save batch results to results/ directory."""
        if not self._batch_results or not self._subarray_dir:
            return
        out_dir = os.path.join(self._subarray_dir, "results")
        os.makedirs(out_dir, exist_ok=True)

        for r in self._batch_results:
            sid = r.get("subarray_id", -1)
            npz_path = os.path.join(out_dir, f"subarray_{sid:03d}_fj_result.npz")
            save_dict = {
                "subarray_id": sid,
                "first_station": r.get("first_station", ""),
                "last_station": r.get("last_station", ""),
                "n_stations": int(r.get("n_stations", 0)),
                "station_ids": np.array(r.get("station_ids", [])),
                "station_lons": np.array(r.get("station_lons", [])),
                "station_lats": np.array(r.get("station_lats", [])),
            }
            for mi_str, mode_data in r.get("modes", {}).items():
                save_dict[f"mode_{mi_str}_freqs"] = np.array(mode_data.get("frequencies", []))
                save_dict[f"mode_{mi_str}_vels"] = np.array(mode_data.get("velocities", []))
            np.savez_compressed(npz_path, **save_dict)

        json_path = os.path.join(out_dir, "subarray_results.json")
        serializable = []
        for r in self._batch_results:
            item = {k: v for k, v in r.items() if k != "masw_energy"}
            serializable.append(item)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    def _on_clear_picks(self) -> None:
        self._manual_picks = []
        self._update_pick_info()
        self._draw_energy_plot()

    def _on_undo_pick(self) -> None:
        if self._manual_picks:
            self._manual_picks.pop()
            self._update_pick_info()
            self._draw_energy_plot()

    # ── Mode Management ──────────────────────────────────

    def _update_mode_label(self) -> None:
        mode_name = self.MODE_NAMES.get(self._current_mode, f"Mode {self._current_mode}")
        self._mode_label.config(text=f"Current Mode: {mode_name}")

    def _on_save_current_mode(self) -> None:
        if not self._manual_picks:
            messagebox.showwarning("No Picks", "No picks in current mode.")
            return
        freqs = np.array([p[0] for p in self._manual_picks])
        vels_kms = np.array([p[1] for p in self._manual_picks])
        sort_idx = np.argsort(freqs)
        self._mode_picks[self._current_mode] = [
            (float(freqs[i]), float(vels_kms[i])) for i in sort_idx
        ]
        self._manual_picks = []
        self._update_batch_result()
        self._draw_energy_plot()
        self._status.set_text(
            f"Mode {self._current_mode} saved ({len(self._mode_picks[self._current_mode])} points)"
        )

    def _on_prev_mode(self) -> None:
        if self._current_mode > 0:
            if self._manual_picks:
                freqs = np.array([p[0] for p in self._manual_picks])
                vels_kms = np.array([p[1] for p in self._manual_picks])
                sort_idx = np.argsort(freqs)
                self._mode_picks[self._current_mode] = [
                    (float(freqs[i]), float(vels_kms[i])) for i in sort_idx
                ]
            self._current_mode -= 1
            self._manual_picks = list(self._mode_picks.get(self._current_mode, []))
            self._update_mode_label()
            self._update_pick_info()
            self._draw_energy_plot()

    def _on_next_mode(self) -> None:
        if self._manual_picks:
            freqs = np.array([p[0] for p in self._manual_picks])
            vels_kms = np.array([p[1] for p in self._manual_picks])
            sort_idx = np.argsort(freqs)
            self._mode_picks[self._current_mode] = [
                (float(freqs[i]), float(vels_kms[i])) for i in sort_idx
            ]
        self._current_mode += 1
        self._manual_picks = list(self._mode_picks.get(self._current_mode, []))
        self._update_mode_label()
        self._update_pick_info()
        self._draw_energy_plot()

    def _on_save_picks(self) -> None:
        if not self._manual_picks:
            messagebox.showwarning("No Picks", "Left-click on the plot to add picks first.")
            return

        freqs = np.array([p[0] for p in self._manual_picks])
        vels_kms = np.array([p[1] for p in self._manual_picks])
        sort_idx = np.argsort(freqs)
        freqs = freqs[sort_idx]
        vels_kms = vels_kms[sort_idx]

        self._finalize_picks(freqs, vels_kms * 1000.0)
        self._status.set_text(f"Saved {len(freqs)} picks. Continue or switch mode.")

    def _on_auto_extract(self) -> None:
        """Auto-extract multi-mode dispersion curves."""
        if self._energy is None or self._processing:
            return

        self._processing = True
        self._disp_btn.config(state=tk.DISABLED, text="Extracting...")
        self._status.set_text("Auto-extracting multi-mode dispersion...")

        min_snr = self._disp_group.get("min_snr")
        max_vel_jump = self._disp_group.get("max_vel_jump")
        n_modes = int(self._disp_group.get("n_modes"))
        min_continuous = int(self._disp_group.get("min_continuous"))

        def _extract(q):
            modes = auto_extract_fj_modes(
                self._energy.T, self._frequencies, self._velocities,
                min_snr=min_snr, max_vel_jump=max_vel_jump,
                n_modes=n_modes, min_continuous=min_continuous,
            )
            return modes

        def _on_done(modes):
            self._mode_picks = {}
            for m in modes:
                mi = m["mode_index"]
                vels_kms = m["velocities"] / 1000.0
                self._mode_picks[mi] = [
                    (float(m["frequencies"][i]), float(vels_kms[i]))
                    for i in range(len(m["frequencies"]))
                ]

            self._manual_picks = []
            self._current_mode = 0
            if 0 in self._mode_picks:
                self._manual_picks = list(self._mode_picks[0])
            self._update_mode_label()
            self._update_batch_result()
            self._draw_energy_plot()

            self._disp_btn.config(state=tk.NORMAL, text="Auto Extract")
            self._save_current_picks_btn.config(state=tk.NORMAL)
            self._save_mode_btn.config(state=tk.NORMAL)
            self._prev_mode_btn.config(state=tk.NORMAL)
            self._next_mode_btn.config(state=tk.NORMAL)
            self._status.set_text(
                f"Auto-extracted: {len(modes)} modes, "
                + ", ".join(f"M{mi}: {len(self._mode_picks[mi])}pts"
                            for mi in sorted(self._mode_picks.keys()))
            )
            self._processing = False

        def _on_error(exc):
            self._disp_btn.config(state=tk.NORMAL, text="Auto Extract")
            self._status.set_text(f"Extraction failed: {exc}")
            self._processing = False
            messagebox.showerror("Extraction Failed", str(exc))

        run_in_thread(
            self.root, lambda q: _extract(q),
            on_done=_on_done, on_error=_on_error,
        )

    def _update_pick_info(self) -> None:
        info = self._info_label.cget("text")
        # 优先显示当前手动拾取点数，若为空则汇总所有已保存模态
        n_picks = len(self._manual_picks)
        if n_picks == 0 and self._mode_picks:
            n_picks = sum(len(p) for p in self._mode_picks.values())
        if "Picks:" in info:
            info = info.rsplit("Picks:", 1)[0] + f"Picks: {n_picks}"
        else:
            info += f"    Picks: {n_picks}"
        self._info_label.config(text=info)

    # ── Mouse hover crosshair ────────────────────────────

    def _on_mouse_move(self, event: MouseEvent) -> None:
        """Show (f, v, energy) values on mouse hover."""
        if event.inaxes != self._plot.ax:
            self._hover_text.set_text("")
            for line in self._crosshair_lines:
                line.remove()
            self._crosshair_lines.clear()
            self._plot.draw()
            return

        if self._energy is None or self._frequencies is None or self._velocities is None:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        fi = np.argmin(np.abs(self._frequencies - x))
        vi = np.argmin(np.abs(self._velocities - y * 1000.0))

        f_val = self._frequencies[fi]
        v_val = self._velocities[vi] / 1000.0
        e_val = self._energy[vi, fi]  # shape: (n_vels, n_freqs)

        self._hover_text.set_text(
            f"f = {f_val:.3f} Hz\nv = {v_val:.2f} km/s\nE = {e_val:.3f}",
        )

        for line in self._crosshair_lines:
            line.remove()
        self._crosshair_lines.clear()

        line_h = self._plot.ax.axhline(y=v_val, color="gray", ls="--", lw=0.5, alpha=0.5)
        line_v = self._plot.ax.axvline(x=f_val, color="gray", ls="--", lw=0.5, alpha=0.5)
        self._crosshair_lines.extend([line_h, line_v])

        self._plot.draw()

    # ── Batch Processing ─────────────────────────────────

    def _on_batch(self) -> None:
        """Batch auto-extract all subarrays."""
        if self._processing or not self._metadata:
            return

        if self._batch_results:
            if not messagebox.askyesno("Confirm",
                                       f"Already have {len(self._batch_results)} results. Reprocess?"):
                return

        self._processing = True
        self._batch_btn.config(state=tk.DISABLED, text="Processing...")
        self._status.set_text("Batch processing subarrays...")
        self._status.show_progress()
        self._status.set_progress(0, len(self._metadata))

        vel_low_km = self._fj_group.get("vel_low")
        vel_high_km = self._fj_group.get("vel_high")
        n_vels = int(self._fj_group.get("n_vels"))
        freq_low = self._fj_group.get("freq_low")
        freq_high = self._fj_group.get("freq_high")
        itype_val = int(self._fj_group.get("itype"))
        func_val = int(self._fj_group.get("func"))
        min_snr = self._disp_group.get("min_snr")
        max_vel_jump = self._disp_group.get("max_vel_jump")
        n_modes = int(self._disp_group.get("n_modes"))
        min_continuous = int(self._disp_group.get("min_continuous"))
        subarray_dir = self._subarray_dir

        vels_grid = np.linspace(vel_low_km * 1000.0, vel_high_km * 1000.0, n_vels)

        def _batch_worker(q):
            results = []
            for idx in range(len(self._metadata)):
                if not self._processing:
                    break

                meta = self._metadata[idx]
                npz_path = os.path.join(subarray_dir, f"subarray_{idx:03d}.npz")
                if not os.path.isfile(npz_path):
                    results.append({
                        "subarray_id": idx,
                        "first_station": meta.get("first_station", ""),
                        "last_station": meta.get("last_station", ""),
                        "error": "NPZ file not found",
                    })
                    q.put(("progress", idx + 1))
                    continue

                try:
                    data = dict(np.load(npz_path, allow_pickle=True))
                    ncfs = data["ncfs"]
                    r_meters = data["r_meters"]
                    freqs_input = data["freqs"]

                    # 全频率计算 FJ（避免 Hilbert 变换因截取导致失真）
                    energy_full = ccfj.fj_noise(
                        np.real(ncfs), r_meters, vels_grid, freqs_input,
                        fstride=1, itype=itype_val, func=func_val,
                    )
                    # 计算后再按需求频率范围裁剪
                    f_mask = (freqs_input >= freq_low) & (freqs_input <= freq_high)
                    f_selected = freqs_input[f_mask]
                    energy = energy_full[:, f_mask]

                    # Save spectrum
                    spec_dir = os.path.join(subarray_dir, "results", "spectra")
                    os.makedirs(spec_dir, exist_ok=True)
                    spec_path = os.path.join(spec_dir, f"subarray_{idx:03d}_fj_spectrum.npz")
                    np.savez_compressed(spec_path, energy=energy, frequencies=f_selected, velocities=vels_grid)

                    # Auto-extract multi-mode
                    modes = auto_extract_fj_modes(
                        energy.T, f_selected, vels_grid,
                        min_snr=min_snr, max_vel_jump=max_vel_jump,
                        n_modes=n_modes, min_continuous=min_continuous,
                    )

                    modes_dict = {}
                    for m in modes:
                        mi = m["mode_index"]
                        vels_kms = m["velocities"] / 1000.0
                        modes_dict[str(mi)] = {
                            "frequencies": m["frequencies"].tolist(),
                            "velocities": vels_kms.tolist(),
                        }

                    station_ids_raw = data.get("station_ids", [])
                    station_ids = station_ids_raw.tolist() if hasattr(station_ids_raw, "tolist") else list(station_ids_raw)
                    station_lons_raw = data.get("station_lons", [])
                    station_lons = station_lons_raw.tolist() if hasattr(station_lons_raw, "tolist") else list(station_lons_raw)
                    station_lats_raw = data.get("station_lats", [])
                    station_lats = station_lats_raw.tolist() if hasattr(station_lats_raw, "tolist") else list(station_lats_raw)

                    results.append({
                        "subarray_id": idx,
                        "first_station": meta.get("first_station", ""),
                        "last_station": meta.get("last_station", ""),
                        "n_stations": meta.get("n_stations", 0),
                        "station_ids": station_ids,
                        "station_lons": station_lons,
                        "station_lats": station_lats,
                        "modes": modes_dict,
                    })
                except Exception as exc:
                    results.append({
                        "subarray_id": idx,
                        "first_station": meta.get("first_station", ""),
                        "last_station": meta.get("last_station", ""),
                        "error": str(exc),
                    })

                q.put(("progress", idx + 1))
                q.put(("text", f"Batch: {idx + 1}/{len(self._metadata)}"))

            return results

        self._batch_queue = queue.Queue()

        def _check_batch_progress():
            try:
                while True:
                    msg_type, msg_data = self._batch_queue.get_nowait()
                    if msg_type == "progress":
                        self._status.set_progress(msg_data, len(self._metadata))
                    elif msg_type == "text":
                        self._status.set_text(msg_data)
                    self._batch_queue.task_done()
            except queue.Empty:
                pass

            if self._batch_thread.is_alive():
                self.root.after(100, _check_batch_progress)
            else:
                self._on_batch_done()

        def _batch_runner():
            try:
                result = _batch_worker(self._batch_queue)
                self._batch_queue.put(("done", result))
            except Exception as exc:
                self._batch_queue.put(("error", str(exc)))

        self._batch_thread = threading.Thread(target=_batch_runner, daemon=True)
        self._batch_thread.start()
        self.root.after(100, _check_batch_progress)

    def _on_batch_done(self) -> None:
        """Batch completion callback."""
        while True:
            try:
                msg_type, msg_data = self._batch_queue.get_nowait()
                if msg_type == "done":
                    self._batch_results = msg_data
                    succeeded = sum(1 for r in msg_data if "error" not in r)
                    total_modes = sum(len(r.get("modes", {})) for r in msg_data if "error" not in r)
                    self._summary_label.config(
                        text=f"Loaded {len(self._metadata)} subarrays | "
                             f"Complete {succeeded}/{len(self._metadata)} | "
                             f"Total modes: {total_modes}",
                    )
                    self._stats_btn.config(state=tk.NORMAL if succeeded >= 2 else tk.DISABLED)
                    self._export_btn.config(state=tk.NORMAL if succeeded > 0 else tk.DISABLED)
                    self._status.set_text(f"Batch complete: {succeeded}/{len(self._metadata)} succeeded")
                elif msg_type == "error":
                    self._status.set_text(f"Batch failed: {msg_data}")
                    messagebox.showerror("Batch Failed", msg_data)
                self._batch_queue.task_done()
            except queue.Empty:
                break

        self._status.hide_progress()
        self._processing = False
        self._batch_btn.config(state=tk.NORMAL, text="Batch All")

    # ── Save / Export ────────────────────────────────────

    def _on_statistics(self) -> None:
        """Show multi-mode statistics window."""
        if len(self._batch_results) < 2:
            messagebox.showwarning("Insufficient Data", "Need at least 2 valid results for statistics")
            return

        stats = aggregate_fj_dispersion(self._batch_results)

        stat_win = tk.Toplevel(self.root)
        stat_win.title("Subarray Dispersion Statistics - FJ Multi-Mode")
        stat_win.geometry("1200x600")

        top_frame = tk.Frame(stat_win, padx=5, pady=3)
        top_frame.pack(fill=tk.X)
        tk.Label(top_frame, text="Mode:").pack(side=tk.LEFT)
        mode_var = tk.IntVar(value=0)

        plot = EmbeddedPlot(stat_win, figsize=(12, 5), dpi=90)
        plot.pack(fill=tk.BOTH, expand=True)

        def _draw_mode(mode_idx):
            mode_stats = stats.get(mode_idx, {})
            freqs = mode_stats.get("mean_frequencies", np.array([]))
            mean_vels = mode_stats.get("mean_velocities", np.array([]))
            std_vels = mode_stats.get("std_velocities", np.array([]))
            cov = mode_stats.get("cov", np.array([]))
            n_ind = mode_stats.get("count", 0)

            plot.fig.clf()
            if len(freqs) == 0 or n_ind == 0:
                ax = plot.fig.add_subplot(111)
                mode_name = self.MODE_NAMES.get(mode_idx, f"Mode {mode_idx}")
                ax.set_title(f"{mode_name} - No data")
                plot.draw()
                return

            ax1 = plot.fig.add_subplot(121)
            ax2 = plot.fig.add_subplot(122)

            mode_name = self.MODE_NAMES.get(mode_idx, f"Mode {mode_idx}")

            ax1.fill_between(
                freqs, mean_vels - std_vels, mean_vels + std_vels,
                alpha=0.3, color="blue", label="+/- 1 sigma",
            )
            ax1.plot(freqs, mean_vels, "b-", lw=2, label="Mean dispersion")

            individuals = mode_stats.get("individual", [])
            for r in individuals:
                if len(r.get("frequencies", [])) > 0:
                    ax1.plot(r["frequencies"], r["velocities"], "gray", alpha=0.2, lw=0.5)

            ax1.set_xlabel("Frequency (Hz)", fontsize=10)
            ax1.set_ylabel("Phase Velocity (km/s)", fontsize=10)
            ax1.set_title(f"{mode_name} Mean +/- 1 sigma (n={n_ind})", fontsize=11)
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

            ax2.plot(freqs, cov * 100, "r.-", lw=1.5, ms=6)
            ax2.axhline(y=5.0, color="gray", ls="--", alpha=0.5, label="5% COV")
            ax2.axhline(y=10.0, color="gray", ls=":", alpha=0.5, label="10% COV")
            ax2.set_xlabel("Frequency (Hz)", fontsize=10)
            ax2.set_ylabel("COV (%)", fontsize=10)
            ax2.set_title(f"{mode_name} Dispersion Uncertainty", fontsize=11)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

            plot.fig.tight_layout()
            plot.draw()

        def _mode_spin_changed(*args):
            try:
                mi = mode_var.get()
                _draw_mode(mi)
            except tk.TclError:
                pass

        mode_spin = tk.Spinbox(top_frame, textvariable=mode_var, from_=0, to=10,
                               width=5, command=_mode_spin_changed)
        mode_spin.pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Refresh", command=_mode_spin_changed).pack(side=tk.LEFT, padx=5)

        _draw_mode(0)

    def _on_export(self) -> None:
        """Export all batch results as JSON/NPZ with station lat/lon."""
        if not self._batch_results or not self._subarray_dir:
            return

        out_dir = os.path.join(self._subarray_dir, "results")
        os.makedirs(out_dir, exist_ok=True)

        for r in self._batch_results:
            if "error" in r:
                continue
            sid = r.get("subarray_id", -1)
            npz_path = os.path.join(out_dir, f"subarray_{sid:03d}_fj_result.npz")
            save_dict = {
                "subarray_id": sid,
                "first_station": r.get("first_station", ""),
                "last_station": r.get("last_station", ""),
                "n_stations": int(r.get("n_stations", 0)),
                "station_ids": np.array(r.get("station_ids", [])),
                "station_lons": np.array(r.get("station_lons", [])),
                "station_lats": np.array(r.get("station_lats", [])),
            }
            for mi_str, mode_data in r.get("modes", {}).items():
                save_dict[f"mode_{mi_str}_freqs"] = np.array(mode_data.get("frequencies", []))
                save_dict[f"mode_{mi_str}_vels"] = np.array(mode_data.get("velocities", []))
            np.savez_compressed(npz_path, **save_dict)

        json_path = os.path.join(out_dir, "subarray_results.json")
        serializable = []
        for r in self._batch_results:
            item = {k: v for k, v in r.items() if k != "masw_energy"}
            serializable.append(item)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        # Statistics export
        valid_results = [r for r in self._batch_results if "error" not in r]
        if len(valid_results) >= 2:
            stats = aggregate_fj_dispersion(self._batch_results)
            for mi, mode_stats in stats.items():
                if mode_stats.get("count", 0) >= 2:
                    npz_path = os.path.join(out_dir, f"statistics_mode_{mi}.npz")
                    np.savez_compressed(
                        npz_path,
                        mean_frequencies=mode_stats["mean_frequencies"],
                        mean_velocities=mode_stats["mean_velocities"],
                        std_velocities=mode_stats["std_velocities"],
                        cov=mode_stats["cov"],
                        count=mode_stats["count"],
                        mode_index=mi,
                    )

        self._status.set_text(f"Exported to: {out_dir}/")
        messagebox.showinfo("Export Complete", f"Results saved to:\n{out_dir}/")

    # ── Display Update ───────────────────────────────────

    def _update_display(self) -> None:
        if not self._metadata:
            self._prev_btn.config(state=tk.DISABLED)
            self._next_btn.config(state=tk.DISABLED)
            self._subarray_label.config(text="No subarrays")
            self._summary_label.config(text="Loaded 0 subarrays")
            self._batch_btn.config(state=tk.DISABLED)
            self._stats_btn.config(state=tk.DISABLED)
            self._export_btn.config(state=tk.DISABLED)
            self._fj_btn.config(state=tk.DISABLED)
            self._disp_btn.config(state=tk.DISABLED)
            self._clear_picks_btn.config(state=tk.DISABLED)
            self._undo_btn.config(state=tk.DISABLED)
            self._save_mode_btn.config(state=tk.DISABLED)
            self._prev_mode_btn.config(state=tk.DISABLED)
            self._next_mode_btn.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = DispersionExtractorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
