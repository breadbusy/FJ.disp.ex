"""FJ Subarray Splitter (Step 1 GUI).

Load cross-correlation SAC data, split stations into sliding-window subarrays,
and export each subarray as an independent NPZ file (frequency-domain ncfs + station lat/lon).

Usage:
    python -m gui.subarray_splitter
"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import tkinter as tk
from glob import glob
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, List, Optional, Tuple

import numpy as np

from .common import EmbeddedPlot, FileSelector, ParamGroup, StatusBar, run_in_thread


# ── Station list loading ────────────────────────────

def _load_stalist(stalist_file: str) -> Tuple[List[str], List[float], List[float]]:
    """Load station list file, return (ids, lons, lats)."""
    stas, lons, lats = [], [], []
    with open(stalist_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                stas.append(parts[0])
                lons.append(float(parts[1]))
                lats.append(float(parts[2]))
    return stas, lons, lats


# ── Subarray splitting ──────────────────────────────

def _create_subarrays(
    stations: List[Dict],
    window_size: int,
    stride: int = 1,
) -> List[Dict]:
    """Generate subarrays via sliding window.

    Each subarray contains:
        indices: station indices in the stations list
        stations: list of station dicts
        size: number of stations
        first_id: first station ID
        last_id: last station ID
    """
    n = len(stations)
    actual_window = min(window_size, n)
    subarrays = []
    for start in range(0, n - actual_window + 1, stride):
        indices = list(range(start, start + actual_window))
        sa_stations = [stations[i] for i in indices]
        subarrays.append({
            "indices": indices,
            "stations": sa_stations,
            "size": actual_window,
            "first_id": stations[start]["id"],
            "last_id": stations[start + actual_window - 1]["id"],
        })
    return subarrays


class SubarraySplitterApp:
    """Subarray splitting GUI application."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FJ Subarray Splitter")
        self.root.geometry("1024x768")
        self.root.minsize(800, 600)

        # State
        self._all_ncfs: Optional[np.ndarray] = None      # (n_pairs, n_freqs)
        self._all_distances: Optional[np.ndarray] = None  # (n_pairs,) km
        self._all_stations: List[Dict] = []               # [{id, lon, lat}, ...]
        self._station_indices: Dict[str, int] = {}        # id -> index
        self._sr: float = 0.0
        self._n_subarrays: int = 0
        self._processing: bool = False

        # Default paths
        self._project_dir = Path(__file__).resolve().parent.parent
        self._default_sac = str(self._project_dir / "cross_correlation")
        self._default_stalist = str(self._project_dir / "chishui2.txt")
        self._default_output = str(self._project_dir / "subarrays")

        self._build_ui()
        self._update_subarray_count()

    # ── UI Construction ──────────────────────────────

    def _build_ui(self) -> None:
        self._pw = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=3)
        self._pw.pack(fill=tk.BOTH, expand=True)

        # Left panel (scrollable)
        left_frame = tk.Frame(self._pw)
        self._pw.add(left_frame, width=380)

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

        # ── Data Source ──
        self._data_group = ParamGroup(self._left_scrollable, "Data Source")
        self._data_group.pack(fill=tk.X, padx=5, pady=3)

        self._data_group.add_file_selector("sac_dir", "SAC Dir:",
                                           default=self._default_sac, mode="directory")
        self._data_group.add_file_selector("stalist", "Station List:",
                                           default=self._default_stalist, mode="file",
                                           filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        self._data_group.add_file_selector("output_dir", "Output Dir:",
                                           default=self._default_output, mode="directory")

        # ── Preprocess Parameters ──
        self._preproc_group = ParamGroup(self._left_scrollable, "Preprocess Parameters")
        self._preproc_group.pack(fill=tk.X, padx=5, pady=3)
        self._preproc_group.add_spinbox("time_half", "Time Window Half (s):", 40.0, 1.0, 200.0, 5.0, decimal_places=1)
        self._preproc_group.on_change(lambda _k, _v: None)

        # ── Subarray Parameters ──
        self._subarray_group = ParamGroup(self._left_scrollable, "Subarray Parameters")
        self._subarray_group.pack(fill=tk.X, padx=5, pady=3)
        self._subarray_group.add_spinbox("window_size", "Window Size:", 12, 2, 200, 1, integer=True)
        self._subarray_group.add_spinbox("stride", "Stride:", 1, 1, 50, 1, integer=True)
        self._subarray_group.add_spinbox("min_stations", "Min Stations:", 4, 2, 200, 1, integer=True)
        self._subarray_group.on_change(lambda _k, _v: self._update_subarray_count())

        # ── Info ──
        info_frame = tk.LabelFrame(self._left_scrollable, text="Info", font=("", 10, "bold"), padx=5, pady=5)
        info_frame.pack(fill=tk.X, padx=5, pady=3)

        self._station_info_label = tk.Label(info_frame, text="Stations: --    Pairs: --", anchor="w", font=("", 9))
        self._station_info_label.pack(fill=tk.X, padx=2)

        self._subarray_info_label = tk.Label(
            info_frame, text="Expected: -- subarrays", anchor="w", font=("", 9, "bold"),
        )
        self._subarray_info_label.pack(fill=tk.X, padx=2, pady=(3, 0))

        # ── Buttons ──
        btn_frame = tk.Frame(self._left_scrollable)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        self._load_btn = tk.Button(
            btn_frame, text="Load Data", command=self._on_load,
            bg="#4A90D9", fg="white", font=("", 10, "bold"), height=2,
        )
        self._load_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self._export_btn = tk.Button(
            btn_frame, text="Split & Export", command=self._on_export,
            bg="#E67E22", fg="white", font=("", 10, "bold"), height=2, state=tk.DISABLED,
        )
        self._export_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # ── Right Plot ──
        right_frame = tk.Frame(self._pw)
        self._pw.add(right_frame, width=600)

        self._plot = EmbeddedPlot(right_frame, figsize=(6, 5), dpi=90)
        self._plot.pack(fill=tk.BOTH, expand=True)

        # ── Status Bar ──
        self._status = StatusBar(self.root)
        self._status.pack(side=tk.BOTTOM, fill=tk.X)

    # ── Logic ────────────────────────────────────────

    def _update_subarray_count(self) -> None:
        window_size = int(self._subarray_group.get("window_size"))
        n_stations = len(self._all_stations)
        if n_stations == 0:
            self._subarray_info_label.config(text="Expected: -- subarrays")
            return
        actual_window = min(window_size, n_stations)
        n = max(0, n_stations - actual_window + 1)
        stride = max(1, int(self._subarray_group.get("stride")))
        n_sub = (n + stride - 1) // stride if stride > 0 else 0
        self._subarray_info_label.config(
            text=f"Expected: {n_sub} subarrays ({n_stations} stations, window={actual_window})",
        )

    def _on_load(self) -> None:
        """Load cross-correlation SAC data (using obspy)."""
        sac_dir = self._data_group.get("sac_dir")
        stalist = self._data_group.get("stalist")

        if not sac_dir or not os.path.isdir(sac_dir):
            messagebox.showerror("Error", "Please enter a valid SAC directory")
            return

        sac_files = sorted(glob(os.path.join(sac_dir, "*.SAC")))
        if not sac_files:
            messagebox.showerror("Error", f"No .SAC files found in:\n{sac_dir}")
            return

        if not stalist or not os.path.isfile(stalist):
            messagebox.showerror("Error", "Please select a valid station list file")
            return

        self._load_btn.config(state=tk.DISABLED, text="Loading...")
        self._export_btn.config(state=tk.DISABLED)
        self._status.set_text("Loading cross-correlation data...")
        self._status.show_progress()
        self._status.set_progress(0, len(sac_files))

        sac_dir_val = sac_dir
        stalist_val = stalist
        time_half = self._preproc_group.get("time_half")

        def _load_worker(q):
            # Load station list
            stas, lons, lats = _load_stalist(stalist_val)

            # Sort stations by integer ID (ascending), matching YN.CCFJ.move3.py behavior.
            # chishui2.txt is in descending ID order (22,21,...,3). Sorting by ID
            # ascending (3,4,...,22) orders stations geographically NW→SE along the
            # profile, so sliding-window subarrays group spatially contiguous stations.
            sort_idx = np.argsort([int(s) for s in stas])
            stas = [stas[i] for i in sort_idx]
            lons = [lons[i] for i in sort_idx]
            lats = [lats[i] for i in sort_idx]

            sta_dict = {}
            for i, s in enumerate(stas):
                sta_dict[s] = {"index": i, "lon": lons[i], "lat": lats[i]}

            ncfs_list = []
            distances_list = []
            pair_stations = []  # (sta1_idx, sta2_idx)

            for i, sac in enumerate(sac_files):
                try:
                    from obspy import read
                    tr = read(sac)[0]
                except Exception as e:
                    q.put(("text", f"Skipping {os.path.basename(sac)}: {e}"))
                    q.put(("progress", i + 1))
                    continue

                # Extract station pair from filename
                fname = os.path.basename(sac)
                match = re.match(r"(\d+)-(\d+)\.BHZ-BHZ\.SAC", fname)
                if match:
                    sta1, sta2 = match.group(1), match.group(2)
                else:
                    # Try alternate format
                    match = re.match(r"([A-Z0-9.]+)_([A-Z0-9.]+)_pws\.SAC", fname)
                    if match:
                        sta1, sta2 = match.group(1), match.group(2)
                    else:
                        q.put(("text", f"Skipping {fname}: cannot parse station pair"))
                        q.put(("progress", i + 1))
                        continue

                if sta1 not in sta_dict or sta2 not in sta_dict:
                    q.put(("text", f"Skipping {fname}: stations {sta1} or {sta2} not in list"))
                    q.put(("progress", i + 1))
                    continue

                # Extract symmetric time window and apply FFT
                b = tr.stats.sac.b
                dt = tr.stats.delta
                t1, t2 = -time_half, time_half
                n1 = max(0, int((t1 - b) / dt))
                n2 = min(tr.stats.npts, int((t2 - b) / dt))
                if n2 <= n1:
                    q.put(("text", f"Skipping {fname}: invalid time window"))
                    q.put(("progress", i + 1))
                    continue

                d = tr.data[n1:n2]
                d = np.fft.fftshift(d)
                tmp = np.fft.fft(d)
                n_fft = len(tmp)
                tmp = tmp[: n_fft // 2]

                ncfs_list.append(tmp)
                idx1, idx2 = sta_dict[sta1]["index"], sta_dict[sta2]["index"]
                coord1 = (sta_dict[sta1]["lat"], sta_dict[sta1]["lon"])
                coord2 = (sta_dict[sta2]["lat"], sta_dict[sta2]["lon"])
                try:
                    from geopy.distance import great_circle
                    dist = great_circle(coord1, coord2).km
                except ImportError:
                    dlon = (coord1[1] - coord2[1]) * 111.32
                    dlat = (coord1[0] - coord2[0]) * 111.32
                    dist = np.sqrt(dlon**2 + dlat**2)
                distances_list.append(dist)
                pair_stations.append((idx1, idx2))

                q.put(("progress", i + 1))
                if (i + 1) % 20 == 0:
                    q.put(("text", f"Processing {i + 1}/{len(sac_files)}"))

            if not ncfs_list:
                return {"error": "No valid cross-correlation data"}

            ncfs_arr = np.array(ncfs_list)
            dists_arr = np.array(distances_list)

            # Sort by distance
            sort_idx = np.argsort(dists_arr)
            ncfs_arr = ncfs_arr[sort_idx]
            dists_arr = dists_arr[sort_idx]
            pair_stations = [pair_stations[j] for j in sort_idx]

            # Frequency axis (dt=0.004 = 250 Hz assumed)
            nd = ncfs_arr.shape[1]
            actual_dt = 0.004
            freqs_arr = np.arange(nd) / (2 * nd * actual_dt)
            sr = 1.0 / actual_dt

            return {
                "ncfs": ncfs_arr,
                "distances": dists_arr,
                "freqs": freqs_arr,
                "sr": sr,
                "stations": [{"id": s, "lon": lo, "lat": la} for s, lo, la in zip(stas, lons, lats)],
                "station_indices": sta_dict,
                "pair_stations": pair_stations,
            }

        def _on_loaded(data):
            if "error" in data:
                self._status.set_text(f"Load failed: {data['error']}")
                self._status.hide_progress()
                self._load_btn.config(state=tk.NORMAL, text="Load Data")
                messagebox.showerror("Load Failed", data["error"])
                return

            self._all_ncfs = data["ncfs"]
            self._all_distances = data["distances"]
            self._all_stations = data["stations"]
            self._station_indices = data["station_indices"]
            self._sr = data["sr"]
            self._cached_pairs = data["pair_stations"]

            n_stations = len(self._all_stations)
            n_pairs = len(self._all_distances)
            self._station_info_label.config(text=f"Stations: {n_stations}    Pairs: {n_pairs}")
            self._update_subarray_count()
            self._draw_station_map()

            self._status.set_text(f"Data loaded: {n_stations} stations, {n_pairs} station pairs")
            self._status.hide_progress()
            self._load_btn.config(state=tk.NORMAL, text="Load Data")
            self._export_btn.config(state=tk.NORMAL)
            messagebox.showinfo("Loaded", f"Successfully loaded {n_stations} stations, {n_pairs} station pairs")

        def _on_error(exc):
            self._status.set_text(f"Load failed: {exc}")
            self._status.hide_progress()
            self._load_btn.config(state=tk.NORMAL, text="Load Data")
            messagebox.showerror("Load Failed", str(exc))

        run_in_thread(self.root, lambda q: _load_worker(q), on_done=_on_loaded, on_error=_on_error)

    def _draw_station_map(self) -> None:
        if not self._all_stations:
            return
        ax = self._plot.ax
        ax.clear()
        ids = [s["id"] for s in self._all_stations]
        indices = list(range(len(ids)))

        ax.scatter(indices, [0] * len(indices), c="steelblue", s=60, zorder=3,
                   edgecolors="white", linewidth=0.5)

        window_size = int(self._subarray_group.get("window_size"))
        highlight_end = min(window_size, len(indices))
        if highlight_end > 0:
            ax.axvspan(-0.5, highlight_end - 0.5, alpha=0.15, color="orange")
            ax.axvline(-0.5, color="orange", ls="--", lw=1, alpha=0.7)
            ax.axvline(highlight_end - 0.5, color="orange", ls="--", lw=1, alpha=0.7)

        step = max(1, len(indices) // 15)
        for i in range(0, len(indices), step):
            ax.annotate(ids[i], (indices[i], 0), textcoords="offset points",
                        xytext=(0, 8 if i % 2 == 0 else -15), fontsize=7,
                        ha="center", color="darkblue")

        ax.set_xlabel("Station Index", fontsize=10)
        ax.set_title(f"Station Distribution ({len(indices)} stations)", fontsize=12)
        ax.set_yticks([])
        ax.set_xlim(-1, len(indices))
        ax.grid(True, alpha=0.2, axis="x")
        self._plot.fig.tight_layout()
        self._plot.draw()

    def _on_export(self) -> None:
        if self._processing:
            return

        output_dir = self._data_group.get("output_dir")
        if not output_dir:
            messagebox.showerror("Error", "Please enter a valid output directory")
            return

        window_size = int(self._subarray_group.get("window_size"))
        stride = int(self._subarray_group.get("stride"))

        subarrays = _create_subarrays(self._all_stations, window_size, stride)
        min_stations = int(self._subarray_group.get("min_stations"))
        subarrays = [s for s in subarrays if s["size"] >= min_stations]

        if not subarrays:
            messagebox.showerror("Error", "No subarrays match the current parameters")
            return

        self._n_subarrays = len(subarrays)
        self._processing = True
        self._load_btn.config(state=tk.DISABLED)
        self._export_btn.config(state=tk.DISABLED, text="Exporting...")
        self._status.set_text(f"Exporting {len(subarrays)} subarrays...")
        self._status.show_progress()
        self._status.set_progress(0, len(subarrays))

        all_ncfs = self._all_ncfs
        all_distances = self._all_distances
        all_stations = self._all_stations
        sr_val = self._sr
        cached_pairs = getattr(self, "_cached_pairs", None)

        def _export_worker(q):
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            metadata_list = []

            for i, sa in enumerate(subarrays):
                sa_indices = set(sa["indices"])
                # Find station pairs where both stations are in the subarray
                pair_mask = np.zeros(len(all_distances), dtype=bool)
                for pi, (s1_idx, s2_idx) in enumerate(cached_pairs or []):
                    if s1_idx in sa_indices and s2_idx in sa_indices:
                        pair_mask[pi] = True

                ncfs_sub = all_ncfs[pair_mask]
                dists_sub = all_distances[pair_mask]
                r_meters = dists_sub * 1000.0

                sa_station_ids = [all_stations[idx]["id"] for idx in sa["indices"]]
                sa_station_lons = np.array([all_stations[idx]["lon"] for idx in sa["indices"]])
                sa_station_lats = np.array([all_stations[idx]["lat"] for idx in sa["indices"]])

                n_freqs = ncfs_sub.shape[1] if ncfs_sub.size > 0 else 0

                npz_path = out_path / f"subarray_{i:03d}.npz"
                np.savez_compressed(
                    npz_path,
                    ncfs=ncfs_sub,
                    r_meters=r_meters,
                    freqs=np.array([]) if n_freqs == 0 else np.arange(n_freqs) * (sr_val / 2 / n_freqs),
                    sample_rate=sr_val,
                    station_ids=sa_station_ids,
                    station_lons=sa_station_lons,
                    station_lats=sa_station_lats,
                    first_station=sa["first_id"],
                    last_station=sa["last_id"],
                    n_stations=sa["size"],
                )

                metadata_list.append({
                    "id": i,
                    "first_station": sa["first_id"],
                    "last_station": sa["last_id"],
                    "n_stations": sa["size"],
                    "n_pairs": int(np.sum(pair_mask)),
                    "min_distance_m": float(np.min(r_meters)) if len(r_meters) > 0 else 0.0,
                    "max_distance_m": float(np.max(r_meters)) if len(r_meters) > 0 else 0.0,
                    "sample_rate": float(sr_val),
                    "n_freqs": n_freqs,
                    "station_ids": sa_station_ids,
                    "station_lons": sa_station_lons.tolist(),
                    "station_lats": sa_station_lats.tolist(),
                })

                q.put(("progress", i + 1))
                q.put(("text", f"Exported {i + 1}/{len(subarrays)}: stations {sa['first_id']}-{sa['last_id']}"))

            meta_path = out_path / "subarrays_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata_list, f, indent=2, ensure_ascii=False)
            return metadata_list

        self._export_queue = queue.Queue()

        def _check_progress():
            try:
                while True:
                    msg_type, msg_data = self._export_queue.get_nowait()
                    if msg_type == "progress":
                        self._status.set_progress(msg_data, self._n_subarrays)
                    elif msg_type == "text":
                        self._status.set_text(msg_data)
                    self._export_queue.task_done()
            except queue.Empty:
                pass
            if self._export_thread.is_alive():
                self.root.after(100, _check_progress)
            else:
                self._on_export_done()

        def _export_runner():
            try:
                result = _export_worker(self._export_queue)
                self._export_queue.put(("done", result))
            except Exception as exc:
                self._export_queue.put(("error", str(exc)))

        self._export_thread = threading.Thread(target=_export_runner, daemon=True)
        self._export_thread.start()
        self.root.after(100, _check_progress)

    def _on_export_done(self) -> None:
        while True:
            try:
                msg_type, msg_data = self._export_queue.get_nowait()
                if msg_type == "done":
                    n = len(msg_data)
                    self._status.set_text(f"Export complete: {n} subarrays")
                    messagebox.showinfo("Export Complete",
                                        f"Successfully exported {n} subarrays to:\n{self._data_group.get('output_dir')}")
                elif msg_type == "error":
                    self._status.set_text(f"Export failed: {msg_data}")
                    messagebox.showerror("Export Failed", msg_data)
                self._export_queue.task_done()
            except queue.Empty:
                break
        self._status.hide_progress()
        self._processing = False
        self._load_btn.config(state=tk.NORMAL)
        self._export_btn.config(state=tk.NORMAL, text="Split & Export")


def main():
    root = tk.Tk()
    app = SubarraySplitterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
