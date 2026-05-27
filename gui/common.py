"""FJ GUI shared components module.

Provides base components used by both GUI tools:
  - StatusBar: status text + progress bar
  - LabeledSpinbox: labeled numeric input
  - LabeledEntry: labeled text input
  - FileSelector: file/directory picker
  - ParamGroup: parameter group container
  - run_in_thread: background thread helper
  - EmbeddedPlot: embedded matplotlib Figure container
"""
from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# ── StatusBar ───────────────────────────────────────────


class StatusBar(tk.Frame):
    """Bottom status bar: status text + progress bar."""

    def __init__(self, master: tk.Widget, **kwargs):
        super().__init__(master, **kwargs)
        self._label = tk.Label(self, text="Ready", anchor="w", font=("", 9))
        self._label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        self._progress = ttk.Progressbar(
            self, mode="determinate", length=200,
        )

    def set_text(self, text: str) -> None:
        """Set status text."""
        self._label.config(text=text)

    def show_progress(self) -> None:
        """Show progress bar."""
        self._progress.pack(side=tk.RIGHT, padx=(5, 5))
        self._progress.config(value=0)

    def hide_progress(self) -> None:
        """Hide progress bar."""
        self._progress.pack_forget()
        self._progress.config(value=0)

    def set_progress(self, value: float, maximum: float = 100.0) -> None:
        """Update progress (0-maximum)."""
        self._progress.config(maximum=maximum, value=min(value, maximum))
        self.update_idletasks()


# ── Input widgets ───────────────────────────────────────


class LabeledSpinbox(tk.Frame):
    """Labeled numeric input (integer or float)."""

    def __init__(
        self,
        master: tk.Widget,
        label: str,
        default: float = 0.0,
        from_: float = 0.0,
        to: float = 100.0,
        increment: float = 0.1,
        width: int = 8,
        decimal_places: int = 2,
        integer: bool = False,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self._var = tk.DoubleVar(value=default)

        tk.Label(self, text=label, font=("", 9)).pack(side=tk.LEFT, padx=(0, 3))
        self._spinbox = tk.Spinbox(
            self,
            textvariable=self._var,
            from_=from_,
            to=to,
            increment=increment,
            width=width,
            format=f"%.{decimal_places}f",
        )
        self._spinbox.pack(side=tk.LEFT)

    @property
    def value(self) -> float:
        try:
            return float(self._var.get())
        except (ValueError, tk.TclError):
            return 0.0

    @value.setter
    def value(self, v: float) -> None:
        self._var.set(v)


class LabeledEntry(tk.Frame):
    """Labeled text input."""

    def __init__(self, master: tk.Widget, label: str, default: str = "", width: int = 15, **kwargs):
        super().__init__(master, **kwargs)
        self._var = tk.StringVar(value=default)

        tk.Label(self, text=label, font=("", 9)).pack(side=tk.LEFT, padx=(0, 3))
        self._entry = tk.Entry(self, textvariable=self._var, width=width)
        self._entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    @property
    def value(self) -> str:
        return self._var.get()

    @value.setter
    def value(self, v: str) -> None:
        self._var.set(v)


class FileSelector(tk.Frame):
    """File/directory selector: text entry + 'Browse...' button.

    Supports mode='file' or mode='directory'.
    """

    def __init__(
        self,
        master: tk.Widget,
        label: str,
        default: str = "",
        mode: str = "directory",
        filetypes: Optional[List[Tuple[str, str]]] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self._mode = mode
        self._filetypes = filetypes or [("All Files", "*.*")]
        self._var = tk.StringVar(value=default)

        tk.Label(self, text=label, font=("", 9)).pack(side=tk.LEFT, padx=(0, 3))
        self._entry = tk.Entry(self, textvariable=self._var, width=28)
        self._entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        self._browse_btn = tk.Button(
            self, text="Browse...", command=self._browse, width=7,
        )
        self._browse_btn.pack(side=tk.RIGHT)

    def _browse(self) -> None:
        if self._mode == "directory":
            path = filedialog.askdirectory(title="Select Directory")
        else:
            path = filedialog.askopenfilename(title="Select File", filetypes=self._filetypes)
        if path:
            self._var.set(path)

    @property
    def value(self) -> str:
        return self._var.get()

    @value.setter
    def value(self, v: str) -> None:
        self._var.set(v)


# ── ParamGroup ──────────────────────────────────────────


class ParamGroup(tk.LabelFrame):
    """Parameter group container.

    Manages callbacks and validation for a set of LabeledSpinbox/Entry widgets.
    """

    def __init__(self, master: tk.Widget, title: str, **kwargs):
        super().__init__(master, text=title, font=("", 10, "bold"), padx=8, pady=5, **kwargs)
        self._widgets: Dict[str, tk.Widget] = {}
        self._row = 0

    def add_spinbox(
        self,
        key: str,
        label: str,
        default: float = 0.0,
        from_: float = 0.0,
        to: float = 100.0,
        increment: float = 0.1,
        integer: bool = False,
        **kwargs,
    ) -> LabeledSpinbox:
        """Add a labeled numeric control."""
        widget = LabeledSpinbox(
            self, label, default, from_, to, increment, integer=integer, **kwargs,
        )
        widget.grid(row=self._row, column=0, sticky="ew", padx=2, pady=1)
        self._widgets[key] = widget
        self._row += 1
        return widget

    def add_entry(self, key: str, label: str, default: str = "", **kwargs) -> LabeledEntry:
        """Add a labeled text control."""
        widget = LabeledEntry(self, label, default, **kwargs)
        widget.grid(row=self._row, column=0, sticky="ew", padx=2, pady=1)
        self._widgets[key] = widget
        self._row += 1
        return widget

    def add_file_selector(
        self,
        key: str,
        label: str,
        default: str = "",
        mode: str = "directory",
        **kwargs,
    ) -> FileSelector:
        """Add a file/directory selector."""
        widget = FileSelector(self, label, default, mode=mode, **kwargs)
        widget.grid(row=self._row, column=0, sticky="ew", padx=2, pady=1)
        self._widgets[key] = widget
        self._row += 1
        return widget

    def add_widget(self, key: str, widget: tk.Widget) -> tk.Widget:
        """Add a custom widget."""
        widget.grid(row=self._row, column=0, sticky="ew", padx=2, pady=1)
        self._widgets[key] = widget
        self._row += 1
        return widget

    def get(self, key: str) -> Any:
        """Get widget value."""
        w = self._widgets.get(key)
        if w is None:
            raise KeyError(f"Parameter not found: {key}")
        return w.value

    def set(self, key: str, value: Any) -> None:
        """Set widget value."""
        w = self._widgets.get(key)
        if w is None:
            raise KeyError(f"Parameter not found: {key}")
        w.value = value

    def on_change(self, callback: Callable[[str, Any], None]) -> None:
        """Register parameter change callback.

        callback signature: callback(key, value).
        """
        def _make_handler(key: str, widget: tk.Widget):
            def _handler(*args):
                val = widget.value
                callback(key, val)
            return _handler

        for key, w in self._widgets.items():
            if isinstance(w, (LabeledSpinbox, LabeledEntry)):
                w._var.trace_add("write", _make_handler(key, w))


# ── Thread helper ───────────────────────────────────────


def run_in_thread(
    root: tk.Tk,
    target: Callable,
    on_done: Optional[Callable] = None,
    on_error: Optional[Callable] = None,
    daemon: bool = True,
) -> threading.Thread:
    """Execute target in background thread, safely update UI via root.after().

    Parameters
    ----------
    root: Tk root window
    target: function to run in thread, signature target(result_queue)
    on_done: UI callback on completion, signature on_done(result)
    on_error: UI callback on exception, signature on_error(exc)
    daemon: run as daemon thread

    Returns
    -------
    thread: started background thread
    """
    result_queue: queue.Queue = queue.Queue()

    def _wrapper():
        try:
            result = target(result_queue)
            root.after(0, lambda: on_done(result) if on_done else None)
        except Exception as exc:
            root.after(0, lambda: on_error(exc) if on_error else (
                messagebox.showerror("Error", str(exc))
            ))

    thread = threading.Thread(target=_wrapper, daemon=daemon)
    thread.start()
    return thread


# ── EmbeddedPlot ────────────────────────────────────────


class EmbeddedPlot(tk.Frame):
    """Embedded matplotlib Figure container.

    Includes NavigationToolbar2Tk toolbar by default.
    """

    def __init__(
        self,
        master: tk.Widget,
        figsize: Tuple[float, float] = (8, 5),
        dpi: int = 100,
        toolbar: bool = True,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.fig = Figure(figsize=figsize, dpi=dpi)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()

        if toolbar:
            self._toolbar = NavigationToolbar2Tk(self.canvas, self)
            self._toolbar.update()
            self._toolbar.pack(side=tk.TOP, fill=tk.X)

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def clear(self) -> None:
        """Clear the plot."""
        self.ax.clear()
        self.canvas.draw_idle()

    def draw(self) -> None:
        """Refresh the plot."""
        self.canvas.draw_idle()

    def tight_layout(self) -> None:
        """Apply tight layout."""
        self.fig.tight_layout()
        self.canvas.draw_idle()
