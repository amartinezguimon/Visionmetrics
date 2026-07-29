"""Video source — one abstraction over USB index, RTSP stream, and file.

The prototype only knew how to open a webcam index and assumed it never failed.
A store deployment must also pull from an IP camera (RTSP) that drops and comes
back, and we want to replay recorded clips for offline testing. All three are
the same to OpenCV; the differences this class handles are:

* realtime sources (webcam / RTSP) -> a background thread always serves the
  NEWEST frame (stale frames are dropped) and the stream auto-reconnects;
* file sources -> frames are read sequentially in order (none dropped) and the
  stream simply ends.
"""

from __future__ import annotations

import sys
import threading
import time

import cv2


def _delivers_image(cap, secs: float = 1.5) -> bool:
    """True if the capture yields a non-black frame within `secs` (some virtual
    cams open fine but output all-black on the 'wrong' backend)."""
    t = time.time()
    while time.time() - t < secs:
        ok, f = cap.read()
        if ok and f is not None and float(f.mean()) >= 8:
            return True
        time.sleep(0.03)
    return False


def open_capture(source):
    """Open a cv2 capture. On Windows, for a webcam/virtual-cam index, try
    DirectShow then MSMF and KEEP the backend that actually delivers a non-black
    frame — Camo Studio / OBS can open on one backend but output black on the
    other. A string digit ("1") is treated as a webcam index."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    if isinstance(source, int) and sys.platform.startswith("win"):
        for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            cap = cv2.VideoCapture(source, api)
            if cap.isOpened() and _delivers_image(cap):
                return cap
            cap.release()
        # Nothing produced a real frame yet (camera still warming up / not
        # streaming): open with DirectShow anyway; the caller waits for frames.
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def _probe_camera(index: int, warm_secs: float = 1.2) -> tuple[bool, bool]:
    """Probe a webcam index → (opened, delivers_image). `opened` = a backend
    accepted the index; `delivers` = it produced a non-black frame within
    warm_secs. Some indices open but stay black (an absent Continuity Camera, or
    a cam the OS hasn't granted permission to yet)."""
    cap = open_capture(index)
    try:
        if not cap.isOpened():
            return False, False
        return True, _delivers_image(cap, warm_secs)
    finally:
        cap.release()


def pick_working_camera(preferred: int = 0, max_index: int = 3,
                        warm_secs: float = 1.5, slow_warm_secs: float = 6.0):
    """Choose a webcam index that actually produces an image, so the app works on
    a machine we've never seen — a Mac's index 0 is often an absent/black
    Continuity Camera while the real webcam is 1 or 2, so hard-coding 0 shows a
    black feed on someone else's laptop. Tries `preferred` first, then 0..max_index-1.

    Two passes, because a Continuity Camera (iPhone) can take several seconds to
    physically wake — far longer than a built-in FaceTime cam. A short first pass
    keeps startup snappy when a normal webcam is present; only if nothing delivers
    quickly do we go back and give each *opened* index a long warm-up, so a slow
    iPhone camera still gets picked instead of showing a black rectangle.

    Returns the first index that delivers a real frame; if none delivers even after
    the slow pass, the first that at least opened (so the caller's own warm-up/error
    path still runs); or None if nothing opened at all."""
    order: list[int] = []
    for idx in [preferred, *range(max_index)]:
        if idx not in order:
            order.append(idx)

    # Pass 1 — quick probe. Remember which indices at least opened (a backend
    # accepted them) so we can revisit them with patience if none delivered fast.
    opened_indices: list[int] = []
    for idx in order:
        opened, delivers = _probe_camera(idx, warm_secs)
        if delivers:
            return idx
        if opened:
            opened_indices.append(idx)

    # Pass 2 — patient warm-up for a slow-waking camera (Continuity Camera).
    for idx in opened_indices:
        if _probe_camera(idx, slow_warm_secs)[1]:
            return idx

    # Nothing produced a real image; hand back the first that at least opened so
    # the caller keeps waiting on it (it may wake up even later), or None.
    return opened_indices[0] if opened_indices else None


def _mac_camera_names() -> list[str]:
    """Ordered macOS camera names, where position i corresponds to OpenCV index i.

    macOS enumerates AVFoundation devices in a fixed order that OpenCV mirrors,
    and `system_profiler` lists them in that SAME order — so the i-th name here
    is the i-th webcam index. Returns [] if it can't be read (then the caller
    falls back to index-only heuristics). We can't rely on indices alone because
    a Continuity Camera (iPhone) grabs index 0 and pushes the built-in FaceTime
    camera to 1 — the opposite of a plain laptop."""
    import json
    import subprocess
    try:
        out = subprocess.run(["system_profiler", "-json", "SPCameraDataType"],
                             capture_output=True, text=True, timeout=10)
        items = json.loads(out.stdout).get("SPCameraDataType", [])
        return [str(it.get("_name", "")) for it in items]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def _is_builtin_name(name: str) -> bool:
    """True for the Mac's OWN built-in webcam (which we must never use)."""
    return "facetime" in name.lower()


def _builtin_camera_indices() -> set[int]:
    """Camera indices that are the machine's OWN built-in webcam and must never
    be used for capture. The product only makes sense filming the shop window
    from an EXTERNAL camera (phone / Camo / USB).

    On macOS we identify the built-in by NAME ("FaceTime"), not by index, because
    a connected iPhone/Camo takes index 0 and shifts FaceTime to 1. If the names
    can't be read we conservatively exclude nothing (better to show *a* camera
    than none). On other platforms we exclude nothing — the store runs off an
    RTSP/USB source picked explicitly in config."""
    if sys.platform != "darwin":
        return set()
    return {i for i, name in enumerate(_mac_camera_names()) if _is_builtin_name(name)}


def pick_external_camera(max_index: int = 6, warm_secs: float = 1.5,
                         slow_warm_secs: float = 6.0):
    """Pick a working camera that is NOT the machine's built-in webcam.

    Same two-pass warm-up as `pick_working_camera` (a Continuity Camera / phone
    can take seconds to wake), but the built-in indices are skipped entirely so
    we never silently fall back to the laptop's own camera. Returns the external
    index, or None if no external camera delivers a real image — the caller must
    then refuse to run rather than use the built-in."""
    exclude = _builtin_camera_indices()
    order = [i for i in range(max_index) if i not in exclude]

    opened: list[int] = []
    for idx in order:
        opened_ok, delivers = _probe_camera(idx, warm_secs)
        if delivers:
            return idx
        if opened_ok:
            opened.append(idx)
    for idx in opened:
        if _probe_camera(idx, slow_warm_secs)[1]:
            return idx
    return None


def _is_realtime(source) -> bool:
    """Webcam indices and network streams are realtime; file paths are not."""
    if isinstance(source, int):
        return True
    s = str(source).lower()
    return s.startswith(("rtsp://", "http://", "https://", "udp://"))


class VideoSource:
    def __init__(self, source, *, reconnect_delay_s: float = 2.0, loop: bool = False):
        # "1" (string) from a CLI/config still means webcam index 1.
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.source = source
        self.reconnect_delay_s = reconnect_delay_s
        self.realtime = _is_realtime(source)
        # Replay a file source from the start on EOF instead of ending — only
        # meaningful for non-realtime (file) sources; used for demos/previews.
        self.loop = loop and not self.realtime
        self._cap: cv2.VideoCapture | None = None
        self._frame = None
        self._ok = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_config(cls, camera_cfg, *, loop: bool = False) -> "VideoSource":
        return cls(camera_cfg.source, reconnect_delay_s=camera_cfg.reconnect_delay_s, loop=loop)

    # ── lifecycle ────────────────────────────────────────────────
    def open(self) -> bool:
        self._cap = open_capture(self.source)
        if not self._cap.isOpened():
            return False
        if self.realtime:
            self._stop.clear()
            self._thread = threading.Thread(target=self._pump, daemon=True)
            self._thread.start()
        return True

    def read(self):
        """Return ``(ok, frame)``. For realtime sources this is the newest frame."""
        if self.realtime:
            with self._lock:
                return self._ok, (self._frame.copy() if self._frame is not None else None)
        ok, frame = self._cap.read()
        if not ok and self.loop and self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return ok, frame

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()

    # ── properties ───────────────────────────────────────────────
    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0

    @property
    def fps(self) -> float:
        return float(self._cap.get(cv2.CAP_PROP_FPS)) if self._cap else 0.0

    # ── background pump (realtime only) ──────────────────────────
    def _pump(self) -> None:
        # A realtime camera — especially an iPhone Continuity Camera — returns the
        # odd empty read (ok=False / frame=None) without actually being gone. The
        # first version tore the capture down and reopened it on the VERY FIRST
        # miss, which made such cameras "flap": image for a second, then it drops,
        # reconnects, drops again… To the user the camera "connects and then
        # disconnects" on a loop. So tolerate a short burst of misses (keep serving
        # the last good frame) and only truly reconnect after they persist.
        first_fail: float | None = None
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                self._reconnect()
                first_fail = None
                continue
            ok, frame = self._cap.read()
            if not ok or frame is None:
                now = time.time()
                if first_fail is None:
                    first_fail = now
                elif now - first_fail >= 2.0:
                    # Misses have persisted ~2s: the camera really dropped, so do a
                    # full reconnect (release + reopen after reconnect_delay_s).
                    self._reconnect()
                    first_fail = None
                    continue
                time.sleep(0.02)  # brief hiccup: wait a beat, keep the last frame
                continue
            first_fail = None
            with self._lock:
                self._ok, self._frame = ok, frame

    def _reconnect(self) -> None:
        with self._lock:
            self._ok = False
        if self._cap is not None:
            self._cap.release()
        if self._stop.wait(self.reconnect_delay_s):
            return
        self._cap = open_capture(self.source)
