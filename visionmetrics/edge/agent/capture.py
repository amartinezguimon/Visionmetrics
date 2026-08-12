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


def camera_name_at(index: int) -> str | None:
    """The macOS camera NAME at OpenCV `index`, or None off-macOS / unreadable.

    Used to pin a user's chosen camera to a STABLE identity: the AVFoundation
    index a device gets isn't stable (a phone can be 0 one session and 1 the
    next), but its name ("hector gregori Camera") is — so we save the name and
    re-resolve it to whatever index it holds right now."""
    names = _mac_camera_names()
    if 0 <= index < len(names):
        return names[index] or None
    return None


def resolve_camera_by_name(name: str) -> int | None:
    """The CURRENT OpenCV index whose macOS camera name matches `name`
    (case-insensitive), or None if not found / off-macOS. This is how we honor
    'use the camera the user actually picked' even after macOS reshuffles the
    indices between the picker and launch."""
    if not name:
        return None
    target = name.strip().lower()
    for i, n in enumerate(_mac_camera_names()):
        if n.strip().lower() == target:
            return i
    return None


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


def _mac_external_indices() -> list[int]:
    """macOS indices that are a REAL, non-built-in camera (phone / Camo / USB).

    This is the ONLY set of indices we ever allow ourselves to open on macOS, and
    it exists to close a nasty AVFoundation hole: asking OpenCV for an index that is
    BEYOND the number of enumerated cameras silently opens the DEFAULT device — the
    Mac's own built-in. So when the iPhone/Continuity Camera has dropped and only the
    Mac remains (index 0), probing "index 1" returns the MAC, not an external camera,
    and an index-only "is it built-in?" test (which only flags index 0) waves it
    through. That's exactly the "I picked my phone but the Mac's camera runs" bug.

    By restricting every pick/open to the enumerated, non-FaceTime indices, a phantom
    index can never be opened: if the phone isn't listed right now, this is empty and
    we serve no image (and keep retrying) rather than ever falling back to the Mac."""
    names = _mac_camera_names()
    return [i for i, n in enumerate(names) if n.strip() and not _is_builtin_name(n)]


def pick_external_camera(max_index: int = 6, warm_secs: float = 1.5,
                         slow_warm_secs: float = 6.0):
    """Pick a working camera that is NOT the machine's built-in webcam.

    Same two-pass warm-up as `pick_working_camera` (a Continuity Camera / phone
    can take seconds to wake), but the built-in indices are skipped entirely so
    we never silently fall back to the laptop's own camera. Returns the external
    index, or None if no external camera delivers a real image — the caller must
    then refuse to run rather than use the built-in."""
    # On macOS, only ever probe REAL enumerated external indices — probing a phantom
    # index (>= camera count) would silently open the Mac's built-in (see
    # _mac_external_indices). Off-macOS we have no name list, so fall back to the
    # index range minus any known built-in.
    if sys.platform == "darwin":
        order = _mac_external_indices()
    else:
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


def pick_chosen_external(preferred_name: str = "", preferred_index: int | None = None,
                         max_index: int = 6, warm_secs: float = 1.5,
                         slow_warm_secs: float = 6.0):
    """Pick the camera the user chose in the visual picker, GUARANTEED not to be the
    Mac's own built-in — resolved IN THIS PROCESS, right before capture.

    Why this exists: macOS AVFoundation indices aren't stable *between processes*,
    so an index the picker (process A) saved can point at a different camera by the
    time the capture process (B) opens it — the classic "I chose my phone but the
    Mac's FaceTime runs" bug. Doing the choice here, in the same process that
    captures, removes that cross-process gap. We:

      1. skip the built-in FaceTime indices entirely (never fall back to the Mac);
      2. warm up the remaining (external) indices — two passes, because a phone /
         Continuity Camera can take several seconds to wake;
      3. among the externals that DELIVER a real frame, prefer the one whose macOS
         name matches `preferred_name` (the camera the user clicked);
      4. else use the first working external; else `preferred_index` if it's a
         non-built-in (it may still be warming); else None.

    Returns the chosen index, or None if NO external delivers — the caller must
    then refuse to run rather than silently use the Mac's camera."""
    names = _mac_camera_names()
    exclude = _builtin_camera_indices()
    want = (preferred_name or "").strip().lower()

    print(f"[cam] eligiendo cámara (nunca la del Mac). nombres_sistema={names} "
          f"built-in(FaceTime)={sorted(exclude)} elegida='{preferred_name}' "
          f"idx_guardado={preferred_index}")

    def _name_of(i: int) -> str:
        return names[i] if 0 <= i < len(names) else ""

    # STEP 0 — HONOR THE VISUAL PICK. This is the single most reliable signal: the
    # user SAW each camera's live image in the picker and clicked one, so the saved
    # index points at the image they want. It beats macOS's system_profiler names,
    # whose order does NOT always match cv2's index order — on some Macs cv2 index 0
    # streams the iPhone while system_profiler calls index 0 "FaceTime HD Camera". The
    # old name-based resolution then "corrected" the good pick onto the Mac. So if the
    # picked index still delivers a real image (patient warm-up, because a phone that
    # slept since the picker takes seconds to wake), open EXACTLY it, whatever it's
    # named. We deliberately do NOT exclude it for having a built-in-looking name.
    #
    # Crucially we do NOT probe (open+close) here: a Continuity Camera RESETS on every
    # open, so picker-open → probe-open → capture-open = 3 resets in a few seconds and
    # the camera gets caught mid-wake → a BLACK stream. Instead we return the picked
    # index straight away and let VideoSource open it exactly ONCE and warm it patiently.
    # The ONE thing we still refuse here is a PHANTOM index: one at or beyond the
    # number of cameras macOS currently enumerates. Those don't exist, and asking
    # AVFoundation for them silently hands back the Mac's built-in — so honouring a
    # phantom is never "the camera you clicked", it is always the Mac. This happens
    # for real: pick the phone at index 1, the phone sleeps before launch, and index
    # 1 is now past the end of the list. Note we bound by COUNT, not by name — the
    # names are exactly what we don't trust here (see above), and the picker only
    # ever lists cameras that existed, so a live index is still honoured whatever
    # it's called. If the list can't be read at all we don't second-guess the pick.
    if preferred_index is not None and 0 <= preferred_index < max_index:
        if names and preferred_index >= len(names):
            print(f"[cam] el índice guardado {preferred_index} ya NO existe "
                  f"(solo hay {len(names)} cámara(s): {names}). No lo abro: sería un "
                  f"índice fantasma y macOS devolvería la cámara del Mac. "
                  f"¿Móvil dormido/desconectado? Busco tu cámara externa…")
        else:
            print(f"[cam] -> índice {preferred_index} ('{_name_of(preferred_index)}') = "
                  f"la que elegiste EN EL SELECTOR (honro tu clic; sin sondear, para no "
                  f"reiniciar la cámara — VideoSource la abre una sola vez y espera imagen).")
            return preferred_index

    # STEP 1 — resolve by NAME with ZERO camera access (just system_profiler). Fallback
    # for when the visual index no longer delivers (phone slept and macOS reshuffled).
    # NOTE: on some Macs cv2's order and system_profiler's order disagree, so this is a
    # best-effort fallback, not the primary path (STEP 0 is).
    if want:
        for i, n in enumerate(names):
            if n.strip().lower() == want:
                if i in exclude:
                    print(f"[cam] AVISO: la cámara elegida coincide con la built-in; "
                          f"la ignoro por seguridad.")
                    break
                print(f"[cam] -> índice {i} ('{n}') = la que elegiste (resuelto por "
                      f"nombre, sin sondear).")
                return i
        print(f"[cam] la cámara elegida ('{preferred_name}') NO aparece ahora mismo "
              f"en la lista del sistema. ¿Móvil desconectado/bloqueado?")

    # STEP 2 — no name match (unknown pick, or off-macOS where names==[]). Probe the
    # NON-built-in indices and take the first that delivers a real frame. Two passes,
    # because a phone / Continuity Camera can take several seconds to wake. On macOS we
    # restrict to REAL enumerated external indices: probing a phantom index (>= camera
    # count) silently opens the Mac's built-in, which is the very bug we're preventing.
    if sys.platform == "darwin":
        order = _mac_external_indices()
    else:
        order = [i for i in range(max_index) if i not in exclude]
    opened: list[int] = []
    for idx in order:
        opened_ok, delivers = _probe_camera(idx, warm_secs)
        if delivers:
            print(f"[cam] -> índice {idx} ('{_name_of(idx)}') (externa con imagen).")
            return idx
        if opened_ok:
            opened.append(idx)
    for idx in opened:
        if _probe_camera(idx, slow_warm_secs)[1]:
            print(f"[cam] -> índice {idx} ('{_name_of(idx)}') (externa, despertó lenta).")
            return idx

    # STEP 3 — nothing delivered. Hand back the saved index ONLY if it's a valid
    # external (the phone may still wake up); NEVER a built-in and NEVER a phantom
    # index. On macOS "valid external" means it's currently ENUMERATED and non-FaceTime
    # (a saved index that no longer exists would open the Mac by AVFoundation fallback).
    valid_external = (
        preferred_index in _mac_external_indices() if sys.platform == "darwin"
        else (preferred_index is not None and preferred_index not in exclude
              and 0 <= preferred_index < max_index)
    )
    if preferred_index is not None and valid_external:
        print(f"[cam] ninguna externa dio imagen aún; devuelvo el índice guardado "
              f"{preferred_index} (no es la del Mac) por si despierta.")
        return preferred_index
    print("[cam] ERROR: no encuentro tu cámara externa (y NUNCA uso la del Mac).")
    return None


def _is_realtime(source) -> bool:
    """Webcam indices and network streams are realtime; file paths are not."""
    if isinstance(source, int):
        return True
    s = str(source).lower()
    return s.startswith(("rtsp://", "http://", "https://", "udp://"))


class VideoSource:
    def __init__(self, source, *, reconnect_delay_s: float = 2.0, loop: bool = False,
                 identity_name: str | None = None, avoid_builtin: bool = False):
        # "1" (string) from a CLI/config still means webcam index 1.
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.source = source
        # macOS-only safety net for the "I chose my phone but the Mac's camera
        # runs" bug. AVFoundation indices aren't stable: when a Continuity Camera
        # (iPhone) drops for a moment, macOS reshuffles and the numeric index we
        # opened now points at the built-in FaceTime — so a plain index-based
        # reconnect silently grabs the Mac's own camera. To make that IMPOSSIBLE:
        #   * identity_name — the chosen camera's stable NAME; every (re)open
        #     re-resolves it to whatever index it currently holds, so we always
        #     reopen the SAME physical camera (or nothing, if it's gone);
        #   * avoid_builtin — never open an index that is currently the Mac's
        #     built-in, even if identity resolution fails. If the chosen camera
        #     isn't present we serve no image and keep retrying — we NEVER fall
        #     back to the Mac.
        self.identity_name = (identity_name or "").strip() or None
        self.avoid_builtin = avoid_builtin
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
    def from_config(cls, camera_cfg, *, loop: bool = False,
                    identity_name: str | None = None,
                    avoid_builtin: bool = False) -> "VideoSource":
        return cls(camera_cfg.source, reconnect_delay_s=camera_cfg.reconnect_delay_s,
                   loop=loop, identity_name=identity_name, avoid_builtin=avoid_builtin)

    def _resolve_source(self):
        """The source to ACTUALLY open right now. For a macOS webcam pinned to a
        chosen camera, re-resolve its NAME to the current index and refuse the
        built-in — returns None (= "do not open anything, retry later") if the
        chosen camera is absent or the target index is the Mac's own. Non-webcam
        (file/RTSP) and non-macOS sources are returned unchanged."""
        src = self.source
        if not isinstance(src, int) or src < 0 or sys.platform != "darwin":
            return src
        if self.identity_name is not None:
            idx = resolve_camera_by_name(self.identity_name)
            if idx is None:
                return None                       # chosen camera not present now
            src = idx
        # Phantom-index guard, applied even WITHOUT `avoid_builtin`. An index at or
        # beyond the number of enumerated cameras is not a camera at all; AVFoundation
        # answers it with the Mac's default device. So opening one is never right, not
        # even on the "honour the user's visual pick" path (which deliberately turns
        # `avoid_builtin` off because macOS NAMES are unreliable — but the COUNT is
        # not). Returning None means "serve no image and retry", so a phone that is
        # merely asleep gets picked up when it wakes instead of us silently switching
        # to the Mac. Skipped when the name list is unreadable, so a system_profiler
        # failure degrades to the old behaviour rather than blocking every camera.
        _names = _mac_camera_names()
        if _names and src >= len(_names):
            return None
        if self.avoid_builtin and src not in _mac_external_indices():
            # Airtight "never the Mac": only open an index that is a REAL, enumerated,
            # non-FaceTime camera RIGHT NOW. This rejects both the built-in itself and
            # any phantom index (>= camera count) that AVFoundation would silently
            # answer with the Mac's own camera. If the chosen camera isn't listed we
            # return None (serve no image, keep retrying) — never the built-in.
            return None
        return src

    # ── lifecycle ────────────────────────────────────────────────
    def open(self) -> bool:
        resolved = self._resolve_source()
        if resolved is None:
            # Chosen (external) camera not available and we refuse the built-in.
            self._cap = None
            return False
        if resolved != self.source:
            print(f"[cam] '{self.identity_name}' está ahora en el índice {resolved} "
                  f"(reabriendo esa cámara, nunca la del Mac).")
        self._cap = open_capture(resolved)
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
            self._cap = None
        if self._stop.wait(self.reconnect_delay_s):
            return
        # Re-resolve the chosen camera's CURRENT index by name. If it's gone (phone
        # dropped) or would resolve to the Mac's built-in, DON'T open anything —
        # leave _cap None and let the pump call us again after the delay. This is
        # what guarantees a phone drop never hands the session to the Mac camera.
        resolved = self._resolve_source()
        if resolved is None:
            print("[cam] la cámara elegida no está disponible ahora mismo; "
                  "espero a que vuelva (NUNCA abro la del Mac).")
            return
        self._cap = open_capture(resolved)
