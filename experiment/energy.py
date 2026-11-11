"""Energiemessung über interne Sensoren.

GPU: amdgpu hwmon `power1_average` (Board-Leistung in µW) der RX 9070 XT, per Polling integriert.
CPU: RAPL-Paketzähler `intel-rapl:0/energy_uj` (kumulativ, µJ). Braucht Leserecht:
     sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj   (gilt bis zum Neustart)

Nicht erfasst: RAM, Mainboard, Lüfter, Netzteilverluste. Das ist im Paper als Limitation zu nennen.
"""
from __future__ import annotations

import glob
import os
import threading
import time
from dataclasses import dataclass, asdict

GPU_PCI = os.environ.get("NKI_GPU_PCI", "0000:03:00.0")
RAPL_DIR = "/sys/class/powercap/intel-rapl:0"


def _gpu_power_file() -> str:
    hits = glob.glob(f"/sys/bus/pci/devices/{GPU_PCI}/hwmon/hwmon*/power1_average")
    if not hits:
        raise RuntimeError(f"Kein power1_average für GPU {GPU_PCI} gefunden")
    return hits[0]


def _read_int(path: str) -> int:
    with open(path) as f:
        return int(f.read().strip())


def rapl_available() -> bool:
    try:
        _read_int(f"{RAPL_DIR}/energy_uj")
        return True
    except (PermissionError, FileNotFoundError):
        return False


@dataclass
class EnergyResult:
    duration_s: float
    gpu_j: float
    cpu_j: float | None
    gpu_mean_w: float
    n_samples: int

    @property
    def total_j(self) -> float:
        return self.gpu_j + (self.cpu_j or 0.0)

    def net(self, idle: "IdlePower") -> dict:
        """Energie abzüglich der Grundlast über dieselbe Dauer."""
        gpu = self.gpu_j - idle.gpu_w * self.duration_s
        cpu = None if self.cpu_j is None or idle.cpu_w is None else self.cpu_j - idle.cpu_w * self.duration_s
        return {"gpu_j": gpu, "cpu_j": cpu, "total_j": gpu + (cpu or 0.0)}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_j"] = self.total_j
        return d


@dataclass
class IdlePower:
    gpu_w: float
    cpu_w: float | None
    duration_s: float

    def to_dict(self) -> dict:
        return asdict(self)


class PowerMonitor:
    """Kontextmanager: misst Energie zwischen __enter__ und __exit__.

    GPU-Leistung wird mit `hz` abgetastet und trapezförmig integriert.
    CPU-Energie kommt direkt aus dem RAPL-Zähler (Differenz, Überlauf berücksichtigt).
    """

    def __init__(self, hz: float = 50.0, require_rapl: bool = False):
        self.gpu_file = _gpu_power_file()
        self.period = 1.0 / hz
        self.has_rapl = rapl_available()
        if require_rapl and not self.has_rapl:
            raise PermissionError(
                "RAPL nicht lesbar. Einmal ausführen: "
                "sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj"
            )
        self.rapl_max = _read_int(f"{RAPL_DIR}/max_energy_range_uj") if self.has_rapl else None
        self.samples: list[tuple[float, float]] = []
        self.result: EnergyResult | None = None

    def _loop(self):
        nxt = time.perf_counter()
        while not self._stop.is_set():
            self.samples.append((time.perf_counter(), _read_int(self.gpu_file) / 1e6))
            nxt += self.period
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def __enter__(self):
        self.samples = []
        self._stop = threading.Event()
        self._rapl0 = _read_int(f"{RAPL_DIR}/energy_uj") if self.has_rapl else None
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        t1 = time.perf_counter()
        rapl1 = _read_int(f"{RAPL_DIR}/energy_uj") if self.has_rapl else None
        self._stop.set()
        self._thread.join()
        self.samples.append((t1, _read_int(self.gpu_file) / 1e6))
        gpu_j = sum(
            (t_b - t_a) * (p_a + p_b) / 2
            for (t_a, p_a), (t_b, p_b) in zip(self.samples, self.samples[1:])
        )
        cpu_j = None
        if self.has_rapl:
            d = rapl1 - self._rapl0
            if d < 0:
                d += self.rapl_max
            cpu_j = d / 1e6
        dur = t1 - self._t0
        self.result = EnergyResult(dur, gpu_j, cpu_j, gpu_j / dur if dur > 0 else 0.0, len(self.samples))
        return False


def measure_idle(seconds: float = 30.0, hz: float = 50.0) -> IdlePower:
    with PowerMonitor(hz=hz) as pm:
        time.sleep(seconds)
    r = pm.result
    return IdlePower(r.gpu_j / r.duration_s, None if r.cpu_j is None else r.cpu_j / r.duration_s, r.duration_s)


if __name__ == "__main__":
    # Kurzer Sensor-Check ohne Last: 5 s Idle messen.
    print("RAPL lesbar:", rapl_available())
    print("GPU-Sensor:", _gpu_power_file())
    print(measure_idle(5.0).to_dict())
