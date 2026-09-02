"""Read the GPU's real clock, power and temperature through AMD's ADL.

**There is no rocm-smi on Windows**, and hipInfo reports only the maximum clock,
which says nothing about what the card is doing under load. Where the Adrenalin
driver is installed, `atiadlxx.dll` in System32 exposes
`ADL2_New_QueryPMLogData_Get`, the same interface tools like GPU-Z read. This
needs neither torch nor a GPU library of its own.

    python tools/gpu_sensors.py                 # read once
    python tools/gpu_sensors.py --watch 60      # read every second for a minute
    python tools/gpu_sensors.py --interval 0.5 --watch 30

It is worth having because **a compute-only load may not raise the clock at
all**: on some drivers the card sits at its idle clock through a fully busy
GPU, and nothing but a real sensor reading will show it.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from ctypes import POINTER, Structure, byref, c_int, c_void_p

ADL_PMLOG_MAX_SENSORS = 256

# From ADL_PMLOG_SENSORS in adl_defines.h. These are the ones reported as
# supported on the hardware this was written for.
SENSOR_GFXCLK = 1
SENSOR_MEMCLK = 2
SENSOR_SOCCLK = 3
SENSOR_ACTIVITY_GFX = 19
SENSOR_ASIC_POWER = 23
SENSOR_GFX_POWER = 30
SENSOR_TEMP_GFX = 28
SENSOR_TEMP_SOC = 29


class ADLSingleSensorData(Structure):
    _fields_ = [("supported", c_int), ("value", c_int)]


class ADLPMLogDataOutput(Structure):
    _fields_ = [("size", c_int), ("sensors", ADLSingleSensorData * ADL_PMLOG_MAX_SENSORS)]


class AdlSession:
    """Initialises ADL and reads from it. Use it as a context manager."""

    def __init__(self) -> None:
        self._adl = ctypes.CDLL("atiadlxx.dll")
        libc = ctypes.CDLL("msvcrt")
        libc.malloc.restype = c_void_p
        libc.malloc.argtypes = [ctypes.c_size_t]
        malloc_cb = ctypes.CFUNCTYPE(c_void_p, c_int)
        self._cb = malloc_cb(lambda size: libc.malloc(size))  # keep a reference or it crashes
        self._context = c_void_p()
        rc = self._adl.ADL2_Main_Control_Create(self._cb, 1, byref(self._context))
        if rc != 0:
            raise OSError(f"ADL2_Main_Control_Create failed rc={rc}")
        self._query = self._adl.ADL2_New_QueryPMLogData_Get
        self._query.argtypes = [c_void_p, c_int, POINTER(ADLPMLogDataOutput)]

    def read(self, adapter: int = 0) -> dict[str, int]:
        out = ADLPMLogDataOutput()
        rc = self._query(self._context, adapter, byref(out))
        if rc != 0:
            raise OSError(f"ADL2_New_QueryPMLogData_Get failed rc={rc}")
        s = out.sensors
        return {
            "gfx_mhz": s[SENSOR_GFXCLK].value,
            "mem_mhz": s[SENSOR_MEMCLK].value,
            "soc_mhz": s[SENSOR_SOCCLK].value,
            "activity_pct": s[SENSOR_ACTIVITY_GFX].value,
            "asic_w": s[SENSOR_ASIC_POWER].value,
            "gfx_w": s[SENSOR_GFX_POWER].value,
            "temp_gfx_c": s[SENSOR_TEMP_GFX].value,
            "temp_soc_c": s[SENSOR_TEMP_SOC].value,
        }

    def close(self) -> None:
        self._adl.ADL2_Main_Control_Destroy(self._context)

    def __enter__(self) -> AdlSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _print_row(values: dict[str, int]) -> None:
    print(
        f"{time.strftime('%H:%M:%S')}  gfx {values['gfx_mhz']:4d} MHz  "
        f"mem {values['mem_mhz']:4d} MHz  act {values['activity_pct']:3d}%  "
        f"asic {values['asic_w']:3d} W  gfx {values['gfx_w']:3d} W  "
        f"temp {values['temp_gfx_c']:2d}/{values['temp_soc_c']:2d} C",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="read the GPU clock, power and temperature")
    parser.add_argument("--watch", type=float, default=0.0, help="keep reading for this many seconds")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between reads")
    args = parser.parse_args()

    with AdlSession() as adl:
        if args.watch <= 0:
            _print_row(adl.read())
            return 0
        end = time.time() + args.watch
        while time.time() < end:
            _print_row(adl.read())
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
