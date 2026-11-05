"""Whether a process exists, without touching it. On POSIX, signal 0 (which sends nothing). Not on Windows: there
``os.kill(pid, 0)`` sends Ctrl+C to the process's console group (0 is ``CTRL_C_EVENT``) and any other signal ends the
process, so the process is opened and its exit code read instead."""

from __future__ import annotations

import os
import sys

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _STILL_ACTIVE = 259
    _ERROR_INVALID_PARAMETER = 87  # no process has this id

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    _kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

    def _exists(pid: int) -> bool:
        if pid > 0xFFFFFFFF:
            return False
        handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            # Only "no such process" is gone; one that cannot be looked at (another user's) exists.
            return ctypes.get_last_error() != _ERROR_INVALID_PARAMETER
        try:
            code = wintypes.DWORD()
            if not _kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == _STILL_ACTIVE
        finally:
            _kernel32.CloseHandle(handle)

else:

    def _exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True  # another user's
        except OSError:
            return False


def pid_alive(pid: object) -> bool:
    """A process with this pid exists — one of another user's too."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    return _exists(pid)
