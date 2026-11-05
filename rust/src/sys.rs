//! The little the library asks of the operating system: this process, this machine's boot, whether a process lives,
//! what it runs, who owns a file.

use std::path::Path;

/// This process's id.
pub(crate) fn pid() -> u32 {
    std::process::id()
}

/// This machine's name.
pub(crate) fn hostname() -> String {
    #[cfg(unix)]
    {
        let mut buf = [0u8; 256];
        // SAFETY: the buffer is valid for its length, and gethostname writes at most that many bytes.
        let rc = unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) };
        if rc == 0 {
            let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
            return String::from_utf8_lossy(&buf[..end]).into_owned();
        }
        String::new()
    }
    #[cfg(not(unix))]
    {
        std::env::var("COMPUTERNAME").unwrap_or_default()
    }
}

/// When this machine started, in ms: a pid written down before a reboot names some other process now.
pub(crate) fn boot_time() -> f64 {
    let now = crate::js::now_ms();
    #[cfg(target_os = "linux")]
    {
        if let Some(up) = std::fs::read_to_string("/proc/uptime")
            .ok()
            .and_then(|t| t.split_whitespace().next().and_then(|x| x.parse::<f64>().ok()))
        {
            return now - up * 1000.0;
        }
    }
    #[cfg(any(target_os = "macos", target_os = "freebsd", target_os = "openbsd", target_os = "netbsd"))]
    {
        let mut tv = libc::timeval { tv_sec: 0, tv_usec: 0 };
        let mut size = std::mem::size_of::<libc::timeval>();
        let mut mib = [libc::CTL_KERN, libc::KERN_BOOTTIME];
        // SAFETY: mib names kern.boottime, whose value is a timeval; tv and size describe a buffer that holds one.
        let rc = unsafe {
            libc::sysctl(
                mib.as_mut_ptr(),
                2,
                (&mut tv as *mut libc::timeval).cast(),
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 {
            // Whole seconds, as the reference's uptime reads it.
            return tv.tv_sec as f64 * 1000.0;
        }
    }
    now
}

/// The same boot as `boot` (a `boot_time()` of then): a minute apart at most, uptime being in whole seconds on some
/// systems.
pub(crate) fn same_boot(boot: Option<f64>) -> bool {
    boot.is_some_and(|b| (boot_time() - b).abs() < 60_000.0)
}

/// A process with this pid exists — one of another user's too.
pub(crate) fn alive(pid: i64) -> bool {
    if pid <= 0 || pid > i64::from(i32::MAX) {
        return false;
    }
    #[cfg(unix)]
    {
        // SAFETY: signal 0 checks for existence and sends nothing.
        let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
        rc == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(windows)]
    {
        // One that cannot be looked at (another user's) exists; only "no such process" is gone.
        windows::running(pid as u32) != Some(false)
    }
    #[cfg(not(any(unix, windows)))]
    {
        // No cheap check: taken as alive, so nothing is reaped or taken over on a guess.
        true
    }
}

/// Windows: whether a process is running, from its exit code; `None` when it cannot be looked at.
#[cfg(windows)]
mod windows {
    use std::ffi::c_void;

    const PROCESS_TERMINATE: u32 = 0x0001;
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    const STILL_ACTIVE: u32 = 259;
    const ERROR_INVALID_PARAMETER: u32 = 87;

    #[link(name = "kernel32")]
    extern "system" {
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> *mut c_void;
        fn GetExitCodeProcess(process: *mut c_void, code: *mut u32) -> i32;
        fn TerminateProcess(process: *mut c_void, code: u32) -> i32;
        fn CloseHandle(handle: *mut c_void) -> i32;
        fn GetLastError() -> u32;
    }

    pub(super) fn running(pid: u32) -> Option<bool> {
        // SAFETY: plain Win32 calls on a handle opened here and closed before returning.
        unsafe {
            let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if h.is_null() {
                // No process has this id; anything else (access denied) is a process that exists.
                return if GetLastError() == ERROR_INVALID_PARAMETER { Some(false) } else { None };
            }
            let mut code = 0u32;
            let ok = GetExitCodeProcess(h, &mut code) != 0;
            CloseHandle(h);
            ok.then_some(code == STILL_ACTIVE)
        }
    }

    /// Ends a process (nothing when it is gone or not ours to end).
    pub(super) fn terminate(pid: u32) {
        // SAFETY: plain Win32 calls on a handle opened here and closed before returning.
        unsafe {
            let h = OpenProcess(PROCESS_TERMINATE, 0, pid);
            if !h.is_null() {
                TerminateProcess(h, 1);
                CloseHandle(h);
            }
        }
    }
}

/// The process is certainly gone (no such process), as opposed to unknown.
pub(crate) fn gone(pid: i64) -> bool {
    #[cfg(unix)]
    {
        if pid <= 0 || pid > i64::from(i32::MAX) {
            return true;
        }
        // SAFETY: signal 0 checks for existence and sends nothing.
        let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
        rc != 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
    }
    #[cfg(windows)]
    {
        pid <= 0 || pid > i64::from(u32::MAX) || windows::running(pid as u32) == Some(false)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        false
    }
}

/// Send a signal (`TERM` or `KILL`) to a process; on Windows, which has no signals, it is ended either way.
pub(crate) fn signal(pid: i64, kill: bool) {
    #[cfg(unix)]
    {
        if pid > 0 && pid <= i64::from(i32::MAX) {
            // SAFETY: a plain signal to a pid; a gone process gives ESRCH, which is ignored.
            unsafe {
                libc::kill(pid as libc::pid_t, if kill { libc::SIGKILL } else { libc::SIGTERM });
            }
        }
    }
    #[cfg(windows)]
    {
        let _ = kill;
        if pid > 0 && pid <= i64::from(u32::MAX) {
            windows::terminate(pid as u32);
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (pid, kill);
    }
}

/// Run `f` when the process exits normally (the C runtime's `atexit`, on every platform).
pub(crate) fn at_exit(f: extern "C" fn()) {
    extern "C" {
        fn atexit(f: extern "C" fn()) -> i32;
    }
    // SAFETY: registers a plain extern "C" function with the C runtime; it is called once, at exit.
    unsafe {
        atexit(f);
    }
}

/// A running process's command line, or `None` when it cannot be read (no /proc and no ps, Windows).
pub(crate) async fn command_of(pid: i64) -> Option<String> {
    if let Ok(b) = tokio::fs::read(format!("/proc/{pid}/cmdline")).await {
        return Some(String::from_utf8_lossy(&b).split('\0').collect::<Vec<_>>().join(" "));
    }
    if cfg!(windows) {
        return None;
    }
    let ps = tokio::process::Command::new("ps")
        .args(["-ww", "-o", "command=", "-p", &pid.to_string()])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .output();
    let out = tokio::time::timeout(std::time::Duration::from_secs(5), ps).await.ok()?.ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Only what this user alone could have put there: owned by this user, no symlink, writable by no one else. Windows
/// has no owner bits; there the user's profile ACLs keep others out.
pub(crate) fn owned_safely(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let Ok(m) = std::fs::symlink_metadata(path) else { return false };
        // SAFETY: getuid has no preconditions.
        let uid = unsafe { libc::getuid() };
        !m.file_type().is_symlink() && m.uid() == uid && m.mode() & 0o022 == 0
    }
    #[cfg(not(unix))]
    {
        std::fs::symlink_metadata(path).is_ok_and(|m| !m.file_type().is_symlink())
    }
}

/// A folder made with these permissions (0700 for what is private), and set so when it was there already (the
/// builder's mode applies only to a folder it makes).
pub(crate) fn make_dir(path: &Path, mode: u32) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new().recursive(true).mode(mode).create(path)?;
        chmod(path, mode)
    }
    #[cfg(not(unix))]
    {
        let _ = mode;
        std::fs::create_dir_all(path)
    }
}

/// Set a file's or folder's permission bits (nothing where there are none).
pub(crate) fn chmod(path: &Path, mode: u32) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
    }
    #[cfg(not(unix))]
    {
        let _ = (path, mode);
        Ok(())
    }
}
