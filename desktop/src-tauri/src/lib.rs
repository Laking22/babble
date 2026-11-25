//! The Babble desktop app: a native window around the web interface (../src), and the voice engine
//! (the Python app, run.py) started with it and stopped when the app quits.
//!
//! - The engine listens on 127.0.0.1:7895. If it's already running (the launcher, or the app opened twice), the
//!   window just uses it and leaves it running on quit: only an engine this app started is stopped.
//! - Its output goes to ~/Library/Logs/Babble.log on macOS, or <data_dir>/logs/Babble.log on Windows/Linux.
//! - Files the interface offers to save are copied to Downloads and revealed in Finder/Explorer, and links to
//!   other pages open in the browser, not in the window.
//! - On macOS, API keys live in the macOS Keychain (passed over stdin + fd 3 pipe).
//!   On Windows, API keys live in Windows Credential Manager (passed over stdin + authenticated loopback socket).

use std::fs::OpenOptions;
use std::io::Write;
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[cfg(target_os = "macos")]
use std::io::{BufRead, BufReader};
#[cfg(target_os = "macos")]
use std::os::unix::io::FromRawFd;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
#[cfg(target_os = "macos")]
use std::process::{ChildStdin, Stdio};

#[cfg(target_os = "windows")]
use std::io::{BufRead, BufReader};
#[cfg(target_os = "windows")]
use std::net::TcpListener;
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;
#[cfg(target_os = "windows")]
use std::process::{ChildStdin, Stdio};

use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};

const PORT: u16 = 7895;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// The engine this app started (None when it was already running).
struct Engine(Mutex<Option<Child>>);

/// Last startup/crash error from start_engine or watch_engine (shown immediately on Splash).
struct EngineError(Mutex<Option<String>>);

/// Current startup / first-run bootstrap phase message shown on Splash.
struct EnginePhase(Mutex<Option<String>>);

/// PID of any active first-run bootstrap subprocess (curl/uv/tar/powershell), killed cleanly on app exit.
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
struct BootstrapPid(Mutex<Option<u32>>);

/// Set as the app quits, so the watchdog doesn't start the engine again while it's being stopped.
static QUITTING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Quitting to install an update (Restart to update): it isn't installed a second time at quit.
static UPDATING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Guards against spawning duplicate engine watchdog threads when the user clicks Retry setup.
static WATCHER_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn home() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(p) = std::env::var("USERPROFILE") {
            return PathBuf::from(p);
        }
        return PathBuf::from("C:\\");
    }
    #[cfg(not(target_os = "windows"))]
    {
        PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))
    }
}

/// Data folder shared with narrator/paths.py.
fn data_dir() -> PathBuf {
    if let Ok(p) = std::env::var("NARRATOR_HOME") {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }
    #[cfg(target_os = "macos")]
    {
        // the engine moves the folder from its old name when it starts (narrator/move_home.py: same rule)
        let base = home().join("Library/Application Support");
        let (old, new) = (base.join("Narrator Studio"), base.join("Babble"));
        let old_real = std::fs::symlink_metadata(&old).map(|m| m.is_dir()).unwrap_or(false);
        let new_used = new.join("engine-token").exists() || new.join("settings.json").exists();
        if old_real && !new_used { old } else { new }
    }
    #[cfg(target_os = "windows")]
    {
        // (Roaming, not Local: %LOCALAPPDATA%\Babble is where the installer puts the app itself, and an
        // uninstall or update must never touch the voices, books and keys)
        let base = std::env::var("APPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home().join("AppData").join("Roaming"));
        base.join("Babble")
    }
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    {
        let base = std::env::var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home().join(".local/share"));
        base.join("babble")
    }
}

fn log_path() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        home().join("Library/Logs/Babble.log")
    }
    #[cfg(not(target_os = "macos"))]
    {
        let dir = data_dir().join("logs");
        let _ = std::fs::create_dir_all(&dir);
        dir.join("Babble.log")
    }
}

/// A line in the engine's log (Babble.log), from the app itself.
fn note(line: &str) {
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(log_path()) {
        let _ = writeln!(f, "[app] {line}");
    }
}

fn read_log_tail(max_lines: usize) -> String {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open(log_path()) else { return String::new() };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    if len > 8192 {
        let _ = f.seek(SeekFrom::End(-8192));
    }
    let mut buf = Vec::new();
    if f.read_to_end(&mut buf).is_err() {
        return String::new();
    }
    let text = String::from_utf8_lossy(&buf);
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.len().saturating_sub(max_lines);
    lines[start..].join("\n")
}

fn set_engine_error(handle: &tauri::AppHandle, msg: Option<String>) {
    *handle.state::<EngineError>().0.lock().unwrap() = msg;
}

#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
fn set_engine_phase(handle: &tauri::AppHandle, msg: Option<&str>) {
    *handle.state::<EnginePhase>().0.lock().unwrap() = msg.map(|s| s.to_string());
}

/// Launch the engine off the main thread, then watch it and restart on unexpected crashes (up to 3 times in 10 min).
fn watch_engine(handle: tauri::AppHandle) {
    if WATCHER_ACTIVE
        .compare_exchange(false, true, std::sync::atomic::Ordering::SeqCst, std::sync::atomic::Ordering::SeqCst)
        .is_err()
    {
        return;
    }
    std::thread::spawn(move || {
        struct WatcherGuard;
        impl Drop for WatcherGuard {
            fn drop(&mut self) {
                WATCHER_ACTIVE.store(false, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let _guard = WatcherGuard;
        if !engine_running() {
            let root = project_root();
            #[cfg(target_os = "windows")]
            let ready = ensure_windows_runtime(&handle, &root);
            #[cfg(not(target_os = "windows"))]
            let ready = ensure_unix_runtime(&handle, &root);
            if let Err(msg) = ready {
                note(&msg);
                set_engine_error(&handle, Some(msg));
                set_engine_phase(&handle, None);
                return;
            }
            if QUITTING.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
        }
        set_engine_phase(&handle, Some("Starting the voice engine…"));
        let mut started_at = Instant::now();
        match start_engine() {
            Ok(child) => {
                *handle.state::<Engine>().0.lock().unwrap() = child;
                set_engine_error(&handle, None);
            }
            Err(msg) => {
                note(&msg);
                set_engine_error(&handle, Some(msg));
            }
        }
        let mut restarts: Vec<Instant> = Vec::new();
        loop {
            std::thread::sleep(Duration::from_secs(2));
            if QUITTING.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            let state = handle.state::<Engine>();
            let mut slot = state.0.lock().unwrap();
            let ended = match slot.as_mut() {
                Some(child) => child.try_wait().ok().flatten(),
                None => continue,
            };
            let Some(status) = ended else { continue };
            *slot = None;
            let quick_crash = started_at.elapsed() < Duration::from_secs(10);
            restarts.retain(|t| t.elapsed() < Duration::from_secs(600));
            if (quick_crash && !restarts.is_empty()) || restarts.len() >= 3 {
                let msg = format!("The voice engine exited immediately ({status}). Check Babble.log for details.");
                note(&msg);
                set_engine_error(&handle, Some(msg));
                continue;
            }
            restarts.push(Instant::now());
            note(&format!("the engine stopped by itself ({status}); starting it again"));
            drop(slot);
            std::thread::sleep(Duration::from_secs(2));
            if QUITTING.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            started_at = Instant::now();
            match start_engine() {
                Ok(child) => {
                    *handle.state::<Engine>().0.lock().unwrap() = child;
                }
                Err(msg) => {
                    note(&msg);
                    set_engine_error(&handle, Some(msg));
                }
            }
        }
    });
}

fn engine_running() -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], PORT));
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

/// The project folder (run.py and its .venv). NARRATOR_ROOT overrides it; otherwise search installed & dev locations.
fn project_root() -> PathBuf {
    if let Ok(p) = std::env::var("NARRATOR_ROOT") {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.join("engine"));
                candidates.push(dir.join("resources").join("engine"));
                candidates.push(dir.to_path_buf());
                candidates.push(dir.join("resources"));
            }
        }
        #[cfg(target_os = "windows")]
        candidates.push(data_dir().join("repo"));
        // Linux packages keep resources in lib/<product name> next to bin/ (/usr/lib/Babble, or inside the AppImage)
        #[cfg(all(unix, not(target_os = "macos")))]
        if let Some(prefix) = std::env::current_exe().ok().and_then(|e| e.parent().and_then(|d| d.parent()).map(|d| d.to_path_buf())) {
            for name in ["Babble", "babble"] {
                candidates.push(prefix.join("lib").join(name).join("engine"));
            }
        }
        candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."));
        for dir in &candidates {
            if dir.join("run.py").is_file() {
                return dir.clone();
            }
        }
    }
    #[cfg(target_os = "macos")]
    if let Some(res) = std::env::current_exe().ok().and_then(|e| e.parent().and_then(|d| d.parent()).map(|c| c.join("Resources").join("engine"))) {
        if res.join("run.py").is_file() {
            return res; // the released app: its engine is inside it, its Python in the data folder (ensure_unix_runtime)
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn python_exe(root: &Path) -> PathBuf {
    if let Ok(p) = std::env::var("BABBLE_PYTHON") {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }
    #[cfg(target_os = "windows")]
    {
        let local_venv = root.join(".venv").join("Scripts").join("python.exe");
        if local_venv.exists() {
            return local_venv;
        }
        let data_venv = data_dir().join(".venv").join("Scripts").join("python.exe");
        if data_venv.exists() {
            return data_venv;
        }
        PathBuf::from("python.exe")
    }
    #[cfg(target_os = "macos")]
    {
        let local_venv = root.join(".venv/bin/python");
        if local_venv.exists() {
            return local_venv; // (a developer's own checkout)
        }
        data_dir().join(".venv/bin/python") // (set up at first start: ensure_unix_runtime)
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let local_venv = root.join(".venv/bin/python");
        if local_venv.exists() {
            return local_venv;
        }
        data_dir().join(".venv/bin/python") // (set up at first start: ensure_unix_runtime)
    }
}

// ---------- the API keys, in the macOS Keychain ----------

#[cfg(target_os = "macos")]
const VAULT_SERVICE: &str = "Babble";
#[cfg(target_os = "macos")]
const VAULT_ACCOUNT: &str = "api-keys";
#[cfg(target_os = "macos")]
const ITEM_NOT_FOUND: i32 = -25300; // errSecItemNotFound: nothing saved yet

#[cfg(target_os = "macos")]
fn vault_read() -> Result<Option<Vec<u8>>, String> {
    match security_framework::passwords::get_generic_password(VAULT_SERVICE, VAULT_ACCOUNT) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(e) if e.code() == ITEM_NOT_FOUND => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

#[cfg(target_os = "macos")]
fn vault_serve(mut to_engine: ChildStdin, from_engine: i32) {
    let (keys, ok, why) = match vault_read() {
        Ok(Some(bytes)) => (serde_json::from_slice(&bytes).unwrap_or_else(|_| serde_json::json!({})), true, String::new()),
        Ok(None) => (serde_json::json!({}), true, String::new()),
        Err(e) => (serde_json::json!({}), false, e),
    };
    let first = serde_json::json!({"vault": keys, "ok": ok, "why": why});
    if writeln!(to_engine, "{first}").and_then(|_| to_engine.flush()).is_err() {
        return;
    }
    // SAFETY: from_engine is the read end of a pipe made in start_engine, owned by this thread from here on
    let back = unsafe { std::fs::File::from_raw_fd(from_engine) };
    for line in BufReader::new(back).lines() {
        let Ok(line) = line else { break };
        let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
        let seq = msg.get("seq").and_then(|v| v.as_u64()).unwrap_or(0);
        let reply = if !ok {
            serde_json::json!({"failed": seq, "why": "Babble couldn't read its keys from the Keychain when it opened."})
        } else {
            let body = serde_json::to_vec(msg.get("vault").unwrap_or(&serde_json::json!({}))).unwrap_or_default();
            match security_framework::passwords::set_generic_password(VAULT_SERVICE, VAULT_ACCOUNT, &body) {
                Ok(()) => serde_json::json!({"saved": seq}),
                Err(e) => serde_json::json!({"failed": seq, "why": e.to_string()}),
            }
        };
        if writeln!(to_engine, "{reply}").and_then(|_| to_engine.flush()).is_err() {
            break;
        }
    }
}

// ---------- the API keys, in Windows Credential Manager ----------

#[cfg(target_os = "windows")]
const CRED_MAX_BLOB: usize = 2560; // CRED_MAX_CREDENTIAL_BLOB_SIZE (5 * 512)

#[cfg(target_os = "windows")]
fn to_wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

#[cfg(target_os = "windows")]
fn zeroize(buf: &mut [u8]) {
    for b in buf.iter_mut() {
        unsafe { std::ptr::write_volatile(b, 0) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

#[cfg(target_os = "windows")]
fn vault_read_win() -> Result<Option<Vec<u8>>, String> {
    use windows_sys::Win32::Foundation::{GetLastError, ERROR_NOT_FOUND};
    use windows_sys::Win32::Security::Credentials::{CredFree, CredReadW, CREDENTIALW, CRED_TYPE_GENERIC};
    struct CredGuard(*mut CREDENTIALW);
    impl Drop for CredGuard {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { CredFree(self.0 as *const std::ffi::c_void) };
            }
        }
    }
    let target = to_wide("Babble:api-keys");
    let mut ptr: *mut CREDENTIALW = std::ptr::null_mut();
    let ok = unsafe { CredReadW(target.as_ptr(), CRED_TYPE_GENERIC, 0, &mut ptr) };
    if ok == 0 {
        let err = unsafe { GetLastError() };
        if err == ERROR_NOT_FOUND {
            return Ok(None);
        }
        return Err(format!("Windows Credential Manager error {err}"));
    }
    let _guard = CredGuard(ptr);
    let bytes = unsafe {
        let cred = &*ptr;
        if cred.CredentialBlob.is_null() || cred.CredentialBlobSize == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(cred.CredentialBlob, cred.CredentialBlobSize as usize).to_vec()
        }
    };
    Ok(Some(bytes))
}

#[cfg(target_os = "windows")]
fn vault_write_win(body: &[u8]) -> Result<(), String> {
    use windows_sys::Win32::Foundation::GetLastError;
    use windows_sys::Win32::Security::Credentials::{
        CredWriteW, CREDENTIALW, CRED_PERSIST_LOCAL_MACHINE, CRED_TYPE_GENERIC,
    };
    if body.len() > CRED_MAX_BLOB {
        return Err(format!("Saved keys exceed Windows Credential Manager's {CRED_MAX_BLOB}-byte limit."));
    }
    let mut target = to_wide("Babble:api-keys");
    let mut user = to_wide("api-keys");
    let mut blob = body.to_vec();
    let mut cred: CREDENTIALW = unsafe { std::mem::zeroed() };
    cred.Type = CRED_TYPE_GENERIC;
    cred.TargetName = target.as_mut_ptr();
    cred.UserName = user.as_mut_ptr();
    cred.Persist = CRED_PERSIST_LOCAL_MACHINE;
    cred.CredentialBlobSize = blob.len() as u32;
    cred.CredentialBlob = if blob.is_empty() { std::ptr::null_mut() } else { blob.as_mut_ptr() };
    let ok = unsafe { CredWriteW(&cred, 0) };
    zeroize(&mut blob);
    if ok == 0 {
        let err = unsafe { GetLastError() };
        return Err(format!("Windows Credential Manager write error {err}"));
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn random_hex_nonce() -> Result<String, String> {
    let mut buf = [0u8; 16];
    getrandom::fill(&mut buf).map_err(|e| format!("CSPRNG failed: {e}"))?;
    let out: String = buf.iter().map(|b| format!("{b:02x}")).collect();
    zeroize(&mut buf);
    Ok(out)
}

#[cfg(target_os = "windows")]
fn ct_eq(a: &str, b: &str) -> bool {
    let (ab, bb) = (a.as_bytes(), b.as_bytes());
    if ab.len() != bb.len() {
        return false;
    }
    ab.iter().zip(bb.iter()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(target_os = "windows")]
fn vault_serve_win(mut to_engine: ChildStdin, listener: TcpListener, nonce: String) {
    let port = match listener.local_addr() {
        Ok(addr) => addr.port(),
        Err(_) => return,
    };
    let (keys, ok, why) = match vault_read_win() {
        Ok(Some(mut bytes)) => {
            let parsed = serde_json::from_slice(&bytes).unwrap_or_else(|_| serde_json::json!({}));
            zeroize(&mut bytes);
            (parsed, true, String::new())
        }
        Ok(None) => (serde_json::json!({}), true, String::new()),
        Err(e) => (serde_json::json!({}), false, e),
    };
    let first = serde_json::json!({
        "vault": keys, "ok": ok, "why": why,
        "back_port": port, "back_nonce": nonce
    });
    if writeln!(to_engine, "{first}").and_then(|_| to_engine.flush()).is_err() {
        return;
    }
    // Loop accept under a 10s deadline with 2s per-connection read timeout so a stray local probe can't lock out Python.
    let _ = listener.set_nonblocking(true);
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut authed_reader = None;
    while Instant::now() < deadline {
        match listener.accept() {
            Ok((stream, _)) => {
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                let mut reader = BufReader::new(stream);
                let mut auth_line = String::new();
                if reader.read_line(&mut auth_line).is_ok() {
                    let auth_ok = serde_json::from_str::<serde_json::Value>(&auth_line)
                        .ok()
                        .and_then(|v| v.get("nonce").and_then(|n| n.as_str()).map(|s| ct_eq(s, &nonce)))
                        .unwrap_or(false);
                    if auth_ok {
                        let _ = reader.get_ref().set_read_timeout(None);
                        authed_reader = Some(reader);
                        break;
                    }
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(_) => break,
        }
    }
    drop(listener);
    let Some(reader) = authed_reader else { return };
    for line in reader.lines() {
        let Ok(line) = line else { break };
        let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
        let seq = msg.get("seq").and_then(|v| v.as_u64()).unwrap_or(0);
        let reply = if !ok {
            serde_json::json!({"failed": seq, "why": "Babble couldn't read its keys from Credential Manager when it opened."})
        } else {
            let mut body = serde_json::to_vec(msg.get("vault").unwrap_or(&serde_json::json!({}))).unwrap_or_default();
            let res = vault_write_win(&body);
            zeroize(&mut body);
            match res {
                Ok(()) => serde_json::json!({"saved": seq}),
                Err(e) => serde_json::json!({"failed": seq, "why": e}),
            }
        };
        if writeln!(to_engine, "{reply}").and_then(|_| to_engine.flush()).is_err() {
            break;
        }
    }
}

// ---------- starting and stopping the engine ----------

#[cfg(target_os = "macos")]
fn start_engine() -> Result<Option<Child>, String> {
    if engine_running() {
        return Ok(None);
    }
    let root = project_root();
    let log = OpenOptions::new().create(true).append(true).open(log_path()).map_err(|e| format!("Couldn't open Babble.log: {e}"))?;
    let err = log.try_clone().map_err(|e| format!("Couldn't clone log handle: {e}"))?;
    // the pipe the engine sends key changes back on: its write end becomes the engine's fd 3
    let mut fds = [0i32; 2];
    // SAFETY: plain pipe(2) and fcntl(2) on the two new descriptors
    let piped = unsafe { libc::pipe(fds.as_mut_ptr()) } == 0;
    let (back_r, back_w) = (fds[0], fds[1]);
    if piped {
        unsafe {
            libc::fcntl(back_r, libc::F_SETFD, libc::FD_CLOEXEC);
            libc::fcntl(back_w, libc::F_SETFD, libc::FD_CLOEXEC);
        }
    }
    let py = python_exe(&root);
    let mut cmd = Command::new(&py);
    cmd.args(["run.py", "--port", &PORT.to_string()])
        .current_dir(&root)
        .env_remove("NARRATOR_HOME") // always the real data folder
        .env("BABBLE_APP_VERSION", env!("CARGO_PKG_VERSION")) // (updates: narrator/updater.py)
        .env("BABBLE_MOVE_DATA", "1") // this app knows the folder's new name: the engine may move it (move_home.py)
        .env("PATH", format!("{}:{}:/opt/homebrew/bin:/usr/local/bin", data_dir().join("tools").join("ffmpeg").display(),
                             std::env::var("PATH").unwrap_or_else(|_| "/usr/bin:/bin".into())))
        .stdout(log)
        .stderr(err);
    if !root.join(".venv").exists() {
        cmd.env("PYTHONDONTWRITEBYTECODE", "1"); // the released app: its engine files are signed and may be read-only
    }
    if piped {
        cmd.env("BABBLE_VAULT", "1").env("BABBLE_VAULT_FD", "3").stdin(Stdio::piped());
        // SAFETY: only dup2/fcntl (async-signal-safe) between fork and exec
        unsafe {
            cmd.pre_exec(move || {
                let _ = libc::setpgid(0, 0);
                if back_w == 3 {
                    if libc::fcntl(3, libc::F_SETFD, 0) < 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                } else if libc::dup2(back_w, 3) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
    let res = cmd.spawn();
    if piped {
        unsafe { libc::close(back_w) };
    }
    match res {
        Ok(mut child) => {
            if piped {
                match child.stdin.take() {
                    Some(stdin) => {
                        std::thread::spawn(move || vault_serve(stdin, back_r));
                    }
                    None => unsafe {
                        libc::close(back_r);
                    },
                }
            }
            Ok(Some(child))
        }
        Err(e) => {
            if piped {
                unsafe { libc::close(back_r) };
            }
            Err(format!("Couldn't start Python ({}): {e}", py.display()))
        }
    }
}

#[cfg(target_os = "windows")]
fn system32_exe(rel: &str, fallback: &str) -> PathBuf {
    let root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".to_string());
    let p = PathBuf::from(root).join("System32").join(rel);
    if p.is_file() { p } else { PathBuf::from(fallback) }
}

#[cfg(target_os = "windows")]
fn rename_with_retry(src: &Path, dst: &Path) -> Result<(), String> {
    let mut last_err = String::new();
    for attempt in 0..5 {
        if dst.exists() {
            let _ = std::fs::remove_file(dst);
        }
        match std::fs::rename(src, dst) {
            Ok(()) => return Ok(()),
            Err(e) => {
                last_err = e.to_string();
                if attempt < 4 {
                    std::thread::sleep(Duration::from_millis(200));
                }
            }
        }
    }
    Err(format!("Couldn't move {} into place: {last_err}", dst.display()))
}

#[cfg(target_os = "windows")]
fn run_logged(handle: &tauri::AppHandle, cmd: &mut Command, what: &str) -> Result<(), String> {
    if QUITTING.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("Setup cancelled because Babble is closing.".into());
    }
    let log = OpenOptions::new().create(true).append(true).open(log_path()).map_err(|e| format!("Couldn't open Babble.log: {e}"))?;
    let err = log.try_clone().map_err(|e| format!("Couldn't clone log handle: {e}"))?;
    let mut child = cmd
        .creation_flags(CREATE_NO_WINDOW)
        .stdout(log)
        .stderr(err)
        .spawn()
        .map_err(|e| format!("{what} failed to launch: {e}"))?;
    *handle.state::<BootstrapPid>().0.lock().unwrap() = Some(child.id());
    let status = child.wait().map_err(|e| format!("{what} wait failed: {e}"));
    *handle.state::<BootstrapPid>().0.lock().unwrap() = None;
    let status = status?;
    if !status.success() {
        return Err(format!("{what} failed ({status}). Check Babble.log for details."));
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn download_file_win(handle: &tauri::AppHandle, url: &str, dest: &Path) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let part = PathBuf::from(format!("{}.part", dest.display()));
    let _ = std::fs::remove_file(&part);
    let mut curl = Command::new(system32_exe("curl.exe", "curl.exe"));
    curl.args([
        "-fsSL",
        "--connect-timeout", "20",
        "--speed-limit", "10240",
        "--speed-time", "30",
        "--max-time", "600",
        "--retry", "3",
        "--retry-delay", "2",
        "-o",
    ])
    .arg(&part)
    .arg(url);
    if run_logged(handle, &mut curl, "Downloading with curl.exe").is_ok() && part.is_file() {
        return rename_with_retry(&part, dest);
    }
    if QUITTING.load(std::sync::atomic::Ordering::SeqCst) {
        let _ = std::fs::remove_file(&part);
        return Err("Setup cancelled because Babble is closing.".into());
    }
    let _ = std::fs::remove_file(&part);
    // Fallback to PowerShell Invoke-WebRequest with env vars (safe for apostrophes/spaces in paths)
    let mut ps = Command::new(system32_exe(r"WindowsPowerShell\v1.0\powershell.exe", "powershell.exe"));
    ps.args([
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy", "Bypass",
        "-Command",
        "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing -TimeoutSec 600 -Uri $env:BABBLE_DL_URL -OutFile $env:BABBLE_DL_DEST",
    ])
    .env("BABBLE_DL_URL", url)
    .env("BABBLE_DL_DEST", &part);
    run_logged(handle, &mut ps, &format!("Downloading {url}"))?;
    if !part.is_file() {
        return Err(format!("Download from {url} did not produce a file."));
    }
    rename_with_retry(&part, dest)
}

#[cfg(target_os = "windows")]
fn extract_zip_win(handle: &tauri::AppHandle, zip: &Path, dest_dir: &Path) -> Result<(), String> {
    let _ = std::fs::create_dir_all(dest_dir);
    let mut tar = Command::new(system32_exe("tar.exe", "tar.exe"));
    tar.arg("-xf").arg(zip).arg("-C").arg(dest_dir);
    if run_logged(handle, &mut tar, "Extracting archive with tar.exe").is_ok() {
        return Ok(());
    }
    if QUITTING.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("Setup cancelled because Babble is closing.".into());
    }
    let mut ps = Command::new(system32_exe(r"WindowsPowerShell\v1.0\powershell.exe", "powershell.exe"));
    ps.args([
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy", "Bypass",
        "-Command",
        "$ProgressPreference='SilentlyContinue'; Expand-Archive -Force -Path $env:BABBLE_ZIP_SRC -DestinationPath $env:BABBLE_ZIP_DST",
    ])
    .env("BABBLE_ZIP_SRC", zip)
    .env("BABBLE_ZIP_DST", dest_dir);
    run_logged(handle, &mut ps, "Extracting archive")
}

#[cfg(target_os = "windows")]
fn find_file_recursive(dir: &Path, filename: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_file() {
            if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                if name.eq_ignore_ascii_case(filename) {
                    return Some(p);
                }
            }
        } else if p.is_dir() {
            if let Some(found) = find_file_recursive(&p, filename) {
                return Some(found);
            }
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn venv_valid(venv_dir: &Path, venv_py: &Path) -> bool {
    if !venv_dir.join("pyvenv.cfg").is_file() || !venv_py.is_file() {
        return false;
    }
    Command::new(venv_py)
        .args(["-c", "import sys"])
        .creation_flags(CREATE_NO_WINDOW)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(target_os = "windows")]
fn ensure_ffmpeg_win(handle: &tauri::AppHandle) {
    let ff_dir = data_dir().join("tools").join("ffmpeg");
    let ok_sentinel = ff_dir.join("ffmpeg.ok");
    if ok_sentinel.is_file() && ff_dir.join("ffmpeg.exe").is_file() && ff_dir.join("ffprobe.exe").is_file() {
        return;
    }
    if Command::new(system32_exe("where.exe", "where.exe"))
        .arg("ffmpeg.exe")
        .creation_flags(CREATE_NO_WINDOW)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        return;
    }
    set_engine_phase(handle, Some("Downloading audio tools (FFmpeg)… (first run only)"));
    note("downloading FFmpeg for Windows...");
    let zip = ff_dir.join("ffmpeg.zip");
    let unpack = ff_dir.join("_unpack");
    let _ = std::fs::remove_dir_all(&unpack);
    let url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip";
    if let Err(e) = download_file_win(handle, url, &zip) {
        note(&format!("FFmpeg download skipped (non-fatal): {e}"));
        return;
    }
    if let Err(e) = extract_zip_win(handle, &zip, &unpack) {
        note(&format!("FFmpeg extract skipped (non-fatal): {e}"));
        let _ = std::fs::remove_file(&zip);
        let _ = std::fs::remove_dir_all(&unpack);
        return;
    }
    let mut got_ff = false;
    let mut got_fp = false;
    if let Some(ff) = find_file_recursive(&unpack, "ffmpeg.exe") {
        let tmp = ff_dir.join("ffmpeg.exe.tmp");
        if std::fs::copy(&ff, &tmp).is_ok() && rename_with_retry(&tmp, &ff_dir.join("ffmpeg.exe")).is_ok() {
            got_ff = true;
        }
    }
    if let Some(fp) = find_file_recursive(&unpack, "ffprobe.exe") {
        let tmp = ff_dir.join("ffprobe.exe.tmp");
        if std::fs::copy(&fp, &tmp).is_ok() && rename_with_retry(&tmp, &ff_dir.join("ffprobe.exe")).is_ok() {
            got_fp = true;
        }
    }
    if got_ff && got_fp {
        let _ = std::fs::write(&ok_sentinel, "ok\n");
    }
    let _ = std::fs::remove_file(&zip);
    let _ = std::fs::remove_dir_all(&unpack);
}

/// First-run automatic Windows bootstrap: downloads uv, standalone Python 3.11, Babble packages, and FFmpeg.
/// What Babble's Python packages are installed from on Windows and Linux: requirements.txt and requirements-pc.txt
/// (the same text as narrator/pc_setup.py stamp()). Kept next to the installed packages; when an update changes it,
/// they're installed again.
fn requirements_stamp(root: &Path) -> Result<String, String> {
    let mut parts = Vec::new();
    for name in ["requirements.txt", "requirements-pc.txt"] {
        let f = root.join(name);
        parts.push(std::fs::read_to_string(&f).map_err(|e| format!("Couldn't read {}: {e}", f.display()))?);
    }
    Ok(parts.join("\n"))
}

#[cfg(target_os = "windows")]
fn ensure_windows_runtime(handle: &tauri::AppHandle, root: &Path) -> Result<(), String> {
    if !root.join("run.py").is_file() {
        return Err(format!(
            "Babble's engine files (run.py) weren't found in {}. Install the newest Babble for Windows over this one.",
            root.display()
        ));
    }
    // Respect explicit BABBLE_PYTHON or an existing repo-local .venv
    let custom_py = std::env::var("BABBLE_PYTHON").ok().map(|s| !s.trim().is_empty()).unwrap_or(false);
    let repo_venv = root.join(".venv").join("Scripts").join("python.exe").is_file();
    if !custom_py && !repo_venv {
        let data = data_dir();
        let venv_dir = data.join(".venv");
        let venv_py = venv_dir.join("Scripts").join("python.exe");
        let stamp_path = venv_dir.join(".babble-req-stamp");
        // what the packages are installed from (narrator/pc_setup.py): when an update changes it, they're installed again
        let filtered_req = requirements_stamp(root)?;
        let stamp_ok = venv_valid(&venv_dir, &venv_py)
            && std::fs::read_to_string(&stamp_path).map(|s| s == filtered_req).unwrap_or(false);
        if !stamp_ok {
            let uv_dir = data.join("tools").join("uv");
            let uv_exe = uv_dir.join("uv.exe");
            if !uv_exe.is_file() {
                set_engine_phase(handle, Some("Downloading Python installer (uv)…"));
                note("downloading uv.exe for first-run Python bootstrap...");
                let uv_zip = uv_dir.join("uv.zip");
                let url = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip";
                download_file_win(handle, url, &uv_zip)?;
                extract_zip_win(handle, &uv_zip, &uv_dir)?;
                let _ = std::fs::remove_file(&uv_zip);
                if !uv_exe.is_file() {
                    if let Some(found) = find_file_recursive(&uv_dir, "uv.exe") {
                        let tmp = uv_dir.join("uv.exe.tmp");
                        if std::fs::copy(&found, &tmp).is_ok() {
                            let _ = rename_with_retry(&tmp, &uv_exe);
                        }
                    }
                }
                if !uv_exe.is_file() {
                    return Err("Couldn't extract uv.exe for Python setup.".into());
                }
            }
            let rt_dir = data.join("rt");
            let py_install_dir = rt_dir.join("py");
            let uv_cache_dir = rt_dir.join("uvc");
            let create_venv = |clear_existing: bool| -> Result<(), String> {
                if clear_existing {
                    let _ = std::fs::remove_dir_all(&venv_dir);
                }
                set_engine_phase(handle, Some("Downloading Python 3.11… (first run only)"));
                note("creating standalone Python 3.11 virtual environment via uv...");
                let mut cmd = Command::new(&uv_exe);
                cmd.args(["venv", "--clear", "--python", "3.11", "--python-preference", "only-managed"])
                    .arg(&venv_dir)
                    .env("UV_PYTHON_INSTALL_DIR", &py_install_dir)
                    .env("UV_CACHE_DIR", &uv_cache_dir)
                    .env("UV_LINK_MODE", "copy");
                run_logged(handle, &mut cmd, "Setting up Python 3.11")
            };
            let had_venv = venv_valid(&venv_dir, &venv_py);
            if !had_venv {
                create_venv(venv_dir.exists())?;
            }
            let install_pkgs = || -> Result<(), String> {
                set_engine_phase(handle, Some("Installing Babble packages… (a few minutes, first run only)"));
                note("installing Python packages (narrator/pc_setup.py base)...");
                let mut pip = Command::new(&venv_py);
                pip.args(["-m", "narrator.pc_setup", "base", "--uv"])
                    .arg(&uv_exe)
                    .current_dir(root)
                    .env("PYTHONDONTWRITEBYTECODE", "1")
                    .env("UV_PYTHON_INSTALL_DIR", &py_install_dir)
                    .env("UV_CACHE_DIR", &uv_cache_dir)
                    .env("UV_LINK_MODE", "copy");
                run_logged(handle, &mut pip, "Installing Python packages")
            };
            if let Err(first_err) = install_pkgs() {
                if had_venv && !QUITTING.load(std::sync::atomic::Ordering::SeqCst) {
                    note(&format!("uv pip install failed on existing .venv ({first_err}); recreating cleanly..."));
                    create_venv(true)?;
                    install_pkgs()?;
                } else {
                    return Err(first_err);
                }
            }
            let stamp_tmp = venv_dir.join(".babble-req-stamp.tmp");
            if std::fs::write(&stamp_tmp, &filtered_req).is_ok() {
                let _ = rename_with_retry(&stamp_tmp, &stamp_path);
            }
        }
    }
    ensure_ffmpeg_win(handle);
    Ok(())
}

// ---------- Linux: the first start sets up Python like Windows does (uv, a Python 3.11 of its own, Babble's packages) ----------

#[cfg(unix)]
fn run_logged(handle: &tauri::AppHandle, cmd: &mut Command, what: &str) -> Result<(), String> {
    if QUITTING.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("Setup cancelled because Babble is closing.".into());
    }
    let log = OpenOptions::new().create(true).append(true).open(log_path()).map_err(|e| format!("Couldn't open Babble.log: {e}"))?;
    let err = log.try_clone().map_err(|e| format!("Couldn't clone log handle: {e}"))?;
    let mut child = cmd.stdout(log).stderr(err).spawn().map_err(|e| format!("{what} failed to launch: {e}"))?;
    *handle.state::<BootstrapPid>().0.lock().unwrap() = Some(child.id());
    let status = child.wait().map_err(|e| format!("{what} wait failed: {e}"));
    *handle.state::<BootstrapPid>().0.lock().unwrap() = None;
    if !status?.success() {
        return Err(format!("{what} failed. Check Babble.log for details."));
    }
    Ok(())
}

#[cfg(unix)]
fn find_named(dir: &Path, filename: &str) -> Option<PathBuf> {
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let p = entry.path();
        if p.is_file() && p.file_name().and_then(|n| n.to_str()) == Some(filename) {
            return Some(p);
        }
        if p.is_dir() {
            if let Some(found) = find_named(&p, filename) {
                return Some(found);
            }
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn ensure_ffmpeg_mac(handle: &tauri::AppHandle) {
    use std::os::unix::fs::PermissionsExt;
    let ff_dir = data_dir().join("tools").join("ffmpeg");
    let have = |dir: &Path| dir.join("ffmpeg").is_file() && dir.join("ffprobe").is_file();
    if have(&ff_dir) || have(Path::new("/opt/homebrew/bin")) || have(Path::new("/usr/local/bin")) {
        return;
    }
    set_engine_phase(handle, Some("Downloading audio tools (FFmpeg)… (first run only)"));
    note("downloading FFmpeg for the Mac...");
    let _ = std::fs::create_dir_all(&ff_dir);
    let arch = if cfg!(target_arch = "aarch64") { "arm64" } else { "amd64" };
    for tool in ["ffmpeg", "ffprobe"] {
        let zip = ff_dir.join(format!("{tool}.zip"));
        let url = format!("https://ffmpeg.martin-riedl.de/redirect/latest/macos/{arch}/release/{tool}.zip");
        let mut curl = Command::new("curl");
        curl.args(["-fsSL", "--connect-timeout", "20", "--retry", "3", "-o"]).arg(&zip).arg(&url);
        if let Err(e) = run_logged(handle, &mut curl, &format!("Downloading {tool}")) {
            note(&format!("FFmpeg download skipped (non-fatal): {e}"));
            return;
        }
        let mut unzip = Command::new("/usr/bin/ditto");
        unzip.args(["-x", "-k"]).arg(&zip).arg(&ff_dir);
        let _ = run_logged(handle, &mut unzip, &format!("Unpacking {tool}"));
        let _ = std::fs::remove_file(&zip);
        let exe = ff_dir.join(tool);
        let _ = std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755));
        let _ = Command::new("/usr/bin/xattr").args(["-d", "com.apple.quarantine"]).arg(&exe).status();
    }
}

/// First start on the Mac and Linux: uv, Python 3.11 and Babble's packages in the data folder (and FFmpeg on the
/// Mac), again when an update changes the requirements. A developer's checkout with its own .venv skips all this.
#[cfg(unix)]
fn ensure_unix_runtime(handle: &tauri::AppHandle, root: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    if !root.join("run.py").is_file() {
        return Err(format!(
            "Babble's engine files (run.py) weren't found in {}. Install the newest Babble.",
            root.display()
        ));
    }
    #[cfg(target_os = "macos")]
    ensure_ffmpeg_mac(handle);
    let custom_py = std::env::var("BABBLE_PYTHON").ok().map(|s| !s.trim().is_empty()).unwrap_or(false);
    if custom_py || root.join(".venv/bin/python").is_file() {
        return Ok(()); // (a developer's own checkout)
    }
    let data = data_dir();
    let venv_dir = data.join(".venv");
    let venv_py = venv_dir.join("bin/python");
    let stamp_path = venv_dir.join(".babble-req-stamp");
    let stamp = requirements_stamp(root)?;
    let python_ok = |py: &Path| Command::new(py).args(["-c", "import sys"]).status().map(|s| s.success()).unwrap_or(false);
    if venv_py.is_file() && python_ok(&venv_py) && std::fs::read_to_string(&stamp_path).map(|s| s == stamp).unwrap_or(false) {
        return Ok(());
    }
    let uv_dir = data.join("tools/uv");
    let uv = uv_dir.join("uv");
    if !uv.is_file() {
        set_engine_phase(handle, Some("Downloading Python installer (uv)…"));
        let _ = std::fs::create_dir_all(&uv_dir);
        let archive = uv_dir.join("uv.tar.gz");
        let build = if cfg!(target_os = "macos") {
            if cfg!(target_arch = "aarch64") { "aarch64-apple-darwin" } else { "x86_64-apple-darwin" }
        } else if cfg!(target_arch = "aarch64") {
            "aarch64-unknown-linux-gnu"
        } else {
            "x86_64-unknown-linux-gnu"
        };
        let url = format!("https://github.com/astral-sh/uv/releases/latest/download/uv-{build}.tar.gz");
        let url = url.as_str();
        let mut curl = Command::new("curl");
        curl.args(["-fsSL", "--connect-timeout", "20", "--retry", "3", "-o"]).arg(&archive).arg(url);
        if run_logged(handle, &mut curl, "Downloading uv with curl").is_err() {
            let mut wget = Command::new("wget");
            wget.args(["-q", "-O"]).arg(&archive).arg(url);
            run_logged(handle, &mut wget, "Downloading uv with wget")
                .map_err(|_| "Couldn't download uv: Babble needs curl or wget and an internet connection the first time.".to_string())?;
        }
        let mut tar = Command::new("tar");
        tar.arg("-xzf").arg(&archive).arg("-C").arg(&uv_dir);
        run_logged(handle, &mut tar, "Extracting uv")?;
        let _ = std::fs::remove_file(&archive);
        if let Some(found) = find_named(&uv_dir, "uv") {
            if found != uv {
                std::fs::copy(&found, &uv).map_err(|e| format!("Couldn't put uv in place: {e}"))?;
            }
        }
        let _ = std::fs::set_permissions(&uv, std::fs::Permissions::from_mode(0o755));
        if !uv.is_file() {
            return Err("Couldn't extract uv for Python setup.".into());
        }
    }
    let py_install_dir = data.join("rt/py");
    let uv_cache_dir = data.join("rt/uvc");
    if !(venv_py.is_file() && python_ok(&venv_py)) {
        set_engine_phase(handle, Some("Downloading Python 3.11… (first run only)"));
        let mut venv = Command::new(&uv);
        venv.args(["venv", "--clear", "--python", "3.11", "--python-preference", "only-managed"])
            .arg(&venv_dir)
            .env("UV_PYTHON_INSTALL_DIR", &py_install_dir)
            .env("UV_CACHE_DIR", &uv_cache_dir);
        run_logged(handle, &mut venv, "Setting up Python 3.11")?;
    }
    set_engine_phase(handle, Some("Installing Babble packages… (a few minutes, first run only)"));
    let mut pip = Command::new(&venv_py);
    // the Mac's voices run on MLX (requirements.txt as it is); Windows and Linux use PyTorch (narrator/pc_setup.py)
    pip.args(["-m", "narrator.pc_setup", if cfg!(target_os = "macos") { "mac" } else { "base" }, "--uv"])
        .arg(&uv)
        .current_dir(root)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("UV_PYTHON_INSTALL_DIR", &py_install_dir)
        .env("UV_CACHE_DIR", &uv_cache_dir);
    run_logged(handle, &mut pip, "Installing Python packages")?;
    let tmp = venv_dir.join(".babble-req-stamp.tmp");
    if std::fs::write(&tmp, &stamp).is_ok() {
        let _ = std::fs::rename(&tmp, &stamp_path);
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn start_engine() -> Result<Option<Child>, String> {
    if engine_running() {
        return Ok(None);
    }
    let root = project_root();
    if !root.join("run.py").is_file() {
        return Err(format!(
            "Babble's engine files (run.py) weren't found in {}. Install the newest Babble for Windows over this one.",
            root.display()
        ));
    }
    let log = OpenOptions::new().create(true).append(true).open(log_path()).map_err(|e| format!("Couldn't open Babble.log: {e}"))?;
    let err = log.try_clone().map_err(|e| format!("Couldn't clone log handle: {e}"))?;
    let py = python_exe(&root);
    if py == Path::new("python.exe") {
        if let Ok(out) = Command::new(system32_exe("where.exe", "where.exe")).arg("python.exe").creation_flags(CREATE_NO_WINDOW).output() {
            let found = String::from_utf8_lossy(&out.stdout);
            let first = found.lines().next().unwrap_or("").trim();
            if first.is_empty() || first.to_ascii_lowercase().contains("windowsapps") {
                return Err(
                    "Python 3.11 (.venv) wasn't found. Click Retry setup with an internet connection to finish first-time setup.".into()
                );
            }
        }
    }
    let data = data_dir();
    let _ = std::fs::create_dir_all(&data);
    let mut cmd = Command::new(&py);
    cmd.arg(root.join("run.py"))
        .args(["--port", &PORT.to_string()])
        .current_dir(&data)
        .env("NARRATOR_HOME", &data)
        .env("BABBLE_APP_VERSION", env!("CARGO_PKG_VERSION"))
        .env("PYTHONUTF8", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .creation_flags(CREATE_NO_WINDOW)
        .stdout(log)
        .stderr(err);
    let ff_dir = data_dir().join("tools").join("ffmpeg");
    if ff_dir.is_dir() {
        let old_path = std::env::var("PATH").unwrap_or_default();
        cmd.env("PATH", format!("{};{old_path}", ff_dir.display()));
    }
    let vault_setup = match (TcpListener::bind("127.0.0.1:0"), random_hex_nonce()) {
        (Ok(lst), Ok(nonce)) => Some((lst, nonce)),
        _ => None,
    };
    if vault_setup.is_some() {
        cmd.env("BABBLE_VAULT", "1").stdin(Stdio::piped());
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Couldn't start Python ({}) in {}: {e}", py.display(), root.display()))?;
    if let (Some((lst, nonce)), Some(stdin)) = (vault_setup, child.stdin.take()) {
        std::thread::spawn(move || vault_serve_win(stdin, lst, nonce));
    }
    Ok(Some(child))
}

#[cfg(all(unix, not(target_os = "macos")))]
fn start_engine() -> Result<Option<Child>, String> {
    if engine_running() {
        return Ok(None);
    }
    let root = project_root();
    if !root.join("run.py").is_file() {
        return Err(format!("Babble's engine files (run.py) weren't found in {}. Set NARRATOR_ROOT.", root.display()));
    }
    let log = OpenOptions::new().create(true).append(true).open(log_path()).map_err(|e| format!("Couldn't open Babble.log: {e}"))?;
    let err = log.try_clone().map_err(|e| format!("Couldn't clone log handle: {e}"))?;
    let py = python_exe(&root);
    let mut cmd = Command::new(&py);
    cmd.args(["run.py", "--port", &PORT.to_string()])
        .current_dir(&root)
        .env_remove("NARRATOR_HOME")
        .env("BABBLE_APP_VERSION", env!("CARGO_PKG_VERSION"))
        .env("PYTHONDONTWRITEBYTECODE", "1") // (the engine's files can be read-only: an AppImage, /usr/lib)
        .stdout(log)
        .stderr(err);
    unsafe {
        cmd.pre_exec(|| {
            let _ = libc::setpgid(0, 0);
            Ok(())
        });
    }
    let child = cmd.spawn().map_err(|e| format!("Couldn't start Python ({}): {e}", py.display()))?;
    Ok(Some(child))
}

/// Ask the engine's entire process tree (Python + grandchild ffmpeg/workers) to stop cleanly, then force-kill.
fn stop_engine(child: &mut Child) {
    #[cfg(unix)]
    {
        let pgid = child.id() as libc::pid_t;
        unsafe { libc::killpg(pgid, libc::SIGTERM) };
        let deadline = Instant::now() + Duration::from_secs(6);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(150));
        }
        unsafe { libc::killpg(pgid, libc::SIGKILL) };
        let _ = child.kill();
        let _ = child.wait();
    }
    #[cfg(target_os = "windows")]
    {
        let pid = child.id().to_string();
        let tk = system32_exe("taskkill.exe", "taskkill.exe");
        // Ask nicely first (no /F), wait up to 3s for in-flight JSON/file writes, then force-kill tree.
        let _ = Command::new(&tk)
            .args(["/PID", &pid, "/T"])
            .creation_flags(CREATE_NO_WINDOW)
            .status();
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(150));
        }
        let _ = Command::new(&tk)
            .args(["/PID", &pid, "/T", "/F"])
            .creation_flags(CREATE_NO_WINDOW)
            .status();
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// Allowed roots for save_file (matches narrator/api.py _roots): only Babble's own export/output/media subfolders,
/// never the top-level data dir containing engine-token, settings.json, or spending.jsonl.
fn allowed_save_roots() -> Vec<PathBuf> {
    let base = data_dir();
    let roots = vec![
        base.join("outputs"),
        base.join("sfx"),
        base.join("sounds"),
        base.join("voices"),
        base.join("training"),
        base.join("train-work").join("results"),
        base.join("work").join("results"),
        base.join("models"),
        project_root().join("outputs"),
    ];
    roots.into_iter().filter_map(|p| std::fs::canonicalize(p).ok()).collect()
}

fn path_within_root(src: &Path, root: &Path) -> bool {
    #[cfg(target_os = "windows")]
    {
        let mut s = src.components();
        for rc in root.components() {
            let Some(sc) = s.next() else { return false };
            if sc.as_os_str().to_string_lossy().to_ascii_lowercase()
                != rc.as_os_str().to_string_lossy().to_ascii_lowercase()
            {
                return false;
            }
        }
        true
    }
    #[cfg(not(target_os = "windows"))]
    {
        src.starts_with(root)
    }
}

fn reveal_file(dest: &Path) {
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg("-R").arg(dest).status();
    }
    #[cfg(target_os = "windows")]
    {
        let s = dest.to_string_lossy();
        let clean = s.strip_prefix(r"\\?\").unwrap_or(&s);
        if !clean.contains('"') {
            let _ = Command::new("explorer.exe").raw_arg(format!("/select,\"{clean}\"")).status();
        }
    }
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    {
        if let Some(parent) = dest.parent() {
            let _ = Command::new("xdg-open").arg(parent).status();
        }
    }
}

fn open_url_in_browser(url: &str) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg("--").arg(url).status().map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::UI::Shell::ShellExecuteW;
        let op = to_wide("open");
        let target = to_wide(url);
        let res = unsafe {
            ShellExecuteW(
                std::ptr::null_mut(),
                op.as_ptr(),
                target.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                1, // SW_SHOWNORMAL
            )
        } as isize;
        if res <= 32 {
            return Err(format!("Couldn't open browser (ShellExecuteW {res})."));
        }
    }
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    {
        Command::new("xdg-open").arg(url).status().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Copy a file the engine made to Downloads (a fresh name if one is there) and reveal it in Finder/Explorer.
#[tauri::command]
fn save_file(path: String) -> Result<String, String> {
    let src = std::fs::canonicalize(&path).map_err(|_| "That file isn't there any more.".to_string())?;
    if !src.is_file() {
        return Err("That file isn't there any more.".into());
    }
    let roots = allowed_save_roots();
    if !roots.iter().any(|r| path_within_root(&src, r)) {
        return Err("Only files created by Babble can be saved.".into());
    }
    let downloads = home().join("Downloads");
    let _ = std::fs::create_dir_all(&downloads);
    let name = src.file_name().ok_or("No file name")?.to_string_lossy().to_string();
    let (stem, ext) = match name.rsplit_once('.') {
        Some((s, e)) => (s.to_string(), format!(".{e}")),
        None => (name.clone(), String::new()),
    };
    let mut dest = downloads.join(&name);
    let mut n = 2;
    while dest.exists() {
        dest = downloads.join(format!("{stem} {n}{ext}"));
        n += 1;
    }
    std::fs::copy(&src, &dest).map_err(|e| format!("Couldn't save it: {e}"))?;
    reveal_file(&dest);
    Ok(dest.to_string_lossy().to_string())
}

/// The engine's key for this run (it writes it at start, readable only by you): the window sends it with every call,
/// so other programs and web pages can't use the engine (narrator/guard.py).
#[tauri::command]
fn engine_token() -> Result<String, String> {
    let path = data_dir().join("engine-token");
    std::fs::read_to_string(path)
        .map(|t| t.trim().to_string())
        .map_err(|_| "The engine hasn't started yet.".into())
}

/// Current engine startup status, bootstrap phase, and recent log tail, so Splash surfaces progress & failures.
#[tauri::command]
fn engine_status(app: tauri::AppHandle) -> serde_json::Value {
    let err = app.state::<EngineError>().0.lock().unwrap().clone();
    let phase = app.state::<EnginePhase>().0.lock().unwrap().clone();
    let state = if err.is_some() {
        "failed"
    } else if app.state::<Engine>().0.lock().unwrap().is_some() {
        "running"
    } else {
        "starting"
    };
    serde_json::json!({
        "state": state,
        "phase": phase,
        "error": err,
        "log_path": log_path().to_string_lossy().to_string(),
        "log_tail": read_log_tail(14),
    })
}

/// Reveal Babble.log (or its folder) in Finder / File Explorer.
#[tauri::command]
fn open_log_folder() -> Result<(), String> {
    let p = log_path();
    if p.exists() {
        reveal_file(&p);
    } else if let Some(parent) = p.parent() {
        let _ = std::fs::create_dir_all(parent);
        reveal_file(parent);
    }
    Ok(())
}

// ---------- updates: narrator/updater.py downloads a release's installer, the app puts it in place ----------

/// "0.2.0" > "0.2.0-beta.2" > "0.2.0-beta.1" > "0.1.9" (as narrator/updater.py version_key).
fn version_key(v: &str) -> (Vec<u64>, u64, Vec<u64>) {
    let v = v.trim().trim_start_matches(['v', 'V']);
    let (main, pre) = v.split_once('-').unwrap_or((v, ""));
    let nums = |t: &str| t.split(|c: char| !c.is_ascii_digit()).filter(|x| !x.is_empty()).filter_map(|x| x.parse().ok()).collect::<Vec<u64>>();
    let mut m = nums(main);
    m.truncate(3);
    m.resize(3, 0);
    (m, if pre.is_empty() { 1 } else { 0 }, nums(pre))
}

/// The downloaded update (<data>/updates/ready.json): its installer, and whether to install it when Babble quits.
fn ready_update() -> Option<(PathBuf, bool)> {
    let dir = data_dir().join("updates");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("ready.json")).ok()?).ok()?;
    let file = v.get("file")?.as_str()?;
    if file.is_empty() || file.contains(['/', '\\']) || file.starts_with('.') {
        return None;
    }
    if version_key(v.get("version")?.as_str()?) <= version_key(env!("CARGO_PKG_VERSION")) {
        return None;
    }
    let path = dir.join(file);
    path.is_file().then(|| (path, v.get("on_quit").and_then(|b| b.as_bool()).unwrap_or(false)))
}

/// Start the program that waits for Babble to close, puts the new version in its place and (reopen) opens it.
#[cfg(target_os = "macos")]
fn start_install(installer: &Path, reopen: bool) -> Result<(), String> {
    let app = app_bundle_path().ok_or("Babble isn't running from Babble.app, so it can't update itself.")?;
    let parent = app.parent().ok_or("Couldn't tell where Babble is.")?.to_path_buf();
    let probe = parent.join(".babble-update-check");
    if std::fs::write(&probe, b"").is_err() {
        return Err(format!(
            "Babble can't replace itself in {}. Move Babble to Applications or your Desktop, or download the update from its release page.",
            parent.display()
        ));
    }
    let _ = std::fs::remove_file(&probe);
    let updates = data_dir().join("updates");
    let stage = updates.join("new");
    let _ = std::fs::remove_dir_all(&stage);
    std::fs::create_dir_all(&stage).map_err(|e| format!("Couldn't unpack the update: {e}"))?;
    let ok = Command::new("/usr/bin/ditto").args(["-x", "-k"]).arg(installer).arg(&stage).status().map(|s| s.success()).unwrap_or(false);
    let new_app = std::fs::read_dir(&stage).ok().and_then(|d| {
        d.flatten().map(|e| e.path()).find(|p| p.extension().and_then(|e| e.to_str()) == Some("app"))
    });
    let Some(new_app) = new_app.filter(|_| ok) else {
        return Err("Couldn't unpack the update. Download it again from Settings.".into());
    };
    let _ = Command::new("/usr/bin/xattr").args(["-dr", "com.apple.quarantine"]).arg(&new_app).status();
    let script = updates.join("install.sh");
    std::fs::write(&script, "#!/bin/sh\n# Babble's update: wait for Babble to close, put the new version in its place, open it.\n\
        while /bin/kill -0 \"$1\" 2>/dev/null; do sleep 0.3; done\nsleep 1\n\
        APP=\"$2\"; NEW=\"$3\"\nrm -rf \"$APP.old\"\n\
        if mv \"$APP\" \"$APP.old\"; then\n  if mv \"$NEW\" \"$APP\"; then rm -rf \"$APP.old\"; rm -f \"$5/ready.json\" \"$6\"; echo \"[update] installed\";\n\
          else mv \"$APP.old\" \"$APP\"; echo \"[update] couldn't put the new version in place\"; fi\nfi\n\
        [ \"$4\" = \"1\" ] && /usr/bin/open \"$APP\"\nexit 0\n")
        .map_err(|e| format!("Couldn't write the update program: {e}"))?;
    let log = OpenOptions::new().create(true).append(true).open(log_path()).map_err(|e| e.to_string())?;
    let err = log.try_clone().map_err(|e| e.to_string())?;
    let mut cmd = Command::new("/bin/sh");
    cmd.arg(&script)
        .arg(std::process::id().to_string())
        .arg(&app)
        .arg(&new_app)
        .arg(if reopen { "1" } else { "0" })
        .arg(&updates)
        .arg(installer)
        .stdin(Stdio::null())
        .stdout(log)
        .stderr(err);
    // SAFETY: setsid only (async-signal-safe): it outlives Babble, which it waits for
    unsafe {
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    cmd.spawn().map_err(|e| format!("Couldn't start the update: {e}"))?;
    Ok(())
}

#[cfg(target_os = "windows")]
fn start_install(installer: &Path, reopen: bool) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let q = |p: &Path| p.display().to_string().replace('\'', "''");
    let ready = data_dir().join("updates").join("ready.json");
    let again = if reopen { format!("; Start-Process -FilePath '{}'", q(exe.as_path())) } else { String::new() };
    let script = format!(
        "$p={}; while (Get-Process -Id $p -ErrorAction SilentlyContinue) {{ Start-Sleep -Milliseconds 300 }}; \
         Start-Process -Wait -FilePath '{}' -ArgumentList '/S'; Remove-Item -Force -ErrorAction SilentlyContinue '{}', '{}'{}",
        std::process::id(), q(installer), q(ready.as_path()), q(installer), again
    );
    Command::new(system32_exe("WindowsPowerShell\\v1.0\\powershell.exe", "powershell.exe"))
        .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-Command", &script])
        .creation_flags(CREATE_NO_WINDOW | 0x0000_0200) // (and its own process group: it outlives Babble)
        .spawn()
        .map_err(|e| format!("Couldn't start the update: {e}"))?;
    Ok(())
}

#[cfg(all(unix, not(target_os = "macos")))]
fn start_install(installer: &Path, reopen: bool) -> Result<(), String> {
    if installer.extension().and_then(|e| e.to_str()) == Some("deb") {
        let _ = Command::new("xdg-open").arg(installer).spawn();
        return Err("The update is downloaded and opened in your software installer: install it there, then open Babble again.".into());
    }
    let target = std::env::var("APPIMAGE").map_err(|_| "Babble can only update itself as an AppImage. Download the update from its release page.".to_string())?;
    let ready = data_dir().join("updates").join("ready.json");
    let script = format!(
        "while kill -0 {pid} 2>/dev/null; do sleep 0.3; done; cp \"$1\" \"$2.new\" && chmod +x \"$2.new\" && mv \"$2.new\" \"$2\" && rm -f \"$3\" \"$1\"; {again}",
        pid = std::process::id(),
        again = if reopen { "nohup \"$2\" >/dev/null 2>&1 &" } else { "" }
    );
    let mut cmd = Command::new("/bin/sh");
    cmd.args(["-c", &script, "babble-update"]).arg(installer).arg(&target).arg(&ready);
    // SAFETY: setsid only (async-signal-safe): it outlives Babble, which it waits for
    unsafe {
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    cmd.spawn().map_err(|e| format!("Couldn't start the update: {e}"))?;
    Ok(())
}

/// Restart to update: Babble quits, the new version is put in its place, and it opens again.
#[tauri::command]
fn apply_update(app: tauri::AppHandle) -> Result<(), String> {
    let (installer, _) = ready_update().ok_or("There's no downloaded update.")?;
    start_install(&installer, true)?;
    UPDATING.store(true, std::sync::atomic::Ordering::SeqCst);
    note("quitting to install an update");
    app.exit(0);
    Ok(())
}

/// With "Update automatically" on, a downloaded update is installed when Babble quits.
fn install_at_quit() {
    if UPDATING.load(std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    let Some((installer, true)) = ready_update() else { return };
    #[cfg(all(unix, not(target_os = "macos")))]
    if std::env::var("APPIMAGE").is_err() {
        return; // (a .deb is installed by the person, not at quit)
    }
    match start_install(&installer, false) {
        Ok(()) => note("installing the downloaded update now that Babble is quitting"),
        Err(e) => note(&format!("the downloaded update wasn't installed: {e}")),
    }
}

/// Retry first-run bootstrap and engine launch after a failure.
#[tauri::command]
fn retry_engine(app: tauri::AppHandle) -> Result<(), String> {
    set_engine_error(&app, None);
    set_engine_phase(&app, Some("Retrying setup…"));
    watch_engine(app);
    Ok(())
}

#[cfg(target_os = "macos")]
const ICON_LIGHT: &[u8] = include_bytes!("../icons/icon.png");
#[cfg(target_os = "macos")]
const ICON_DARK: &[u8] = include_bytes!("../icons/icon-dark.png");

#[cfg(target_os = "macos")]
fn app_bundle_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let bundle = exe.parent()?.parent()?.parent()?.to_path_buf();
    (bundle.extension().and_then(|e| e.to_str()) == Some("app")).then_some(bundle)
}

#[cfg(target_os = "macos")]
fn apply_macos_icon(bytes: &'static [u8]) {
    use std::ffi::{c_char, c_void, CString};
    extern "C" {
        fn objc_getClass(name: *const c_char) -> *mut c_void;
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        fn objc_msgSend();
    }
    unsafe {
        let send0: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
            std::mem::transmute(objc_msgSend as *const c_void);
        let send1: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void =
            std::mem::transmute(objc_msgSend as *const c_void);
        let send_bytes: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void, usize) -> *mut c_void =
            std::mem::transmute(objc_msgSend as *const c_void);
        let send_cstr: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_char) -> *mut c_void =
            std::mem::transmute(objc_msgSend as *const c_void);
        let send_set_icon: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void, usize) -> i8 =
            std::mem::transmute(objc_msgSend as *const c_void);

        let cls_data = objc_getClass(b"NSData\0".as_ptr() as *const c_char);
        let sel_data = sel_registerName(b"dataWithBytes:length:\0".as_ptr() as *const c_char);
        let ns_data = send_bytes(cls_data, sel_data, bytes.as_ptr() as *const c_void, bytes.len());
        if ns_data.is_null() { return; }

        let cls_img = objc_getClass(b"NSImage\0".as_ptr() as *const c_char);
        let sel_alloc = sel_registerName(b"alloc\0".as_ptr() as *const c_char);
        let sel_init = sel_registerName(b"initWithData:\0".as_ptr() as *const c_char);
        let img = send1(send0(cls_img, sel_alloc), sel_init, ns_data);
        if img.is_null() { return; }

        let cls_app = objc_getClass(b"NSApplication\0".as_ptr() as *const c_char);
        let sel_shared_app = sel_registerName(b"sharedApplication\0".as_ptr() as *const c_char);
        let sel_set_app_icon = sel_registerName(b"setApplicationIconImage:\0".as_ptr() as *const c_char);
        let ns_app = send0(cls_app, sel_shared_app);
        if !ns_app.is_null() {
            let _ = send1(ns_app, sel_set_app_icon, img);
        }

        if let Some(bundle) = app_bundle_path() {
            if let Ok(c_path) = CString::new(bundle.to_string_lossy().as_bytes()) {
                let cls_str = objc_getClass(b"NSString\0".as_ptr() as *const c_char);
                let sel_utf8 = sel_registerName(b"stringWithUTF8String:\0".as_ptr() as *const c_char);
                let ns_path = send_cstr(cls_str, sel_utf8, c_path.as_ptr());
                let cls_ws = objc_getClass(b"NSWorkspace\0".as_ptr() as *const c_char);
                let sel_shared_ws = sel_registerName(b"sharedWorkspace\0".as_ptr() as *const c_char);
                let sel_set_file_icon = sel_registerName(b"setIcon:forFile:options:\0".as_ptr() as *const c_char);
                let ws = send0(cls_ws, sel_shared_ws);
                if !ws.is_null() && !ns_path.is_null() {
                    let _ = send_set_icon(ws, sel_set_file_icon, img, ns_path, 0);
                }
            }
        }

        let sel_release = sel_registerName(b"release\0".as_ptr() as *const c_char);
        let _ = send0(img, sel_release);
    }
}

/// Switch the live Dock icon and outside .app bundle icon to match the active theme ("light" | "dark").
#[tauri::command]
fn set_app_icon(app: tauri::AppHandle, theme: String) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let bytes = if theme == "light" { ICON_LIGHT } else { ICON_DARK };
        app.run_on_main_thread(move || apply_macos_icon(bytes)).map_err(|e| e.to_string())?;
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (app, theme);
    Ok(())
}

/// Open a web page in the browser (only public http and https).
#[tauri::command]
fn open_external(url: String) -> Result<(), String> {
    let parsed = tauri::Url::parse(&url).map_err(|_| "Invalid web address.".to_string())?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err("Only web pages can be opened.".into());
    }
    if let Some(host) = parsed.host_str() {
        let h = host.to_ascii_lowercase();
        if h == "localhost" || h == "127.0.0.1" || h == "[::1]" || h == "::1" || h == "0.0.0.0" || h.ends_with(".localhost") {
            return Err("Local addresses can't be opened externally.".into());
        }
    }
    open_url_in_browser(parsed.as_str())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![save_file, open_external, engine_token, set_app_icon, engine_status, open_log_folder, retry_engine, apply_update])
        .setup(|app| {
            app.manage(Engine(Mutex::new(None)));
            app.manage(EngineError(Mutex::new(None)));
            app.manage(EnginePhase(Mutex::new(None)));
            app.manage(BootstrapPid(Mutex::new(None)));
            watch_engine(app.handle().clone());
            let builder = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("Babble")
                .inner_size(1280.0, 840.0)
                .min_inner_size(900.0, 620.0)
                // the page's own drag and drop (sounds into a story, files onto drop zones) instead of the window's
                .disable_drag_drop_handler()
                // stay on the interface: anything else opens in the browser
                .on_navigation(|url| {
                    let ours = matches!(url.scheme(), "tauri" | "asset") || url.host_str() == Some("tauri.localhost")
                        || url.host_str() == Some("localhost");
                    if !ours && matches!(url.scheme(), "http" | "https") {
                        let _ = open_url_in_browser(url.as_str());
                    }
                    ours
                });
            #[cfg(target_os = "macos")]
            let builder = builder.title_bar_style(tauri::TitleBarStyle::Overlay).hidden_title(true);
            builder.build()?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("couldn't start Babble");

    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            QUITTING.store(true, std::sync::atomic::Ordering::SeqCst);
            install_at_quit();
            #[cfg(target_os = "windows")]
            if let Some(pid) = handle.state::<BootstrapPid>().0.lock().unwrap().take() {
                let _ = Command::new(system32_exe("taskkill.exe", "taskkill.exe"))
                    .args(["/PID", &pid.to_string(), "/T", "/F"])
                    .creation_flags(CREATE_NO_WINDOW)
                    .status();
            }
            if let Some(mut child) = handle.state::<Engine>().0.lock().unwrap().take() {
                stop_engine(&mut child);
            }
        }
    });
}
