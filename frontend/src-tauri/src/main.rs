// Avalon desktop shell (Tauri). Port of the former Electron main process
// (electron/main.cjs): owns the Python dashboard backend sidecar, the tray,
// the API key, and the two IPC commands the renderer uses.
//
// One deliberate difference: closing the window DESTROYS the webview instead
// of hiding it. Electron kept a hidden ~2.3 GB renderer (plus its 1 s pollers
// and the Chromium network/GPU processes) alive for as long as the tray ran.
// The backend and tray keep running; "Show Avalon" rebuilds the window, and
// all dashboard state is re-read from the backend on load.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde_json::{json, Value};
use tauri::image::Image;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};

const DASHBOARD_HOST: &str = "127.0.0.1";
const MAIN_WINDOW: &str = "main";

#[derive(Default)]
struct Backend {
    child: Mutex<Option<Child>>,
    owns: AtomicBool,
    stopping: AtomicBool,
}

fn env_port(name: &str, default: u16) -> u16 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn dashboard_port() -> u16 {
    env_port("AVALON_DASHBOARD_PORT", 8771)
}

fn gateway_port() -> u16 {
    env_port("AVALON_GATEWAY_PORT", 8787)
}

fn dashboard_url(path: &str) -> String {
    format!("http://{DASHBOARD_HOST}:{}{path}", dashboard_port())
}

fn is_packaged() -> bool {
    !cfg!(debug_assertions)
}

/// Electron's `app.getPath('userData')` was `<appData>/avalon`. Keep using it
/// so the existing API key and downloaded models carry over.
fn user_data_dir(app: &AppHandle) -> PathBuf {
    app.path()
        .config_dir()
        .expect("no config dir")
        .join("avalon")
}

/// Repository root (dev only): src-tauri lives at <root>/frontend/src-tauri.
fn repo_root() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    root.canonicalize().unwrap_or(root)
}

fn sidecar_path(app: &AppHandle, name: &str) -> Option<PathBuf> {
    if !is_packaged() {
        return None;
    }
    let file = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    let path = app.path().resource_dir().ok()?.join("sidecars").join(file);
    path.exists().then_some(path)
}

fn api_key(app: &AppHandle) -> String {
    let file = user_data_dir(app).join("api-key");
    if let Ok(value) = fs::read_to_string(&file) {
        let value = value.trim();
        if !value.is_empty() {
            return value.to_string();
        }
    }
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes).expect("random bytes");
    let value = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes);
    if let Some(dir) = file.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let _ = fs::write(&file, format!("{value}\n"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&file, fs::Permissions::from_mode(0o600));
    }
    value
}

// MARK: - Logging
//
// A Finder-launched app has no terminal, so eprintln! goes nowhere. Shell
// messages go to shell.log and backend stdout/stderr to backend.log, both
// in ~/Library/Logs/com.avalon.llamadash (platform log dir elsewhere).

fn log_dir(app: &AppHandle) -> PathBuf {
    let dir = app
        .path()
        .app_log_dir()
        .unwrap_or_else(|_| user_data_dir(app).join("logs"));
    let _ = fs::create_dir_all(&dir);
    dir
}

fn log(app: &AppHandle, message: impl AsRef<str>) {
    use std::io::Write;
    let message = message.as_ref();
    eprintln!("{message}");
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if let Ok(mut file) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_dir(app).join("shell.log"))
    {
        let _ = writeln!(file, "[{secs}] {message}");
    }
}

// MARK: - Dashboard HTTP

/// Status code of a request, treating HTTP error statuses as answers.
fn request(method: &str, path: &str) -> Result<u16, String> {
    match ureq::request(method, &dashboard_url(path))
        .timeout(Duration::from_millis(1500))
        .call()
    {
        Ok(res) => Ok(res.status()),
        Err(ureq::Error::Status(code, _)) => Ok(code),
        Err(err) => Err(err.to_string()),
    }
}

struct PostError {
    message: String,
    status: Option<u16>,
    retry_after: u64,
    payload: Value,
}

fn post_json(path: &str, payload: &Value) -> Result<Value, PostError> {
    let result = ureq::post(&dashboard_url(path))
        .timeout(Duration::from_secs(310))
        .send_json(payload.clone());
    match result {
        Ok(res) => {
            let text = res.into_string().unwrap_or_default();
            if text.is_empty() {
                Ok(json!({}))
            } else {
                serde_json::from_str(&text).map_err(|e| PostError {
                    message: format!("invalid JSON from dashboard: {e}"),
                    status: None,
                    retry_after: 5,
                    payload: Value::Null,
                })
            }
        }
        Err(ureq::Error::Status(code, res)) => {
            let retry_after = res
                .header("retry-after")
                .and_then(|v| v.parse().ok())
                .unwrap_or(5);
            let text = res.into_string().unwrap_or_default();
            Err(PostError {
                message: format!("dashboard returned {code}: {text}"),
                status: Some(code),
                retry_after,
                payload: serde_json::from_str(&text).unwrap_or(json!({})),
            })
        }
        Err(err) => Err(PostError {
            message: err.to_string(),
            status: None,
            retry_after: 5,
            payload: Value::Null,
        }),
    }
}

fn start_gateway() -> Result<Value, String> {
    let device = std::env::var("AVALON_GATEWAY_DEVICE").unwrap_or_default();
    post_json(
        "/api/api-server/start",
        &json!({
            "model_id": "",
            "port": gateway_port(),
            "mode": "both",
            "device": device,
            "gguf_backend": device,
            "openvino_device": std::env::var("AVALON_OPENVINO_DEVICE").unwrap_or_else(|_| "NPU".into()),
            "gpu_index": "",
        }),
    )
    .map_err(|e| e.message)
}

// MARK: - Backend lifecycle

fn wait_for_backend(timeout: Duration) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok(status) = request("GET", "/api/api-server/status") {
            if status < 500 {
                return Ok(());
            }
        }
        thread::sleep(Duration::from_millis(250));
    }
    Err(format!(
        "Avalon dashboard did not start on {DASHBOARD_HOST}:{}",
        dashboard_port()
    ))
}

fn spawn_backend(app: &AppHandle) -> Result<(), String> {
    // Only dev builds may touch the source checkout. In a packaged app,
    // resolving it (a path on another volume here) makes macOS hold the
    // open() for a removable-volume consent prompt — the backend never
    // started when launched from Finder.
    let (mut cmd, data_dir, cwd) = match sidecar_path(app, "avalon-backend") {
        Some(path) => (
            Command::new(path),
            user_data_dir(app).join("data"),
            app.path().resource_dir().map_err(|e| e.to_string())?,
        ),
        None if is_packaged() => return Err("avalon-backend sidecar is missing from the app bundle".into()),
        None => {
            let root = repo_root();
            let python = if cfg!(windows) {
                root.join("backend").join(".venv").join("Scripts").join("python.exe")
            } else {
                root.join("backend").join(".venv").join("bin").join("python")
            };
            let mut cmd = Command::new(python);
            cmd.arg(root.join("backend").join("main.py"));
            (cmd, root.join("data"), root)
        }
    };
    cmd.current_dir(cwd)
        // The dashboard remains local-only through main.py middleware; only
        // the one-time pairing accept route is reachable from the LAN.
        .env("AVALON_HOST", "0.0.0.0")
        .env("AVALON_PORT", dashboard_port().to_string())
        .env("AVALON_GATEWAY_PORT", gateway_port().to_string())
        .env(
            "AVALON_GATEWAY_HOST",
            std::env::var("AVALON_GATEWAY_HOST").unwrap_or_else(|_| "0.0.0.0".into()),
        )
        // Lower than the library default (3072 MB): this Mac's 18 GB unified
        // memory is shared with other running apps, so a smaller safety
        // reserve leaves small/medium local models admittable in practice.
        .env(
            "AVALON_MEMORY_RESERVE_MB",
            std::env::var("AVALON_MEMORY_RESERVE_MB").unwrap_or_else(|_| "1024".into()),
        )
        .env("AVALON_API_KEY", api_key(app))
        .env("AVALON_DATA_DIR", data_dir)
        .stdin(Stdio::null());
    match fs::File::create(log_dir(app).join("backend.log")) {
        Ok(file) => {
            let err = file.try_clone().map(Stdio::from).unwrap_or_else(|_| Stdio::null());
            cmd.stdout(Stdio::from(file)).stderr(err);
        }
        Err(_) => {
            cmd.stdout(Stdio::null()).stderr(Stdio::null());
        }
    }
    if let Some(gateway) = sidecar_path(app, "avalon-gateway") {
        cmd.env("AVALON_GATEWAY_EXECUTABLE", gateway);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    log(app, format!("spawning backend: {}", cmd.get_program().to_string_lossy()));
    let child = cmd.spawn().map_err(|e| format!("could not start backend: {e}"))?;
    let state = app.state::<Backend>();
    *state.child.lock().unwrap() = Some(child);
    state.owns.store(true, Ordering::SeqCst);

    let handle = app.clone();
    thread::spawn(move || watch_backend(handle));
    Ok(())
}

/// Polls the owned backend and tells the renderer if it exits unexpectedly.
fn watch_backend(app: AppHandle) {
    loop {
        thread::sleep(Duration::from_millis(500));
        let state = app.state::<Backend>();
        let mut guard = state.child.lock().unwrap();
        let Some(child) = guard.as_mut() else { return };
        let Ok(Some(status)) = child.try_wait() else { continue };
        *guard = None;
        drop(guard);
        log(&app, format!("backend exited: {status}"));
        if !state.stopping.load(Ordering::SeqCst) {
            #[cfg(unix)]
            let signal = {
                use std::os::unix::process::ExitStatusExt;
                status.signal().map(signal_name)
            };
            #[cfg(not(unix))]
            let signal: Option<String> = None;
            let _ = app.emit(
                "avalon:backend-exit",
                json!({ "code": status.code(), "signal": signal }),
            );
        }
        return;
    }
}

#[cfg(unix)]
fn signal_name(signal: i32) -> String {
    match signal {
        libc::SIGTERM => "SIGTERM".into(),
        libc::SIGKILL => "SIGKILL".into(),
        libc::SIGINT => "SIGINT".into(),
        libc::SIGSEGV => "SIGSEGV".into(),
        libc::SIGBUS => "SIGBUS".into(),
        libc::SIGABRT => "SIGABRT".into(),
        other => format!("SIG{other}"),
    }
}

fn ensure_backend(app: &AppHandle) -> Result<(), String> {
    if wait_for_backend(Duration::from_millis(1500)).is_ok() {
        log(app, "attached to an already-running dashboard backend");
        app.state::<Backend>().owns.store(false, Ordering::SeqCst);
        return Ok(());
    }
    spawn_backend(app)?;
    wait_for_backend(Duration::from_secs(90))
}

/// Stops the gateway and the backend we spawned. Safe to call repeatedly.
fn stop_backend(app: &AppHandle) {
    let state = app.state::<Backend>();
    if state.stopping.swap(true, Ordering::SeqCst) {
        return;
    }
    if !state.owns.swap(false, Ordering::SeqCst) {
        return;
    }
    // The dashboard may already have exited.
    let _ = request("POST", "/api/api-server/stop");
    let Some(mut child) = state.child.lock().unwrap().take() else { return };
    terminate(&mut child);
}

/// SIGTERM first: the PyInstaller one-file bootloader forwards it to the real
/// Python child, whereas SIGKILL would orphan that child (the long-lived
/// `avalon-backend` that survived quitting the old app).
fn terminate(child: &mut Child) {
    #[cfg(unix)]
    {
        unsafe {
            libc::kill(child.id() as libc::pid_t, libc::SIGTERM);
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = child.try_wait() {
                return;
            }
            thread::sleep(Duration::from_millis(100));
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

// MARK: - Window & tray

fn set_dock_visible(app: &AppHandle, visible: bool) {
    #[cfg(target_os = "macos")]
    {
        let policy = if visible {
            tauri::ActivationPolicy::Regular
        } else {
            tauri::ActivationPolicy::Accessory
        };
        let _ = app.set_activation_policy(policy);
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (app, visible);
}

fn show_main_window(app: &AppHandle) {
    set_dock_visible(app, true);
    if let Some(window) = app.get_webview_window(MAIN_WINDOW) {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
        return;
    }
    let built = WebviewWindowBuilder::new(app, MAIN_WINDOW, WebviewUrl::App("index.html".into()))
        .title("Avalon")
        .inner_size(1440.0, 960.0)
        .min_inner_size(960.0, 640.0)
        .build();
    match built {
        Ok(window) => {
            let handle = app.clone();
            let closing = window.clone();
            window.on_window_event(move |event| match event {
                // Destroy outright (not just close) so the WebContent
                // process and its JS heap are released immediately.
                WindowEvent::CloseRequested { api, .. } => {
                    api.prevent_close();
                    let _ = closing.destroy();
                }
                WindowEvent::Destroyed => set_dock_visible(&handle, false),
                _ => {}
            });
            let _ = window.set_focus();
        }
        Err(err) => eprintln!("could not create Avalon window: {err}"),
    }
}

fn create_tray(app: &AppHandle) -> tauri::Result<()> {
    let show = MenuItem::with_id(app, "show", "Show Avalon", true, None::<&str>)?;
    let start = MenuItem::with_id(app, "start-gateway", "Start LLM Gateway", true, None::<&str>)?;
    let stop = MenuItem::with_id(app, "stop-gateway", "Stop LLM Gateway", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Avalon", true, None::<&str>)?;
    let menu = Menu::with_items(
        app,
        &[
            &show,
            &PredefinedMenuItem::separator(app)?,
            &start,
            &stop,
            &PredefinedMenuItem::separator(app)?,
            &quit,
        ],
    )?;
    let icon = Image::from_bytes(include_bytes!("../icons/tray.png"))?;
    TrayIconBuilder::with_id("avalon-tray")
        .icon(icon)
        .icon_as_template(cfg!(target_os = "macos"))
        .tooltip("Avalon LLM Dashboard")
        .menu(&menu)
        // macOS convention: the menu opens on click. Elsewhere a left
        // click shows the window and the menu is on right click.
        .show_menu_on_left_click(cfg!(target_os = "macos"))
        .on_menu_event(|app, event| match event.id().as_ref() {
            "show" => show_main_window(app),
            "start-gateway" => {
                thread::spawn(|| {
                    if let Err(err) = start_gateway() {
                        eprintln!("Could not start gateway: {err}");
                    }
                });
            }
            "stop-gateway" => {
                thread::spawn(|| {
                    let _ = request("POST", "/api/api-server/stop");
                });
            }
            "quit" => {
                let app = app.clone();
                thread::spawn(move || {
                    stop_backend(&app);
                    app.exit(0);
                });
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if cfg!(target_os = "macos") {
                return;
            }
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                show_main_window(tray.app_handle());
            }
        })
        .build(app)?;
    Ok(())
}

// MARK: - Commands (window.avalon bridge, see src/desktop.ts)

#[tauri::command]
fn runtime_config(app: AppHandle) -> Value {
    let host = gethostname::gethostname().to_string_lossy().into_owned();
    json!({
        "dashboardUrl": format!("http://{DASHBOARD_HOST}:{}", dashboard_port()),
        "gatewayUrl": format!("http://127.0.0.1:{}", gateway_port()),
        "publicGatewayUrl": format!("http://{host}:{}", gateway_port()),
        "apiKey": api_key(&app),
        "packaged": is_packaged(),
    })
}

/// Retries while the gateway reports resource pressure (503), up to 15 min.
#[tauri::command]
async fn quick_test(payload: Value) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let deadline = Instant::now() + Duration::from_secs(15 * 60);
        loop {
            match post_json("/api/api-server/quick-test", &payload) {
                Ok(value) => return Ok(value),
                Err(err) => {
                    let resource_pressure = err.status == Some(503)
                        && err.payload.pointer("/error/type").and_then(Value::as_str)
                            == Some("resource_pressure");
                    if !resource_pressure || Instant::now() >= deadline {
                        return Err(err.message);
                    }
                    let retry_after = err
                        .payload
                        .pointer("/error/resource/retry_after")
                        .and_then(Value::as_u64)
                        .unwrap_or(err.retry_after);
                    thread::sleep(Duration::from_secs(retry_after.max(1)));
                }
            }
        }
    })
    .await
    .map_err(|e| e.to_string())?
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            show_main_window(app);
        }))
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .plugin(tauri_plugin_opener::init())
        .manage(Backend::default())
        .invoke_handler(tauri::generate_handler![runtime_config, quick_test])
        .setup(|app| {
            let handle = app.handle().clone();
            #[cfg(target_os = "macos")]
            if is_packaged() {
                use tauri_plugin_autostart::ManagerExt;
                let _ = handle.autolaunch().enable();
            }
            create_tray(&handle)?;
            thread::spawn(move || {
                log(&handle, format!("starting (packaged={})", is_packaged()));
                match ensure_backend(&handle) {
                    Ok(()) => {
                        log(&handle, "dashboard backend ready");
                        if std::env::var("AVALON_AUTOSTART_GATEWAY").as_deref() != Ok("false") {
                            if let Err(err) = start_gateway() {
                                log(&handle, format!("Avalon gateway was not auto-started: {err}"));
                            }
                        }
                    }
                    Err(err) => log(&handle, err),
                }
                show_main_window(&handle);
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Avalon");

    app.run(|app, event| match event {
        // Closing the last window must not quit: the tray, backend and
        // gateway keep serving. A window-close exit request arrives after
        // the window is gone; Cmd+Q arrives while it still exists.
        RunEvent::ExitRequested { api, code: None, .. } => {
            if app.webview_windows().is_empty() {
                api.prevent_exit();
            }
        }
        // Dock click / Finder "open" on the running app: macOS sends a
        // reopen event instead of starting a second instance.
        #[cfg(target_os = "macos")]
        RunEvent::Reopen { .. } => show_main_window(app),
        RunEvent::Exit => stop_backend(app),
        _ => {}
    });
}
