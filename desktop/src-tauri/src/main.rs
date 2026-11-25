// No terminal window on Windows release builds (harmless on the Mac).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    narrator_studio_lib::run()
}
