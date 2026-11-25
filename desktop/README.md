# Babble desktop app

A Mac app (Tauri) around a React + TypeScript interface (Vite) that talks to the voice engine's local API
(`narrator/api.py`). The same interface also runs in a browser at **http://127.0.0.1:7895/app/** once built.

```bash
npm install              # once
npm run build            # the interface -> dist/ (the engine serves it at /app/; no restart needed)
npm run dev              # the interface at http://localhost:1420 with live reload, using the engine on port 7895
npx tauri build --bundles app   # the app -> src-tauri/target/release/bundle/macos/Babble.app
npx tauri dev            # the app window with live reload
```

## The app (`src-tauri/`)
- Opening it starts the engine (`.venv/bin/python run.py --port 7895` in the project folder, output to
  `~/Library/Logs/Babble.log`) unless one is already running, in which case it uses that one.
  Quitting stops the engine only if the app started it.
- This build finds the project folder where it was built (or `NARRATOR_ROOT`), so it runs on this Mac. A version
  for other Macs has to carry its own Python and packages, and be signed and notarized.
- Saving a file copies it to ~/Downloads and shows it in Finder; links to other pages open in the browser.
- The window has no title bar of its own (traffic lights over the sidebar); the top strip drags it.

## The interface (`src/`)
- `api.ts`: the API client. Long work (speech, books, sounds) runs as tasks; `useTask` follows one live
  (server-sent events, polling if the stream drops) and can stop it.
- `native.ts`: what differs in the app (saving files, opening pages).
- `components/`: shared pieces (sliders, progress, voice picker, toasts).
- `pages/`: one file per section. Done: Text to Speech, Audiobook, Sounds. Still in the classic view: Voices,
  Library, Train.
- `styles.css`: design tokens and components; light and dark follow the Mac.
- The app's name is one constant (`APP_NAME` in `src/App.tsx`) plus `productName` in `src-tauri/tauri.conf.json`.
