<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="desktop/src-tauri/icons/icon-dark.png">
    <img src="desktop/src-tauri/icons/icon.png" alt="Babble" width="128">
  </picture>
</p>

<h1 align="center">Babble</h1>

<p align="center">Audiobooks with a cast of voices, and voice training, on Apple Silicon Macs. Free and open source.</p>

## Download

**[Babble Beta 1](https://github.com/Laking22/babble/releases/latest)**: pick the file for your computer.

- **Mac** (Apple Silicon, macOS 14 or later): `Babble_…_mac_aarch64.zip`. Unzip it and move Babble to Applications.
  The first time, macOS says it can't check the app: open **System Settings → Privacy & Security** and click
  **Open Anyway**.
- **Windows** (`…_x64-setup.exe`) and **Linux** (`.AppImage` or `.deb`): in testing.

The first start downloads Python, Babble's packages and FFmpeg (a few minutes). After that, Babble tells you when
there's a new version (Settings → General → Updates), and can update itself.

## Listen

**The original:** Mil Nicholson reading *Great Expectations*, chapter 1
(5 min, LibriVox, public domain). This is the original voice the fine-tuned voice was trained on.

Models pick up tone and habits of what they were trained on. A voice trained on a slow, calm, romance narrator will narrate an action story poorly. Models may also learn to change tone and pitch for different speaking characters, however voices may not be constant. 

https://github.com/user-attachments/assets/d87b1308-4706-4738-b99d-8a714c838fcf

**In Babble:** the same voice as a fine-tuned Qwen clone. Each story's sounds were directed by a different model. Models do not determine the quality of sounds played, only what and when sounds are played.

<details>
<summary><b>Action</b></summary>

**Opus 5.5**

https://github.com/user-attachments/assets/29ac64c8-2ad5-46ea-a8f7-ef75f62f84c3

**Sonnet 5.5**

https://github.com/user-attachments/assets/b4ba8baa-c57e-46ad-ba38-8f64a5f74780

**Haiku 5.5**

https://github.com/user-attachments/assets/23e03d30-20bd-4755-8fff-43e965f9261c

</details>

<details>
<summary><b>Fantasy</b></summary>

**Opus 5.5**

https://github.com/user-attachments/assets/d9dd6ed9-90b2-491d-8783-cd02a41f9e3c

**Sonnet 5.5**

https://github.com/user-attachments/assets/ed04cfbf-3163-4e34-a4cc-1f7b9df6dc71

**Haiku 5.5**

https://github.com/user-attachments/assets/21d904c0-0e4f-4c9e-970d-7871eafa1896

</details>

<details>
<summary><b>Romance</b></summary>

**Opus 5.5**

https://github.com/user-attachments/assets/187e0854-9ded-45aa-894b-e92895d9c7f5

**Sonnet 5.5**

https://github.com/user-attachments/assets/2ba2dd57-ce3a-4e46-8420-89c9d3632910

</details>

<details>
<summary><b>Mystery</b></summary>

**Opus 5.5**

https://github.com/user-attachments/assets/610349e5-858f-41f4-bc41-025238399235

**Sonnet 5.5**

https://github.com/user-attachments/assets/9097477f-f157-4523-9194-5f49e6496c11

**Haiku 5.5**

https://github.com/user-attachments/assets/ad6eab63-91d5-4740-8408-dce5013198c0

</details>

## Run (development)

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
NARRATOR_HOME=./data .venv/bin/python run.py          # http://127.0.0.1:7895
cd desktop && npm install && npx tauri dev              # the desktop app
```

Data (voices, models, outputs, training) lives in `~/Library/Application Support/Babble` on a Mac (Babble moves it
there from `Narrator Studio`, its old name, the first time it starts), or wherever `NARRATOR_HOME` points. Needs `ffmpeg` on the PATH.

## Layout

```
narrator/
  tts/            one class per TTS engine (Engine interface in base.py), registry in __init__.py
    qwen.py         Qwen3-TTS: built-in speakers, saved voices, fine-tuned models, design, exact cloning
    orpheus.py      Orpheus 3B: stock voices + fine-tuned models, Mac or API server
    kokoro.py       Kokoro 82M: fast built-in voices for low-memory Macs
  engine.py       render(): groups a cast's lines by engine and model, joins them in story order, mixes sound
  sfx.py          sound library, {sfx:}/{amb:} tags, suggestions, mixer (ambience ducked under speech)
  library.py      installed models, voice packs (.narrator-pack: export/install/catalog), engine downloads
  jobs.py         background jobs with logs on disk
  runtime.py      the single MLX thread + one model slot shared by every engine
  voices.py       voice names, labels and lookups across engines (voices_lib.py: saved voices)
  dataset/        audiobook + book -> clean clips + datasets (pipeline.py)
  training/       recipes (one per model) + back ends (where training runs)
    recipes/orpheus.py        SNAC encode -> LoRA (GPU or MLX) -> 4-bit MLX install     [ready]
    recipes/qwen.py           MLX encode -> LoRA on a T4 -> 6-bit MLX install           [ready, first full run pending]
    backends/kaggle.py        the user's own Kaggle account                             [ready]
    backends/ssh.py           any NVIDIA machine over SSH (own PC, RunPod/Vast pods)    [ready]
    backends/modal_cloud.py   Modal                                                     [beta, untested]
    backends/local_mac.py     this Mac with MLX (Orpheus)                                [beta]
desktop/          the Mac app (Tauri + React) around the engine
babble_mcp.py     MCP server so AI agents can use Babble (Settings → General)
tests/engine_check.py   render one engine (or a mixed cast) and check speed, Whisper accuracy, pauses
```

## Sound tags

`{sfx: door creak}`, `{sfx: thunder, loud}`, `{amb: rain}`, `{amb: tavern, quiet}`, `{amb: off}`, anywhere in the
text. Curly braces so they never clash with cast tags (`[Mara: whispering]`).

## Adding a TTS engine

Write a class in `narrator/tts/` that subclasses `Engine`: its voice prefix, `voices()`, `model_key()`,
and `speak()` (yield `(index, audio)` at 24 kHz). Load models through `runtime.slot` so only one is in
memory. Add it to `ENGINE_CLASSES`, then run `tests/engine_check.py <prefix>:<voice>`.

## Adding a training recipe or back end

Recipe (`training/base.py`): `prepare()` turns Train-tab datasets into a `Package` (data folder + script +
pip packages), `install()` turns results into an installed model. Back end: `check`, `estimate`, `submit`,
`status`, `fetch`. Command line: `python -m narrator.training --help`.

## Voices and rights

Built-in voice packs are free-to-use voices only. Users train voices at their own responsibility.

## License

Babble is free software under the [GNU General Public License v3.0 or later](LICENSE): you can use, change and
share it, and anything you share that's built on it must be open source under the same license.

The Stable Audio 3 code in `narrator/vendor/stable_audio_3` is MIT (see its LICENSE). The models, voices and sounds
Babble downloads have their own licenses, which Babble shows where they're used.
