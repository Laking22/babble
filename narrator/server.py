"""The engine: the local API at /api (narrator/api.py) and the app's interface at /app (desktop/dist), in one
process, so still one voice model in memory at a time. (The old Gradio page, narrator/ui, isn't served anymore.)

Run: .venv/bin/python run.py [--port 7895]
"""


def create_app(port=7895):
    import os
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse
    from . import settings, vault
    vault.start()        # the keys the app keeps in the Keychain (when the app started this engine)
    settings.migrate()   # keys still in settings.json move there
    from . import api
    debug_docs = os.environ.get("BABBLE_API_DOCS") == "1"
    app = FastAPI(title="Babble", docs_url="/api/docs" if debug_docs else None,
                  openapi_url="/api/openapi.json" if debug_docs else None, redoc_url=None)
    api.install(app)

    @app.get("/", include_in_schema=False)
    def home():
        return RedirectResponse("/app/")
    return app


def main(port=7895):
    import uvicorn
    app = create_app(port)
    print(f"* Running on local URL:  http://127.0.0.1:{port}  (API: /api)", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
