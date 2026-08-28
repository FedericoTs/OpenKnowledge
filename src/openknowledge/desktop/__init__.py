"""The desktop app: everything between a double-click and a working chatbot.

This package exists for the installed-on-Windows shape of OpenKnowledge,
where there is no terminal, no pip and no Ollama - just an installer, a
Start-menu entry and a tray icon. It downloads the models the project has
actually measured with, supervises the bundled llama-server processes, and
serves the same FastAPI app the CLI serves. Nothing in here answers
questions; the cascade neither knows nor cares who started the server.
"""
