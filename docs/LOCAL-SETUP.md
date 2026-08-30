# Running OpenKnowledge on your own machine

Everything below happens inside one folder — `~/Documents/Projects/OpenKnowledge`.
Nothing is installed system-wide, no `sudo` is needed, and deleting the folder
removes every trace of it.

Times are for a first run on a normal laptop.

On Windows there is also an installer — no terminal, no Python — described
in [WINDOWS.md](WINDOWS.md). This page is the from-source path, and it works
on Windows too. To serve a whole team from one machine, see
[COMPANY-SERVER.md](COMPANY-SERVER.md).

---

## 0. What you need

| | | |
|---|---|---|
| **git** | required | `git --version` |
| **Python 3.11+** | required | `python3 --version` |
| **[Ollama](https://ollama.com/download)** | only to answer questions | `ollama --version` |
| **Java 17+** | optional | better PDF parsing; without it, pdfplumber is used |

The audit tier — the part that reads your documents and finds contradictions —
needs neither Ollama nor Java. It needs no model, no API key and no GPU.

---

## 1. Install (2 minutes)

```sh
curl -fsSL https://raw.githubusercontent.com/FedericoTs/OpenKnowledge/HEAD/install.sh | sh
```

That clones into `~/Documents/Projects/OpenKnowledge`, builds a virtualenv
*inside* that folder, installs the package into it, and then proves the install
by running the audit against the repository's own test corpus.

If you would rather read it before running it — reasonable, for anything piped
into a shell:

```sh
mkdir -p ~/Documents/Projects && cd ~/Documents/Projects
git clone https://github.com/FedericoTs/OpenKnowledge.git
cd OpenKnowledge
less install.sh          # it is short
./install.sh
```

Both routes end in the same place. Install somewhere else with
`OK_DIR=/path/to/wherever sh install.sh`.

### Put it on your PATH

The commands below assume `openknowledge` is callable. Either prefix every
command with `.venv/bin/`, or add it to your shell for this session:

```sh
cd ~/Documents/Projects/OpenKnowledge
export PATH="$PWD/.venv/bin:$PATH"     # .venv/Scripts on Windows
```

That lasts as long as the terminal. To make it permanent — this is the single
most common thing to trip over, because a new shell forgets it:

```sh
# macOS / Linux
echo "export PATH=\"$HOME/Documents/Projects/OpenKnowledge/.venv/bin:\$PATH\"" >> ~/.zshrc

# Windows, in Git Bash
echo "export PATH=\"$HOME/Documents/Projects/OpenKnowledge/.venv/Scripts:\$PATH\"" >> ~/.bashrc
```

Open a new terminal for it to take effect. The installer will not do this for
you: editing your shell profile is the one thing that would leave a trace
outside the install folder, and then `rm -rf` would no longer be the whole
uninstall.

---

## 2. Audit your documents (30 seconds, costs nothing)

Point it at a real folder of policies:

```sh
openknowledge audit ~/Documents/policies
```

It reads PDF, Word, Excel, PowerPoint, Markdown and plain text, and prints the
places where two documents state different figures for the same rule.

It builds no index, writes no files, starts no server, contacts nothing. You can
run it on a folder you would never upload anywhere, because nothing is uploaded.

Two flags worth knowing:

```sh
openknowledge audit ~/policies --min-overlap 0.45   # flag less; raise to cut noise
openknowledge audit ~/policies --exit-zero          # don't fail the shell on findings
```

Without `--exit-zero` it exits non-zero when it finds something, which is what
makes it usable as a CI check on a policy repository.

---

## 3. Add your documents

```sh
cd ~/Documents/Projects/OpenKnowledge
cp -r ~/Documents/policies/* documents/
openknowledge index
```

`index` re-reads the folder, chunks it, detects conflicts and reports anything it
could not read. It calls no model, so it is free and safe to run whenever files
change.

Subfolders are categories: `documents/HR/handbook.pdf` indexes like any other
file, the chat sidebar groups documents by folder, and uploads through the
browser can pick a destination folder or create one.

To keep documents where they already live instead of copying them, set the
folder in `.env`:

```sh
OK_DOCUMENTS_DIR=/Users/you/Documents/policies
```

---

## 4. Add a model (10 minutes, mostly downloading)

Answering questions needs a model. On your own machine it costs nothing per
question and no document leaves the laptop.

Install [Ollama](https://ollama.com/download), then:

```sh
ollama serve                    # in its own terminal, if it isn't already running
openknowledge model list        # what your machine has, and each one's window
openknowledge model use qwen3:8b
```

`model use` downloads the model if it is missing, pins its context window at
8,192 tokens, and records both in `.env`.

Pinning matters more than it sounds. `/api/show` reports what the *weights*
declare — 40,960 for qwen3:8b — but Ollama runs an unpinned model at **4,096**
unless `OLLAMA_CONTEXT_LENGTH` says otherwise, and its API does not report that
setting. This project's own defaults (`k=6` retrieval plus 1,500 answer tokens)
come to roughly 3,800, so an unpinned model fits until one longer document
doesn't — and then the prompt is truncated from the front, where the grounding
rules are. Pinning makes the number true; `--no-pin` opts out, and then nothing
is recorded and no prompt is checked for fit.

### Rough guide to size

| Model | Machine | Notes |
|---|---|---|
| `qwen3:4b` | 8 GB RAM, CPU only | slow but workable; the floor |
| `qwen3:8b` | 16 GB RAM | the default, and the balance point |
| `qwen3:14b` | GPU, or 32 GB | noticeably better on long policies |
| `qwen3:30b` | GPU with 24 GB+ | mixture-of-experts: large, but only part of it runs |

`openknowledge model list` prints the real sizes and windows from your own
runtime. Trust those over this table.

### A bigger context window

Long policies, or a high `OK_RETRIEVAL_K`, need a window bigger than the default:

```sh
openknowledge model use qwen3:30b --context 131072
```

Ollama's OpenAI-compatible endpoint accepts no context-length parameter, so this
cannot be set per request. The command therefore builds a copy of the model
carrying `num_ctx` — `qwen3-30b-ok131072` — and points `OK_LOCAL_MODEL` at it.
The weights are shared, so the copy costs no extra disk.

Two things to know:

* **Past a model's trained window, quality is a thing to measure.** The command
  says so when you ask for more than the model declares. Run `openknowledge eval`
  before trusting a stretched window.
* **The window is recorded, and enforced.** `OK_LOCAL_CONTEXT_TOKENS` goes into
  `.env`, and a prompt that would not fit is refused before it is sent. Runtimes
  disagree about what an over-long prompt means and the API does not say which
  you have: llama-cpp-python, measured, returns `context_length_exceeded` and
  answers nothing — but a runtime that trims the prompt to fit instead takes it
  off the *front*, where the grounding rules are, and returns an answer that
  looks completely normal and is ungrounded. Checking here makes both cases end
  the same way.

Check the two agree at any time:

```sh
openknowledge model status
```

### Not using Ollama?

Anything that serves `/v1/chat/completions` works — vLLM, LM Studio, llama.cpp's
server, or a remote box:

```sh
OK_LOCAL_BASE_URL=http://192.168.1.50:8000/v1
OK_LOCAL_MODEL=Qwen/Qwen3-8B
OK_LOCAL_CONTEXT_TOKENS=32768
```

Those runtimes fix their window at launch, so `--context` cannot change it from
here; the command tells you the flag to relaunch with (`--ctx-size` for
`llama-server`, `--n_ctx` for llama-cpp-python, `--max-model-len` for vLLM) and
records the number so the fit check still works.

---

## 5. Ask something

```sh
openknowledge ask "what is the travel expense approval threshold?"
```

The answer prints with its sources, which tier produced it, and what it cost.
Ask the same question twice: the second is served from cache, is byte-identical,
and costs nothing.

---

## 6. Run the server

```sh
openknowledge serve
```

| | |
|---|---|
| `http://localhost:8080/` | chat widget |
| `http://localhost:8080/manage` | management page |
| `http://localhost:8080/docs` | HTTP API |
| `http://localhost:8080/site` | the public page, when enabled |

### More than one person asking at once

A local model server answers a fixed number of questions at a time, and a
request arriving with no slot free is not queued by it — the connection is
accepted and then cut off part-way through the answer. Measured on a single
slot: four simultaneous questions, one answered, three refused.

`OK_LOCAL_PARALLEL` is the number of slots, and it does two things at once —
requests queue to it, and the model server is started with exactly that many.
Keeping it one number is the point: a queue wider than the slots severs
streams, and a queue narrower wastes a server's memory.

```sh
OK_LOCAL_PARALLEL=4
```

One (the default) is right for a laptop: four full-context KV buffers is
precisely the allocation an integrated GPU refuses. A shared server with
memory to spare should raise it — each slot holds its own full context, so
budget for that before increasing it. Questions beyond the slots wait their
turn rather than failing.

If you point `OK_LOCAL_BASE_URL` at a server you started yourself, set this
to match the `--parallel` (llama.cpp) or equivalent you launched it with.

### Manage it from the browser

`/manage` is where the folder-and-CLI work moves into the browser: add and
remove documents, re-index, approve or reject drafted answers, resolve
contradictions, change settings, read the cost report. The page itself is
just a page — every action on it calls the admin API, which wants the admin
token. Print it where the server runs and paste it once; it stays in that
browser:

```sh
openknowledge token
```

Uploading through the browser — the page's **Add documents…**, or dropping
files straight onto the chat widget — is off by default on a server, because
a deployed answer engine should not accept writes nobody asked for. Turn it
on with `OK_UPLOAD_ENABLED=true`, or flip `upload_enabled` in the settings
grid on the page. When the server runs from your own per-user state (no
checkout, no `OK_STATE_DIR`), uploads are on and a private token has already
been minted for you — that is your machine, and `openknowledge paths` shows
where everything is.

Settings marked **live** apply from the next question. Settings marked
**rebuild** swap the engine in place — the new one is built first, so if the
change is bad (a wrong URL, a missing model) the old engine keeps serving
and nothing is saved. Every applied change is written to the state `.env`,
so it survives a restart.

To serve the marketing page and its contact form as well:

```sh
OK_WEBSITE_ENABLED=true openknowledge serve
```

It is off by default because a running answer engine has no business accepting
public writes unless somebody asked it to. Submissions land in your own SQLite
file and are read back with `openknowledge contacts`.

### Or with Docker

```sh
docker compose up --build
```

Same thing, with Java present so the better PDF backend is used.

---

## 7. Keep it up to date

```sh
cd ~/Documents/Projects/OpenKnowledge
git pull
.venv/bin/pip install -e .
```

Or just re-run the installer — it updates an existing checkout rather than
re-cloning, and it stops rather than discarding local changes:

```sh
sh install.sh
```

Your `.env`, your `documents/` folder and your `data/` folder are all
git-ignored, so an update never touches them.

---

## 8. Check that it still works

```sh
make check          # lint, types, and the full test suite
openknowledge eval  # accuracy and cost together, on the golden set
```

`eval` is the one that matters. It runs a set of questions with known answers and
reports accuracy, refusals, false answers and total cost. Run it after changing a
model, a window, or a threshold — the point of the whole project is that these
numbers come from something you can run rather than something someone claimed.

Free dry run first, which checks the questions against your corpus without
calling a model:

```sh
openknowledge eval --dry-run
```

---

## Where everything lives

```
~/Documents/Projects/OpenKnowledge/
├── .env              your settings          (git-ignored)
├── documents/        your documents         (git-ignored)
├── data/             indexes, cache, ledger (git-ignored)
├── .venv/            the isolated environment
├── evals/            the golden set and test corpus
└── web/site/         the public page
```

Uninstalling is `rm -rf ~/Documents/Projects/OpenKnowledge`. There is nothing
anywhere else.

Run from *outside* a checkout — which is what the coming desktop build does —
state moves to your platform's per-user data directory instead
(`%LOCALAPPDATA%\OpenKnowledge` on Windows, `~/Library/Application
Support/OpenKnowledge` on macOS, `~/.local/share/OpenKnowledge` on Linux), so
databases never scatter across whatever folder was current at launch.
`openknowledge paths` prints exactly where everything lives for the way you are
running it, and why.

---

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `command not found: openknowledge` | not on PATH | `export PATH="$PWD/.venv/bin:$PATH"` |
| `nothing is answering at http://localhost:11434/v1` | Ollama not running | `ollama serve` |
| `No documents found` | empty folder | put files in `documents/`, or set `OK_DOCUMENTS_DIR` |
| Answers are refusals | the grounding gate is working | check `openknowledge conflicts` — two documents may disagree |
| `the prompt needs about N tokens and the window is M` | window too small | `openknowledge model use <model> --context <bigger>` |
| A PDF contributed nothing | scanned images | there is no OCR; `openknowledge index` names every file it skipped |
| It's reading different files than I edited | two installs, two state dirs | `openknowledge paths` shows which `.env`, data and documents this invocation uses |

Everything the system does is designed to be checkable rather than trusted:
`openknowledge costs` reports what it actually spent from the ledger,
`openknowledge eval` reports what it actually got right, and `openknowledge
audit` reads your documents without writing anything at all.
