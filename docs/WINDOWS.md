# OpenKnowledge on Windows

One installer, no terminal, no admin rights, no cloud. This page covers what
it installs, what the first launch does, how to build it, and — honestly —
what an unsigned installer means until code signing is set up.

If you are comfortable with a terminal, [LOCAL-SETUP.md](LOCAL-SETUP.md) is
the from-source path and works on Windows too.

---

## What the installer does

- Installs **per-user** under `%LOCALAPPDATA%\Programs\OpenKnowledge`. No
  UAC prompt, no services, nothing outside your profile. That is the least
  privilege the app needs, and the honest shape for an installer that is
  not yet code-signed.
- Ships two executables from one bundle:
  - **OpenKnowledgeApp.exe** — the Start-menu entry (shown as “OpenKnowledge”). Starts everything, opens
    the chatbot in your browser, lives in the system tray (open chat,
    manage knowledge, documents folder, quit).
  - **openknowledge.exe** — the same CLI a `pip install` provides, for
    `openknowledge ask`, audits and scripting. An installer option adds it
    to `PATH`.
- Ships llama.cpp's `llama-server` (MIT), the **win-vulkan** build: GPU
  inference wherever a Vulkan driver exists — NVIDIA, AMD and Intel alike —
  and runtime-dispatched CPU paths everywhere else. One artifact, no
  per-vendor downloads, no CUDA runtime.

## What the first launch does

1. The app serves immediately and opens your browser at
   `http://127.0.0.1:8080` — before any model exists. The chat page hands
   over to a setup page that **asks before downloading anything**.
2. One click downloads the two models the project's numbers were measured
   on, with live progress in the page:

   | file | size | license | role |
   |---|---|---|---|
   | Qwen3-4B-Instruct-2507 Q4_K_M | 2.5 GB | Apache-2.0 | chat |
   | nomic-embed-text-v1.5 Q4_K_M | 84 MB | Apache-2.0 | retrieval |

   Each file is verified against a SHA-256 pinned in
   `src/openknowledge/desktop/manifest.py` — the exact bytes behind the
   published accuracy figures; a mismatch is refused, not warned about.
   A dropped connection **retries and resumes by itself**; a connection
   that keeps dying ends in a Resume button in the page, never a native
   dialog and never a relaunch. (The first field test was a laptop whose
   connection died every ~190 MB; this design is its direct product.)
3. The local inference servers start — the page says so while the 2.5 GB
   model loads, which can take minutes on a laptop disk — and the chat
   activates the moment they answer.
4. Your state lives under `%LOCALAPPDATA%\OpenKnowledge`: documents,
   database, settings, models, logs. `openknowledge paths` prints every
   location and why it was chosen.

Settings you change later (in the browser at `/manage`) are respected: the
launcher only starts the servers its own defaults point at. Re-point the
model endpoint at your own Ollama and the launcher stops managing it —
your choice is never overwritten.

Uninstalling removes the program, **not** your knowledge base. The state
folder survives on purpose; delete `%LOCALAPPDATA%\OpenKnowledge` yourself
if you want it gone.

## Updating

From v0.2.2 the app updates itself in one click. It asks github.com whether
a newer release exists - once when the app starts, and at most once a day
after that (an outbound call, documented here; `OK_UPDATE_CHECK=false`
turns it off entirely for air-gapped or IT-managed installs). When one
does, a quiet "Update to vX.Y.Z" button appears in the sidebar. Clicking it
downloads the new installer, verifies it against the SHA-256 digest GitHub
records for that exact release asset, closes the app cleanly, installs
silently, and reopens - about a minute, and your documents, settings,
caches and models are untouched (the 2.6 GB of models live outside the
install directory and are never re-downloaded).

The sidebar always states what it knows: the version you are running, and
whether it is current, one behind, or could not be checked. There is a
"Check for updates" link beside it for the impatient. Up to v0.2.4 the
check ran only once a day and said nothing at all in every other case, so
a release published an hour after a check stayed invisible until the next
day - and restarting, the obvious remedy, did not help, because the
throttle stamp outlives the process. Both halves of that are fixed: a
launch always checks, and the panel is never silent.

Deliberately not automatic: unsigned binaries earn an explicit click, and
enterprise IT rightly bans software that changes itself unannounced. With
sign-in on, only an administrator sees the button do anything. And honesty
about the trust model: the digest check defeats a corrupted download or a
tampering mirror, not a compromised publisher account - that is what code
signing is for, one section down.

## Building it

The build runs in CI on every packaging change
(`.github/workflows/package.yml`) and uploads the installer as an
artifact; the same three commands work on any Windows machine:

```powershell
uv venv --python 3.12; uv pip install -e ".[desktop,packaging,anthropic]"
uv run pyinstaller packaging/pyinstaller/openknowledge.spec
powershell -File packaging/windows/fetch-llama.ps1
iscc /DAppVersion=0.2.0 packaging/windows/installer.iss
```

Output: `dist/installer/OpenKnowledge-Setup-0.2.0.exe`. CI additionally
runs the frozen executables — `paths`, `--version`, a real `serve` with a
`/healthz` probe — and installs the installer silently, checking the
installed app runs. The PyInstaller spec also builds and smoke-tests on
Linux, which catches import-graph and data-file mistakes early; only the
Windows run proves the Windows build.

On demand, an end-to-end job runs the whole product from the installer on
a real Windows machine: true first run, models downloaded and verified,
a document uploaded, a question answered by the local model, the same
question answered again byte-identically from cache. The first green run
is recorded in `evals/measured/windows-e2e-first-run.json` — serving 24
seconds after launch and a grounded $0 answer in 28.4 seconds on 4 CPU
cores. The 24 seconds rode the CI machine's bandwidth; on a home
connection the 2.6 GB download dominates first-run time, and it resumes
if interrupted.

`fetch-llama.ps1` resolves llama.cpp's latest release at build time and
fails loudly if the `*bin-win-vulkan-x64.zip` asset or `llama-server.exe`
inside it disappears; the tag it shipped is recorded in
`{app}\llama\llama-version.txt`.

## Code signing — the honest part

The installer is **not signed yet**. Windows SmartScreen shows "Windows
protected your PC" with a **More info → Run anyway** path, and some antivirus
products treat young, unsigned PyInstaller executables as suspicious on
reputation alone. Every release's notes say which it is, because the build
records it: `SIGNING.txt` travels in the installer artifact, the notes repeat
it, and a build that was *configured* to sign but produced anything Windows
does not call a Valid signature fails before it reaches a download page.

The pipeline is ready; what is missing is an identity. It is wired for
**Azure Artifact Signing** (the service formerly called Trusted Signing):
certificates are issued against a verified identity, short-lived and rotated
by the service, SmartScreen reputation attaches to the identity rather than to
each certificate, and the job authenticates with its own OpenID Connect token,
so no secret is stored in this repository. About $10 a month. The alternative
is a classic OV certificate on a hardware token, which cannot sign from a
hosted runner at all since 2023 without a cloud HSM, and builds reputation
per certificate, slowly.

### Turning it on, once

1. In Azure, create an **Artifact Signing account** and complete **identity
   validation** (individual or organisation; the portal states the current
   eligibility rules), then a **certificate profile** of type Public Trust.
2. Create an **app registration** in Entra ID and add a **federated
   credential** for GitHub Actions with subject
   `repo:FedericoTs/OpenKnowledge:environment:signing` - the installer job
   runs in the `signing` environment precisely so this subject is stable
   across branches and tags.
3. On the signing account, assign that app the role
   **Artifact Signing Certificate Profile Signer**.
4. Set six **repository variables** (Settings → Secrets and variables →
   Actions → Variables; none is a secret):

   | variable | value |
   |---|---|
   | `AZURE_TENANT_ID` | the directory (tenant) id |
   | `AZURE_CLIENT_ID` | the app registration's client id |
   | `AZURE_SUBSCRIPTION_ID` | the subscription holding the account |
   | `SIGNING_ENDPOINT` | the account's regional endpoint, e.g. `https://weu.codesigning.azure.net/` |
   | `SIGNING_ACCOUNT` | the Artifact Signing account name |
   | `SIGNING_PROFILE` | the certificate profile name |

5. Run the next release. The build signs `openknowledge.exe` and
   `OpenKnowledgeApp.exe`, builds the installer from them, signs the installer,
   verifies all three, and the notes change from "Not code-signed" to
   "Signed by …". `llama-server.exe` is upstream's binary and stays as upstream
   shipped it. The uninstaller Inno Setup writes at install time is not signed.

While `SIGNING_ACCOUNT` is unset the same pipeline builds an unsigned
installer, as today, and says so. Pull requests never sign, whatever is set.

### Verifying a download

Right-click the installer → Properties → Digital Signatures, or in PowerShell:

```powershell
Get-AuthenticodeSignature .\OpenKnowledge-Setup-x.y.z.exe | Format-List Status, SignerCertificate
Get-FileHash .\OpenKnowledge-Setup-x.y.z.exe -Algorithm SHA256
```

The hash is in the release notes for every version, signed or not. False
positives from antivirus products should be reported at Microsoft's
[submission portal](https://www.microsoft.com/en-us/wdsi/filesubmission);
do **not** tell users to add antivirus exclusions.

## Troubleshooting

| symptom | likely cause | what to do |
|---|---|---|
| "Windows protected your PC" | unsigned installer (see above) | More info → Run anyway, or build from source |
| first launch sits at "Downloading…" | 2.6 GB on your connection | let it finish; closing and relaunching resumes |
| "did not become ready" with a log tail | llama-server could not load the model | the message includes the log; usually RAM (4 GB free needed) |
| browser opens to an error on 8080 | another app owns port 8080 | free the port; the launcher reuses an existing OpenKnowledge if one is serving |
| the tray icon is missing | tray degraded to console mode | the browser page still works; logs are in `%LOCALAPPDATA%\OpenKnowledge\data\logs` |

The chat model and embedding server each write
`%LOCALAPPDATA%\OpenKnowledge\data\logs\llama-*.log`; those two files are
where "it does not answer" stops being a mystery.
