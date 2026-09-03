# Releasing

A release is one commit that becomes three things, in this order: a Windows
installer proved on a clean machine, a GitHub Release carrying it, and a wheel on
PyPI. All three come from `.github/workflows/release.yml`; nothing is built by hand.

## Cutting one

1. Bump `version` in `pyproject.toml`, run `uv lock`, write
   `docs/release-notes/vX.Y.Z.md` if this version has anything to say for
   itself, commit, push to `main`, and wait for CI to go green on that commit.
2. Dispatch **Release** on `main` with `tag: vX.Y.Z` (Actions → Release → Run
   workflow), or push the tag. The tag must equal the version in `pyproject.toml`;
   the first job checks and refuses otherwise.
3. Read the `windows-upgrade` job log for the line that proves the installer:
   `the app handed off its own upgrade while running, reopened, and served as
   X.Y.Z after Ns`. That line, the tag on the commit, and the asset on the release
   page are what "released" means here.

## What the release page says

Every release carries the same paragraphs - how to install, what the first launch
does, what the pipeline proved before publishing, whether the installer is signed -
built by `tools/release_notes.py` from the signing state the build recorded.

A version that has something of its own to say puts it in
`docs/release-notes/vX.Y.Z.md`, named for its tag. The `publish` job places that
file above a rule, above the standing paragraphs, so the first thing a reader sees
is what changed. Preview exactly what will be published:

```
python3 tools/release_notes.py --tag v0.13.0 \
    --setup OpenKnowledge-Setup-0.13.0.exe --sha 0000 \
    --signing "Not code-signed."
```

With no such file the body is byte for byte what this project has always
published - a promise `tests/test_release_notes.py` holds against the text
v0.12.5 actually shipped. Two ways a file goes quiet, both caught by that test
rather than by a release page that already went out: named for no tag
(`0.13.0.md`) it is never read, and holding only whitespace it is treated as
absent so it cannot leave a bare rule on the page.
[docs/release-notes/README.md](release-notes/README.md) has the convention.

The `pypi` job runs last and only after the GitHub release exists. It builds the
sdist and wheel with `uv build`, checks them with `twine`, installs the wheel into
a fresh environment, runs it from another directory (`--version` must be the tag,
the widget must be found inside the wheel, the audit must run), and then uploads.

## Code signing

The installer job signs the executables and the installer when the six
repository variables in [WINDOWS.md](WINDOWS.md#turning-it-on-once) exist,
never on a pull request, and fails the build if a configured signature does
not verify. The state is recorded in `SIGNING.txt` inside the installer
artifact and repeated in the release notes. Nothing else changes: an
unconfigured repository builds an unsigned installer and says so.

## Setting up PyPI, once

Publishing uses [trusted publishing](https://docs.pypi.org/trusted-publishers/):
PyPI trusts the workflow's identity, so there is no token to store or rotate. It
has to be registered on pypi.org by an account that will own the project:

1. Sign in to pypi.org, open **Your account → Publishing**, and add a **pending
   publisher** (the project does not exist yet, so it cannot be added from a
   project page).
2. Fill in exactly:

   | field | value |
   |---|---|
   | PyPI project name | `openknowledge` |
   | Owner | `FedericoTs` |
   | Repository name | `OpenKnowledge` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

3. Run the next release. The first successful `pypi` job creates the project and
   the pending publisher becomes its publisher. From then on
   `uvx openknowledge audit ./policies` works with nothing installed, and the
   README's git-based `uvx --from` line can be shortened to that.

Until the publisher is registered, the `pypi` job fails with an OIDC error from
PyPI and the release above it stands: the installer and the GitHub release do not
depend on it.

## What can go wrong

- **Tag and version disagree.** The `check` job fails before anything is built.
  Bump the version or fix the tag; never publish a tag whose installer reports a
  different version.
- **The wheel does not carry the web assets.** The smoke step fails on
  `find_asset`. The assets travel through `[tool.hatch.build.targets.wheel.force-include]`
  in `pyproject.toml`; a test in `tests/test_foundations.py` guards the same thing
  on every CI run.
- **PyPI rejects the version.** A version can be uploaded once, ever. Bump and
  release again; do not delete and reuse.
