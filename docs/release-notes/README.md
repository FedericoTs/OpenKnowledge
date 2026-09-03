# Per-version release notes

One file per release, named for its tag: `v0.13.0.md` for `v0.13.0`. The
`publish` job in [`release.yml`](../../.github/workflows/release.yml) puts the
file's contents at the top of the GitHub Release body, above a rule, above the
paragraphs every release carries - how to install, what the first launch does,
what the pipeline proved before publishing, and whether the installer is
signed.

A release without a file here publishes exactly those standing paragraphs, as
every release did before this directory existed.

## Writing one

Write it in the same commit that bumps the version, before the release is cut,
and say what a person downloading the exe needs to know: what changed, what a
measured number moved to, what is still broken. Markdown, no front matter, no
heading required - GitHub renders it directly under the release title.

Preview exactly what will be published:

```
python3 tools/release_notes.py --tag v0.13.0 \
    --setup OpenKnowledge-Setup-0.13.0.exe --sha 0000 \
    --signing "Not code-signed."
```

## Two ways it goes quiet

A file named for no tag (`0.13.0.md`, `next.md`) is never read, and the
release publishes without it. A file holding only whitespace is treated as
absent so it cannot leave a bare rule on the page. Both are checked in
`tests/test_release_notes.py`, because the only other way to find out is a
release page that already went out.

## Why this exists

v0.12.5 was cut to retire an accuracy claim that an evaluation on third-party
documents disproved. Its release page never mentioned it: the notes were a
fixed template, so the published body differed from v0.12.4's only in the
version string and the SHA-256. The retraction reached the README and the git
history, which it would have anyway, and never reached the page a stranger
downloads from.
