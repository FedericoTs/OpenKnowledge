# The website

`index.html` is the whole site. One file, no build step, no dependencies, and it
fetches nothing from any other host — no fonts, no scripts, no analytics. A page
whose argument is that your documents stay put should not quietly call four
other servers to render itself, and a test asserts it doesn't.

## Serving it

**From your own container**, alongside the answer engine:

```bash
OK_WEBSITE_ENABLED=true docker compose up
# page at  http://localhost:8080/site
# form posts to /api/contact on the same origin
```

That is the configuration the page is written for. Submissions go into
`data/contacts.db` — your file, on your machine, read with:

```bash
openknowledge contacts            # newest first
openknowledge contacts --json     # to pipe somewhere
```

**Statically**, on GitHub Pages or any file host:

```bash
# in repository settings, publish from a branch, folder /web/site
```

The page works fully, with one difference: there is no endpoint behind the form,
so submitting it says so and points at the issue tracker rather than failing
silently or pretending to have sent something.

**Somewhere else entirely** — add `data-endpoint` to the form:

```html
<form class="contact" id="contact-form" data-endpoint="https://your-host/api/contact">
```

## Why the form is off by default

`OK_WEBSITE_ENABLED` defaults to false. A running answer engine has no business
accepting public writes unless somebody asked it to, and most deployments serve
the chat widget internally and never need this at all.

When it is on: submissions are capped per hour, a hidden honeypot field drops
bots with a cheerful 201 rather than telling them what failed, and contacts live
in their own SQLite file rather than in the answer store — one holds questions
employees asked, the other holds people who want an email, and those have
different retention rules and different readers.

## Changing the copy

Every number on the page is produced by something in this repository, and the
page says so. If you change a figure, change the thing that produces it:

| Claim | Produced by |
|---|---|
| Cost per question by retrieval width | `tools/measure_prompts.py` |
| Open-weight tier at $0.00032 | `pricing.yaml`, verified rates |
| The live-run table | `evals/measured/first-live-run.json` |
| The audit output | `openknowledge audit evals/corpus/aveline` |

The "What it does not do yet" section is not marketing softening — it is the
reason the rest of the page is worth believing. A test asserts it still names
the connectors, OCR and the unmeasured tiers, so it cannot be quietly dropped in
an edit.
