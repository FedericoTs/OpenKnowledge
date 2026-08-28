# Escalating on the company's Azure OpenAI, click by click

When the local model cannot ground an answer, the ladder can try a
stronger one. For a company on Microsoft, the sanctioned place for that
stronger model is **Azure OpenAI in your own tenant**: the same models,
inside the same compliance boundary your other Azure workloads live in,
billed per token on your own agreement.

The honest sentence first, because it decides the architecture: **a
Microsoft 365 Copilot seat is not a callable model API.** Microsoft does
not offer "use my Copilot licence as a completion endpoint", so "escalate
on our Copilot subscription" translates to Azure OpenAI. If Microsoft
ever opens a Copilot inference API, the provider seam here is ready
for it.

## What escalation sends, and when

Nothing, until an operator turns it on — `OK_ESCALATION_ENABLED` is off
by default and the product answers entirely locally. Turned on, an
escalated question sends **the question and the retrieved passages** —
excerpts of your documents — to your Azure OpenAI deployment. That is
data leaving the machine, so it is a decision: it stays inside your
tenant, and it happens only when every cheaper tier failed. The answer
that comes back is graded by the same grounding gate as every other
tier, cached like every other answer, and refused if it cannot cite
your documents.

## No company yet?

An individual Azure account works — the same one a personal Entra test
tenant comes from ([ENTRA-SETUP.md](ENTRA-SETUP.md) has the signup
notes). Azure OpenAI no longer requires a company use-case application
for standard models. One catch: if resource creation is refused on a
free-trial subscription, upgrade it to pay-as-you-go (Cost Management →
upgrade) — you still pay only per token, and a test run of a few hundred
escalated questions on a small deployment costs cents. Set
`OK_BUDGET_DAILY_USD` anyway; that is what it is for.

## 1. Create the resource and deploy a model

In the [Azure portal](https://portal.azure.com): **Create a resource →
Azure OpenAI** — pick the subscription, a resource group, a region, and
a name (the name becomes your endpoint:
`https://<name>.openai.azure.com`). Then in [Azure AI
Foundry](https://ai.azure.com), open the resource and **deploy a model**
— pick one and give the deployment a name, e.g. `kb-answers`. The
deployment name is yours, not the model's, and it is what OpenKnowledge
calls.

From the resource's **Keys and Endpoint** page copy the endpoint URL and
a key.

## 2. Configure the server

```sh
OK_ESCALATION_ENABLED=true
OK_ESCALATION_PROVIDER=azure
OK_AZURE_OPENAI_ENDPOINT=https://<name>.openai.azure.com
OK_AZURE_OPENAI_DEPLOYMENT=kb-answers
OK_AZURE_OPENAI_API_KEY=<the key>
```

Restart (or flip the escalation settings in /manage — they rebuild the
engine in place). `/healthz` reports `escalation_enabled` so you can see
which state a server is in.

## 3. State your price, or costs are flagged

Azure prices vary by model, region and agreement, so OpenKnowledge does
not ship a number for your deployment — that would be an invented one.
Copy the two rates from your own Azure price sheet:

```sh
OK_AZURE_OPENAI_INPUT_PER_MTOK=2.50     # USD per million input tokens
OK_AZURE_OPENAI_OUTPUT_PER_MTOK=10.00   # USD per million output tokens
```

With them set, every escalated answer carries its real cost and
`openknowledge costs` adds it to the ledger. Without them, calls still
work and every one is marked **"cost not counted"** in its notes — the
ledger flags what it cannot price rather than guessing. Setting only one
of the two is treated as setting neither, loudly.

Worth pairing with the budget governor: `OK_BUDGET_DAILY_USD` caps what
escalation may spend over a rolling day, and a question the ceiling
blocks is refused with the reason, never answered ungrounded.

## 4. Reasoning-family deployments (gpt-5\*, o\*)

**Needs v0.2.1 or later.** On v0.2.0 and earlier, deploy a non-reasoning
chat model (gpt-4o-mini is the sensible test rung) - those releases send
`max_tokens` and a pinned `temperature`, which the gpt-5 family refuses.

The newest model families speak a slightly different dialect: they refuse
`max_tokens` in favour of `max_completion_tokens`, and refuse a pinned
`temperature`. A deployment name reveals nothing about what is behind it,
so OpenKnowledge does not guess - the first refusal from the endpoint
names the parameter, the provider adopts the dialect and retries, and
every later call speaks it directly. Nothing to configure.

Two things are worth knowing anyway:

- **Point the API at the `v1` surface.** The gpt-5 family is reached
  through Azure's next-generation path, which tracks the latest API
  instead of a dated version:

  ```sh
  OK_AZURE_OPENAI_API_VERSION=v1
  ```

  If your region still rejects that path, the error will say so - a
  recent dated preview from the deployment's own page works too.

- **The thinking is billed as output.** A reasoning model spends tokens
  thinking before it answers, from the same budget and at the output
  rate, so the ledger will show more output tokens than the answer's
  length suggests. That is the real cost, honestly counted - and if the
  budget runs out mid-thought the reply comes back empty; the error
  names `OK_MAX_ANSWER_TOKENS` as the lever.

Determinism is unaffected either way: byte-identical serving comes from
the answer cache, not from the model's temperature.

## When it does not work

| You see | It means | Fix |
|---|---|---|
| HTTP 401 in the escalation error | wrong or expired key | Keys and Endpoint → regenerate, update `.env` |
| HTTP 404 | deployment name or endpoint wrong | the deployment name is the one you chose in AI Foundry, not the model's name |
| HTTP 400 mentioning content filtering | Azure's content filter blocked the request or reply | review the resource's content-filter configuration; corporate policy question, not an OpenKnowledge setting |
| answers still local | escalation half-configured | the startup log names the exact `OK_AZURE_OPENAI_*` variable that is missing |
| "cost not counted" notes | prices not stated | step 3 |

The escalated rung changes only the price of trying again: same prompt,
same passages, same grounding gate, same cache. Run `openknowledge eval`
after enabling it — the point of the ladder is measured, not assumed.
