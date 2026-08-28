"""Azure OpenAI: the corporate-sanctioned escalation rung.

The honest sentence behind this module: **a Microsoft 365 Copilot seat is
not a callable model API.** When a company says "escalate on our Copilot
subscription", the thing IT can actually approve is Azure OpenAI in the
company's own tenant - the same models, inside the same compliance
boundary, billed per token on the company's own agreement. That is what
this provider reaches.

The dialect differs from OpenAI's in exactly two places, so this is a
thin subclass rather than a second adapter: the URL names a *deployment*
(the company's chosen name for a model they provisioned) with an
``api-version`` query, and the key travels in an ``api-key`` header
instead of a bearer token. Everything else - the prompt, the grounding
gate, temperature zero, the fit check - is inherited unchanged, because a
rung is only allowed to change the price of trying again.

One thing this provider cannot inherit: a price. Azure charges by
deployment, region and agreement, so shipping "the" price for an
arbitrary deployment name would be an invented number. The operator
states their own from their own price sheet
(``OK_AZURE_OPENAI_INPUT_PER_MTOK`` / ``OK_AZURE_OPENAI_OUTPUT_PER_MTOK``);
until they do, every call's cost is flagged as uncounted rather than
guessed at - the ledger stays honest either way.
"""

from __future__ import annotations

from ..costs import ModelPrice
from .openai_compat import OpenAICompatProvider


class AzureOpenAIProvider(OpenAICompatProvider):
    """Chat provider for an Azure OpenAI deployment in the company tenant."""

    def __init__(
        self,
        *,
        endpoint: str,
        deployment: str,
        api_key: str,
        api_version: str = "2024-06-01",
        timeout: float = 120.0,
        input_per_mtok: float | None = None,
        output_per_mtok: float | None = None,
    ) -> None:
        # `v1` selects Azure's next-generation surface: one un-versioned path
        # that tracks the latest API, with the deployment named in the body
        # rather than the URL. It is what the newest model families (gpt-5*)
        # are reached through; dated api-versions keep working for the rest.
        v1 = api_version == "v1"
        super().__init__(
            model_id=deployment,
            base_url=(
                f"{endpoint.rstrip('/')}/openai/v1"
                if v1
                else f"{endpoint.rstrip('/')}/openai/deployments/{deployment}"
            ),
            api_key=api_key,
            tier="frontier",
            # Azure bills per token whatever the hostname looks like.
            self_hosted=False,
            timeout=timeout,
        )
        self.api_version = api_version
        #: The operator's own numbers, or None - in which case the router
        #: flags each call as cost-not-counted instead of inventing a price.
        self.price_override: ModelPrice | None = None
        if input_per_mtok is not None and output_per_mtok is not None:
            self.price_override = ModelPrice(
                model_id=deployment,
                provider="azure-openai",
                tier="frontier",
                input_per_mtok=input_per_mtok,
                output_per_mtok=output_per_mtok,
            )

    def _headers(self) -> dict[str, str]:
        # Azure's data plane authenticates with `api-key`, not a bearer.
        return {"Content-Type": "application/json", "api-key": self._api_key or ""}

    def _url(self) -> str:
        if self.api_version == "v1":
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/chat/completions?api-version={self.api_version}"
