# GPT 4.1 → GPT 5 Upgrade Notes

## Known Differences (Confirmed)

### 1. `temperature` parameter rejected
- GPT 5 **does not support custom `temperature`** values. Only the default (`1`) is accepted.
- Error: `"Unsupported value: 'temperature' does not support 0.7 with this model. Only the default (1) value is supported."`
- **Impact**: Any code passing `temperature` (LangChain default, direct SDK) will break.
- **Fix**: Remove `temperature` parameter or set to `1` when using GPT 5.

### 2. `max_tokens` → `max_completion_tokens`
- GPT 5 **rejects the `max_tokens` parameter**. Must use `max_completion_tokens` instead.
- Error: `"Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."`
- **Impact**: Both client code AND server-side features (like On Your Data/BYOD) are affected.

### 3. BYOD / On Your Data broken (server-side)
- Even when client code is fixed, the **Azure On Your Data pipeline itself** sends `max_tokens` internally when calling the model. This causes a 400 error on GPT 5.
- This is an **Azure-side issue** — the On Your Data service has not been updated to use `max_completion_tokens` for GPT 5.
- **No client-side workaround exists** for this specific issue.

#### BYOD request flow and root cause

```
Client App → Azure OpenAI API → [On Your Data pipeline] → Azure AI Search
                                         ↓
                                 Retrieves search results
                                         ↓
                                 Internally calls GPT model
                                 with max_tokens parameter
                                         ↓
                                 GPT 5 rejects max_tokens ❌
                                 (requires max_completion_tokens)
```

The client app does NOT send `max_tokens` — the error originates inside Azure's
On Your Data orchestration layer. The error message confirms this:

> `"An error occurred when calling Azure OpenAI: ... 'max_tokens' is not
> supported with this model. Use 'max_completion_tokens' instead."`

The phrase "when calling Azure OpenAI" indicates Azure's own pipeline hit the
error when making its internal call to the GPT 5 model with search-grounded
context. Microsoft must update the On Your Data service to use
`max_completion_tokens` for GPT 5 family models.

## Reproduction Steps

1. Start with baseline app on GPT 4.1: `python app.py --mode byod` → works ✅
2. Change `.env` to `AZURE_OPENAI_DEPLOYMENT=gpt-5`
3. Run `python app.py --mode direct` → fails with `temperature` error
4. Run `python app.py --mode chat` → fails with `temperature` error
5. Run `python app.py --mode byod` → fails with `max_tokens` error (server-side BYOD pipeline)

## Findings

- **`direct` mode**: Breaks due to `temperature=0.7`. Fixable by removing the parameter.
- **`chat` mode**: Same `temperature` issue from LangChain's `AzureChatOpenAI`. Fixable.
- **`byod` mode**: Breaks with `max_tokens` error originating from Azure's On Your Data service, not from our code. **Not fixable client-side** — requires Azure service update.

## Resolution

### Client-side fixes (chat/direct modes)
- Remove or conditionally set `temperature` based on model
- Use `max_completion_tokens` instead of `max_tokens` in any explicit token limits

### BYOD / On Your Data → Foundry IQ Knowledge Base
- **BYOD is deprecated** and does not support GPT 5.
- **Migration**: Replaced with Foundry IQ direct KB retrieve API + GPT-5 Responses API:
  1. Azure AI Search knowledge source + knowledge base (agentic retrieval)
  2. KB configured with API-key auth for model-based query planning (no RBAC needed)
  3. Direct `POST /knowledgebases/{name}/retrieve` API call for retrieval
  4. GPT-5 Responses API for answer synthesis with retrieved context
- **Setup**: Run `python scripts/setup_foundry_iq.py` to provision resources
- **Usage**: `python app.py --mode foundry` for interactive chat
- **Evals**: `python -m evals.eval_byod --use-foundry` to run eval suite
- **No RBAC required**: Uses API-key auth for model, bearer token for search
