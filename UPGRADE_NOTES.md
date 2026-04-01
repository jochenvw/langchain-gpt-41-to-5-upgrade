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

### BYOD / On Your Data
- **Blocked**: Waiting on Azure to update the On Your Data pipeline to support `max_completion_tokens` for GPT 5 models.
- **Workaround**: Continue using GPT 4.1 for BYOD workloads until Azure resolves this.
- Track: https://aka.ms/aoaioydauthentication for updates
