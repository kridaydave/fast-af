# Web search 402 diagnosis

Date checked: 2026-11-11 (session context)
Environment: Linux (/home/kriday), OpenCode harness, T3 Code thread, provider opencode/iceman-free

## What was checked

1. Config files and search integration settings
- Read `~/.config/opencode/opencode.json`: only MCP servers configured (`cua-driver`, `vercel`, `doop`). No `websearch` block, no search provider key fields.
- Read `~/fleet/opencode.json`: instructions only, no `websearch` config.
- Read `~/.t3/caches/opencode.json`: client/runtime cache (auth status, model list), no search provider config.
- Grepped for `search`, `SERPER`, `TAVILY`, `BRAVE`, `BING`, `EXA`, `FIRECRAWL`, `PARALLEL`, `TINYFISH`, `402` under `~/.config/opencode`, `~/.opencode`, `~/fleet`: no configured search-provider keys or websearch settings found (matches were unrelated skill/docs text).
- Searched common profile/config locations for provider API key variable names (`EXA_API_KEY`, `TAVILY_API_KEY`, `FIRECRAWL_API_KEY`, `PARALLEL_API_KEY`, `TINYFISH_API_KEY`, etc.): none found present. (Only key names/presence were checked; no secret values were read or printed.)

2. OpenCode runtime state (local)
- Inspected `~/.local/share/opencode/opencode.db` (tables listed via sqlite3 Python module). KV lookup found: `websearch:provider => "random"`.
- Queried session parts/messages for stored `402`/`websearch` text: no matching rows returned.
- Checked `~/.local/share/opencode/log/opencode.log` for `websearch`, `Payment Required`, `402`, or search-failure lines: no direct websearch/402 error lines found (grep hits were unrelated "research" path strings).
- Environment variables in this session: no search-provider API key variables observed by name in the filtered env output.

3. Fetch-based verification and docs
- `webfetch` works in this session (task states https://github.com/casper-hansen/AutoAWQ fetched successfully today).
- Fetched OpenCode docs:
  - https://opencode.ai/docs/tools/ (websearch described as using the selected integration; permission `websearch`)
  - https://opencode.ai/docs/websearch/
    - Built-in providers: Exa (`exa`), Firecrawl (`firecrawl`), Parallel (`parallel`), Tavily (`tavily`), TinyFish (`tinyfish`).
    - Provider chosen via `/connect` in TUI or env vars before start; can be set in config as `websearch.provider`. `"random"` picks an available provider automatically and the session keeps using it until rate limited.
    - Limits: when a provider returns HTTP 429, random selection retries with another available provider and the rate-limited provider cools down (`Retry-After` or 60s). If every provider is cooling down, search fails immediately.
    - No mention of rotating on HTTP 402 or provider-specific quota codes like 432.
- Fetched GitHub issues:
  - https://github.com/anomalyco/opencode/issues/11982 "Exa websearch returns 402 Payment Required" (closed, Feb 2026): Exa-side payment/quota incident; resolved by Exa. Shows 402 can come from the search vendor/backend.
  - https://github.com/anomalyco/opencode/issues/60176 "Web search: rotate providers on quota and plan-limit failures, not only HTTP 429" (open): reports that with `websearch.provider: "random"` rotation happens only on 429; quota/plan-limit responses (402, Tavily 432) fail outright with no fallback, leaving a session stuck on an exhausted provider.

4. Endpoint/keyless probing
- I cannot identify or safely test the exact vendor endpoint/keyless path OpenCode used for this session without guessing internals or using secrets. No unauthenticated vendor health/docs endpoint was probed as a proxy for the failing request because that would not prove which provider/path returned the reported 402.

## Findings

- Search provider is not explicitly configured in repo/global OpenCode config files checked. Local OpenCode state records `websearch:provider = "random"`.
- No search-provider API keys (Exa/Firecrawl/Parallel/Tavily/TinyFish) were found configured in the checked config/profile locations.
- Web fetch is healthy; the failure is specific to web search, consistent with the search provider/integration path rather than general outbound HTTP.
- Per OpenCode docs, only HTTP 429 triggers provider rotation/cooldown in random mode. Per issue #60176, HTTP 402/432 (quota/payment/plan-limit style failures) do not rotate, so a random session can stick to an exhausted or payment-gated provider and fail with 402.
- The exact provider and response body for this session's 402 are not available in local logs/db text checked, so the vendor vs OpenCode-proxy origin of the 402 cannot be proven from local evidence alone.

## Verdict

cause (1 line): Most likely the session's randomly selected web search provider returned HTTP 402 for quota/payment/plan-limit reasons and OpenCode (with `websearch:provider` set to `random`) did not rotate off it (rotation is documented/implemented for 429, while 402/432 failover is a known open gap, #60176); exact provider and whether the 402 came from the vendor or an OpenCode-side proxy is unresolved from local evidence.

fix for Kriday (1 line): Configure a working web search provider/key for OpenCode (set `websearch.provider` to a provider you have access to and provide its API key via `/connect` or the provider env var like `EXA_API_KEY`/`TAVILY_API_KEY`/`FIRECRAWL_API_KEY`/`PARALLEL_API_KEY`/`TINYFISH_API_KEY`, then restart the session/server) or clear/change the stored random selection so a healthy provider is picked.

workarounds (bullets):
- Use `webfetch` for known URLs and docs pages (confirmed working) instead of `websearch` when you can target sources directly.
- Set an explicit `websearch.provider` in `opencode.json`/`opencode.jsonc` to a provider with available quota and supply its API key; avoid `"random"` until 402/432 rotation is handled (#60176).
- If a session is stuck on a bad random choice, start a new session/restart the OpenCode server (docs note moving/restarting clears remembered provider; cooldowns are shared in the workspace) after configuring a good provider.
- If you rely on keyless/default Exa search, check Exa/OpenCode service status or quota limits; a vendor-side 402 (like #11982 historically) requires the provider/service quota to be restored.

## UNRESOLVED items

- Exact search provider selected for the failing request(s) in this session (not logged locally in checked files/db fields).
- Exact HTTP response/body and whether 402 originated at the upstream vendor endpoint or an OpenCode proxy layer (no matching log line found; endpoint not safely identifiable without guessing).
- Current quota/payment state of each configured-or-available provider account (no keys/accounts found locally to inspect; would require vendor dashboards).

End with: cause (1 line), fix for Kriday (1 line), workarounds (bullets).
(See "Verdict" section above for the required lines.)