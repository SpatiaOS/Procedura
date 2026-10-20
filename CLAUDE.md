@AGENTS.md

For MCP modelling, read `.agents/skills/procedura/SKILL.md`. Use the native
`procedura` MCP tools. Start `python3 mcp/workflow.py run` with Bash
`run_in_background=true`, then service its requests in this conversation. Use
`TaskOutput` only to poll the existing background task. Do not delegate generation
or call an additional model API.
