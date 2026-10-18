#!/usr/bin/env python3
from _common import context, event, status

if __name__ == "__main__":
    event()
    context("SessionStart", "Procedura MCP: the program owns every modelling stage. Read .agents/skills/procedura/SKILL.md for modelling requests. Read request pages one per tool call (never batch cursors) and do not run python3 mcp/workflow.py status while a request is pending.\n" + status())
