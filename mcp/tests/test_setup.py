"""Portable setup preserves user configuration and can be repeated without drift."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mcp"))
import setup
import workflow


class SetupTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="procedura-setup-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "project with spaces"
        self.root.mkdir()
        skill = self.root / ".agents/skills/procedura/SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text("fixture skill\n")
        for target, name, value in ((setup, "ROOT", self.root), (workflow, "ROOT", self.root),
                                    (workflow, "ACTIVE", self.root / "outputs/mcp/ACTIVE_V2"),
                                    (workflow, "TOOLCHAIN", self.root / ".procedura/toolchain.local.json")):
            p = patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(workflow, "check_tools")
        p.start()
        self.addCleanup(p.stop)

    def save(self, path, value):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_both_agents_preserve_models_other_servers_and_hooks_and_are_idempotent(self):
        config = self.root / ".codex/config.toml"
        config.parent.mkdir()
        config.write_text('model = "my-model"\nmodel_auto_compact_token_limit = 220000\n[features]\nother = true\n[mcp_servers.other]\ncommand = "other"\n')
        user_hook = {"matcher": "Bash", "hooks": [{"type": "command", "command": "my-checker"}]}
        self.save(".codex/hooks.json", {"description": "mine", "hooks": {"PreToolUse": [user_hook]}})
        self.save(".claude/settings.local.json", {"model": "my-claude", "permissions": {"deny": ["Bash(rm *)"]}, "hooks": {"PreToolUse": [user_hook]}})
        self.save(".mcp.json", {"mcpServers": {"other": {"type": "stdio", "command": "other"}}})
        setup.configure("both", {"fixture": True})
        first = setup.configurations("both")
        self.assertTrue(all(path.read_text() == text for path, text in first.items()))
        setup.configure("both", {"fixture": True})
        self.assertEqual(first, setup.configurations("both"))
        setup.configure("both", {"fixture": True}, check=True)
        parsed = tomllib.loads(config.read_text())
        self.assertEqual(parsed["model"], "my-model")
        self.assertEqual(parsed["model_auto_compact_token_limit"], 220000)
        self.assertEqual(parsed["mcp_servers"]["other"]["command"], "other")
        for name in (".codex/hooks.json", ".claude/settings.local.json"):
            value = json.loads((self.root / name).read_text())
            self.assertEqual(value["hooks"]["PreToolUse"][0], user_hook)
            self.assertEqual(len(value["hooks"]["PreToolUse"]), 2)
        claude = json.loads((self.root / ".claude/settings.local.json").read_text())
        self.assertEqual(claude["model"], "my-claude")
        self.assertEqual(claude["permissions"]["deny"], ["Bash(rm *)"])
        self.assertFalse((self.root / ".env").exists())

    def test_claude_only_does_not_create_codex_settings(self):
        setup.configure("claude", {})
        self.assertFalse((self.root / ".codex").exists())
        server = json.loads((self.root / ".mcp.json").read_text())["mcpServers"]["procedura"]
        self.assertEqual(server["args"][-1], str(self.root))
        self.assertEqual((self.root / ".claude/skills/procedura/SKILL.md").read_text(), "fixture skill\n")

    def test_unmanaged_name_collision_is_reported_before_any_write(self):
        self.save(".mcp.json", {"mcpServers": {"procedura": {"command": "some-other-server"}}})
        with self.assertRaisesRegex(ValueError, "already exists"):
            setup.configure("both", {})
        self.assertFalse((self.root / ".codex/config.toml").exists())
        self.assertFalse((self.root / ".procedura").exists())

    def test_disabled_codex_hooks_are_not_silently_reenabled(self):
        with self.assertRaisesRegex(ValueError, "explicitly false"):
            setup.codex_config("[features]\nhooks=false\n")

    def test_check_does_not_install_or_write_missing_configuration(self):
        with self.assertRaisesRegex(ValueError, "Missing/stale"):
            setup.configure("both", {}, check=True)
        self.assertFalse((self.root / ".codex").exists())

    def test_active_run_blocks_setup_before_changing_configuration(self):
        workflow.ACTIVE.parent.mkdir(parents=True)
        workflow.ACTIVE.write_text("running_case\n")
        with patch.object(workflow, "resolve_run", return_value=self.root / "outputs/mcp/running_case"), patch.object(workflow, "state", return_value={"phase": "running"}):
            with self.assertRaisesRegex(ValueError, "active MCP run"):
                setup.configure("both", {})
        self.assertFalse((self.root / ".mcp.json").exists())

    def test_no_codex_or_proxy_is_needed_to_discover_native_tools(self):
        binary = self.root / "native-tool"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        args = argparse.Namespace(openscad=str(binary), blender=str(binary), isaac=str(self.root / "no-isaac"))
        with patch.object(setup, "executable", return_value=str(binary)), patch.object(setup, "output", side_effect=lambda a: "--backend Manifold" if "--help" in a else "fixture-version"):
            chain = setup.discover(args)
        self.assertNotIn("codex", chain)
        self.assertNotIn("codex_provider", chain)
        self.assertIn("blender", chain["artifacts"])


if __name__ == "__main__":
    unittest.main()
