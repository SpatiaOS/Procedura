import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mcp/hooks"))
from pre_tool_use import permitted


class HookTests(unittest.TestCase):
    def shell(self, command):
        return permitted({"cwd": str(ROOT), "tool_name": "exec_command", "tool_input": {"cmd": command}}, ROOT / "runs/test")

    def test_fixed_commands_are_allowed(self):
        self.assertTrue(self.shell("python3 mcp/workflow.py status --log"))
        self.assertTrue(self.shell("python3 mcp/workflow.py run"))
        self.assertTrue(self.shell("cat runs/test/.cx/completion.json"))
        self.assertTrue(self.shell("cat AGENTS.md"))
        self.assertTrue(permitted({"tool_name": "clocksleep", "tool_input": {"duration_ms": 30000}}, ROOT))

    def test_shell_chains_edits_and_exploration_are_denied(self):
        for cmd in ["python3 mcp/workflow.py status; touch evil", "python3 mcp/workflow.py status --log;touch evil", "python3 mcp/workflow.py status && touch evil",
                    "python3 mcp/workflow.py status$(touch evil)", "python3 mcp/workflow.py status\ntouch evil",
                    "python3 mcp/workflow.py status > runs/test/plan.json", "python3 mcp/workflow.py part skip wheel",
                    "cat ../Tether/.env", "bun runtime/run.ts", "python3 -c 'print(1)'", "rg --files /home"]:
            with self.subTest(cmd=cmd):
                self.assertFalse(self.shell(cmd))

    def test_model_writes_delegation_and_interactive_shell_input_are_denied(self):
        for name in ("apply_patch", "spawn_agent", "web_search"):
            self.assertFalse(permitted({"tool_name": name}, ROOT / "runs/test"))
        self.assertFalse(permitted({"tool_name": "write_stdin", "tool_input": {"chars": "touch evil\n"}}, ROOT))
        self.assertTrue(permitted({"tool_name": "write_stdin", "tool_input": {"chars": ""}}, ROOT))

    def test_claude_reads_and_background_task_polling_are_allowed(self):
        run = ROOT / "runs/test"
        for path in (ROOT / "AGENTS.md", ROOT / "CLAUDE.md", ROOT / ".agents/skills/procedura/SKILL.md", run / "preview_final/pbr-isometric.png"):
            self.assertTrue(permitted({"tool_name": "Read", "tool_input": {"file_path": str(path)}}, run))
        self.assertFalse(permitted({"tool_name": "Read", "tool_input": {"file_path": str(ROOT / "mcp/workflow.py")}}, run))
        self.assertFalse(permitted({"tool_name": "Read", "tool_input": {"file_path": str(run / "plan.json")}}, run))
        self.assertTrue(permitted({"tool_name": "TaskOutput", "tool_input": {"task_id": "worker", "block": False}}, run))
        self.assertFalse(permitted({"tool_name": "TaskOutput", "tool_input": {}}, run))
        self.assertFalse(permitted({"tool_name": "Task", "tool_input": {}}, run))

    def test_only_procedura_skill_and_tool_discovery_are_allowed_during_a_run(self):
        run = ROOT / "outputs/mcp/test"
        self.assertTrue(permitted({"tool_name": "Skill", "tool_input": {"skill": "procedura"}}, run))
        self.assertFalse(permitted({"tool_name": "Skill", "tool_input": {"skill": "other"}}, run))
        self.assertTrue(permitted({"tool_name": "ToolSearch", "tool_input": {"query": "select:mcp__procedura__next_request"}}, run))
        self.assertFalse(permitted({"tool_name": "ToolSearch", "tool_input": {"query": "other tools"}}, run))

    def test_execution_host_can_wrap_only_one_checked_literal_call(self):
        def host(code):
            return permitted({"cwd": str(ROOT), "tool_name": "functions.exec", "tool_input": code}, ROOT / "runs/test")
        self.assertTrue(host('text(await tools.exec_command({"cmd":"python3 mcp/workflow.py status"}));'))
        self.assertTrue(host('const r = await tools.write_stdin({"session_id":42,"chars":""}); text(r);'))
        self.assertFalse(host('text(await tools.exec_command({"cmd":"touch evil"}));'))
        self.assertFalse(host('text(await tools.exec_command({"cmd":"python3 mcp/workflow.py status"})); await tools.apply_patch("evil");'))
        self.assertFalse(host('text(await tools.exec_command({"cmd":getCommand()}));'))

    def test_only_tether_generation_tools_can_submit_model_artifacts(self):
        run=ROOT/'runs/test'
        self.assertTrue(permitted({'tool_name':'mcp__procedura__next_request','tool_input':{}},run))
        self.assertTrue(permitted({'tool_name':'procedura.submit_artifact','tool_input':{'request_id':'x','artifact_text':'cube(1);'}},run))
        self.assertFalse(permitted({'tool_name':'mcp__other__submit_artifact','tool_input':{}},run))
        self.assertFalse(permitted({'tool_name':'image_gen','tool_input':{}},run))
        code='const r = await tools.mcp__procedura__next_request({"wait_seconds":20}); for (const c of r.content) { if (c.type === "image") image(c); else if (c.type === "text") text(c.text); }'
        self.assertTrue(permitted({'tool_name':'functions.exec','tool_input':code},run))
        batched='const id="req"; for (let cursor=1;cursor<=7;cursor++){const r=await tools.mcp__procedura__next_request({request_id:id,cursor}); for(const c of(r?.content||[])){if(c.type==="text")text(c.text);else if(c.type==="image")image(c);}}'
        self.assertFalse(permitted({'tool_name':'functions.exec','tool_input':batched},run))  # batched pages are truncated by the host


if __name__ == "__main__":
    unittest.main()
