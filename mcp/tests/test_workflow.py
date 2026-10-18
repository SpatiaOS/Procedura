"""Launcher invariants. All run mutations happen in temporary directories."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mcp"))
import workflow as cx


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="Procedura MCP-test-")
        self.addCleanup(self.temp.cleanup)
        self.runs = Path(self.temp.name) / "runs"
        self.patches = [patch.object(cx, "RUNS", self.runs), patch.object(cx, "ACTIVE", self.runs / "ACTIVE_V2"),
                        patch.object(cx, "toolchain", return_value={"test": True}), patch.object(cx, "check_tools")]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def invoke(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return cx.main(list(args))

    def new(self, *extra):
        self.assertEqual(self.invoke("new", "fixture", "--brief", "  A box\n", "--text-only", *extra), 0)
        return self.runs / "fixture"

    def test_freezes_prompt_profile_and_rejects_manifest_edit(self):
        run = self.new()
        spec = cx.validate_inputs(run)
        expected = (ROOT / "prompts/image_prompt.md").read_text().replace("{text}", "A box")
        self.assertEqual((run / ".cx/reference_prompt.txt").read_text(), expected)
        self.assertEqual(spec["profile"]["options"]["maxSteps"], 6)
        spec["options"]["refine"] = False
        cx.write(run / ".cx/inputs.json", spec)
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            cx.validate_inputs(run)

    def test_changed_brief_pack_or_reference_cannot_resume(self):
        run = self.new()
        with patch.object(cx, "pack_hash", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "Pack/profile changed"):
                cx.validate_inputs(run)
        (run / "image.png").write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "Text-only"):
            cx.validate_inputs(run)
        (run / "brief.md").write_text("other")
        with self.assertRaisesRegex(ValueError, "brief changed"):
            cx.validate_inputs(run)

    def test_only_explicit_scope_reductions_are_recorded(self):
        run = self.new("--no-motion", "--no-paint", "--no-assembly")
        spec = cx.validate_inputs(run)
        self.assertEqual(spec["options"], {"motion": False, "motionAware": False,
                         "incrementalMotion": False, "paint": False, "assembly": False})
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            cx.main(["new", "other", "--brief", "x", "--no-refine"])

    def test_approved_model_override_is_frozen_without_changing_effort_or_pipeline(self):
        run = self.new("--model", "gpt-5.6-sol")
        spec = cx.validate_inputs(run)
        self.assertEqual(spec["model"], "gpt-5.6-sol")
        self.assertEqual(spec["reasoning_effort"], "current-session")
        self.assertEqual(spec["profile"]["options"], cx.read(cx.PROFILE)["options"])

    def test_reference_freezes_after_set(self):
        image = Path(self.temp.name) / "reference.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        self.assertEqual(self.invoke("new", "fixture", "--brief", "x"), 0)
        self.assertEqual(self.invoke("ref", "set", str(image)), 0)
        self.assertEqual(self.invoke("ref", "set", str(image)), 1)
        run = self.runs / "fixture"
        (run / "image.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Reference image changed"):
            cx.validate_inputs(run)

    def test_gpt61_sol_override_preserves_the_fixed_session_profile(self):
        run = self.new("--model", "gpt-6.1-sol")
        spec = cx.validate_inputs(run)
        self.assertEqual(spec["model"], "gpt-6.1-sol")
        self.assertEqual(spec["reasoning_effort"], "current-session")
        self.assertEqual(spec["profile"]["transport"], "session")
        self.assertEqual(spec["options"], {})
        self.assertEqual(spec["profile"]["options"], cx.read(cx.PROFILE)["options"])

    def test_active_worker_lock_and_unfinished_run_prevent_duplicates(self):
        run = self.new()
        with cx.run_lock(run):
            self.assertTrue(cx.is_running(run))
            self.assertEqual(self.invoke("run"), 1)
            self.assertEqual(self.invoke("abort", "--reason", "test"), 1)
        self.assertFalse(cx.is_running(run))
        self.assertEqual(self.invoke("new", "other", "--brief", "x"), 1)
        self.assertEqual(self.invoke("finish"), 1)

    def test_custom_host_model_is_metadata_and_not_a_hardcoded_catalog(self):
        run = self.new("--model", "my-provider/my-private-model")
        spec = cx.validate_inputs(run)
        self.assertEqual(spec["model"], "my-provider/my-private-model")
        self.assertEqual(spec["reasoning_effort"], "current-session")
        self.assertEqual(spec["options"], {})

    def test_permission_approvals_do_not_change_pack_but_hook_changes_do(self):
        root = Path(self.temp.name) / "pack"
        for rel in ("mcp/engine.lock.json", "mcp/profile.json", "bun.lock", "package.json", "AGENTS.md", "CLAUDE.md"):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        settings = root / ".claude/settings.local.json"
        cx.write(settings, {"hooks": {"Stop": []}})
        with patch.object(cx, "ROOT", root), patch.object(cx, "LOCK", root / "mcp/engine.lock.json"), patch.object(cx, "PROFILE", root / "mcp/profile.json"):
            original = cx.pack_hash()
            cx.write(settings, {"hooks": {"Stop": []}, "permissions": {"allow": ["Bash(python3 mcp/workflow.py run)"]}, "enabledMcpjsonServers": ["procedura"]})
            self.assertEqual(cx.pack_hash(), original)
            cx.write(settings, {"hooks": {"Stop": [{"hooks": [{"command": "changed"}]}]}})
            self.assertNotEqual(cx.pack_hash(), original)

    def test_claude_gemini_override_records_the_outer_model_without_changing_stages(self):
        run = self.new("--model", "gemini-4-argon-high")
        spec = cx.validate_inputs(run)
        self.assertEqual(spec["model"], "gemini-4-argon-high")
        self.assertEqual(spec["reasoning_effort"], "current-session")
        self.assertEqual(spec["profile"]["transport"], "session")
        self.assertEqual(spec["options"], {})
        self.assertEqual(spec["profile"]["options"], cx.read(cx.PROFILE)["options"])

    def test_failed_worker_records_error_and_modified_checkpoint_is_refused(self):
        run = self.new()
        def failed(*args):
            (run / "draft.scad").write_text("cube(2);")
            return 1
        with patch.object(cx, "execute", side_effect=failed) as worker:
            self.assertEqual(self.invoke("run"), 1)
            self.assertEqual(cx.state(run)["phase"], "error")
            (run / "draft.scad").write_text("cube(200);")
            self.assertEqual(self.invoke("run"), 1)
            self.assertEqual(worker.call_count, 1)

    def test_prewritten_plan_cannot_skip_generation(self):
        run = self.new()
        (run / "plan.json").write_text("[]")
        with patch.object(cx, "execute") as worker:
            self.assertEqual(self.invoke("run"), 1)
            worker.assert_not_called()

    def test_ambient_flags_and_injection_are_not_workflow_controls(self):
        spec = {"profile": cx.read(cx.PROFILE), "toolchain": {"openscad": "/scad", "blender": "/blender", "python": "/python/bin/python3"}}
        with patch.dict(os.environ, {"PROCEDURA_NO_PLAN": "1", "PROCEDURA_SELECT_DRAFTS": "100", "BUN_OPTIONS": "--preload evil",
                                    "NODE_OPTIONS": "--require evil", "OPENAI_API_KEY": "secret", "https_proxy": "http://proxy"}):
            env = cx.engine_env(spec)
        self.assertEqual(env["PROCEDURA_NO_PLAN"], "0")
        for name in ("PROCEDURA_SELECT_DRAFTS", "BUN_OPTIONS", "NODE_OPTIONS", "OPENAI_API_KEY"):
            self.assertNotIn(name, env)
        self.assertEqual(env["https_proxy"], "http://proxy")
        self.assertTrue(env["PATH"].startswith("/python/bin:"))

    def test_completion_reports_degradation_and_seals_outputs(self):
        run = self.new()
        spec = cx.validate_inputs(run)
        for name in ("final.scad", "final.obj", "final_summary.txt", "plan.json"):
            (run / name).write_text("fixture")
        cx.write(run / "parts_summary.json", {"parts": [{"name": "wheel", "generated": False}], "partsGenerated": 0, "plan": [{"name": "wheel"}]})
        result = {"refine": {"ok": True, "verdict": "give_up"}, "paint": {"ok": True, "degraded": ["no subpart colours"]}, "motion": {"ok": True}}
        cx.write(run / ".cx/result.json", result)
        receipt = cx.completion(run, spec, result)
        self.assertEqual(receipt["status"], "done_with_limitations")
        self.assertTrue(any("Isaac" in x for x in receipt["limitations"]))
        self.assertTrue(any("subpart colours" in x for x in receipt["limitations"]))
        self.assertTrue(any("wheel" in x for x in receipt["limitations"]))
        cx.write(run / ".cx/completion.json", receipt)
        cx.set_state(run, receipt["status"])
        self.assertEqual(self.invoke("finish"), 0)
        (run / "final.obj").write_text("changed")
        self.assertEqual(self.invoke("finish"), 1)

    def test_source_is_the_reviewed_engine_bytes(self):
        info = cx.verify_source()
        self.assertGreater(info["files"], 100)

    def complete_fixture(self):
        run = self.new("--no-paint", "--no-motion")
        for name in ("final.scad", "final.obj", "final_summary.txt", "plan.json"):
            (run / name).write_text("fixture")
        cx.write(run / "parts_summary.json", {"parts": [{"name": "base", "generated": True}], "partsGenerated": 1, "plan": [{"name": "base"}]})
        cx.write(run / ".cx/result.json", {"refine": {"ok": True, "verdict": "ok"}})
        receipt = cx.completion(run, cx.read(run / ".cx/inputs.json"), cx.read(run / ".cx/result.json"))
        cx.publish_completion(run, receipt)
        return run

    def test_finished_run_verification_survives_pack_and_toolchain_updates(self):
        run = self.complete_fixture()
        with patch.object(cx, "pack_hash", return_value="updated-software"), patch.object(cx, "check_tools", side_effect=ValueError("tool removed")):
            self.assertEqual(self.invoke("finish"), 0)
            self.assertEqual(self.invoke("run"), 0)
        receipt = cx.read(run / ".cx/completion.json")
        receipt["limitations"] = []
        cx.write(run / ".cx/completion.json", receipt)
        self.assertEqual(self.invoke("finish"), 1)

    def test_finished_run_still_rejects_changed_inputs_and_extra_outputs(self):
        run = self.complete_fixture()
        (run / "extra.scad").write_text("cube(1);")
        self.assertEqual(self.invoke("finish"), 1)
        (run / "extra.scad").unlink()
        (run / ".cx/result.json").write_text("{}")
        self.assertEqual(self.invoke("finish"), 1)

    def test_status_omits_artifact_inventory_and_identifies_recovered_isaac(self):
        run = self.complete_fixture()
        receipt = cx.read(run / ".cx/completion.json")
        receipt["artifacts"].update({f"file_{i}.obj": "0" * 64 for i in range(5000)})
        receipt["limitations"] = ["Motion: isaac validation: first attempt exited 1", "Motion: URDF limitation", "Motion: URDF limitation"]
        cx.write(run / ".cx/completion.json", receipt)
        cx.write(run / ".cx/result.json", {"motion": {"ok": True, "validation": {"ran": True, "ok": True, "strictOk": False, "assetRuleErrors": 12}}})
        (run / ".cx/pipeline.log").write_text("[inc-draft] part 3/44: front_left_tow_shackle (L2)\n [gen 3] committed WITH WARNING\n[inc-draft] part 5/44: roll_cage (L0)\n [gen 3] committed WITH WARNING\n")
        out = io.StringIO()
        with contextlib.redirect_stdout(out): cx.status(argparse.Namespace(log=False))
        self.assertLess(len(out.getvalue()), 5000)
        summary = json.loads(out.getvalue())["completion"]
        self.assertNotIn("artifacts", summary)
        self.assertEqual(summary["limitation_count"], 1)
        self.assertEqual(len(summary["recovered_attempt_warnings"]), 1)
        self.assertEqual(summary["isaac"]["strictOk"], False)
        self.assertEqual(summary["historical_draft_warning_parts"], ["front_left_tow_shackle", "roll_cage"])

    def test_running_state_clears_the_previous_attempt_error(self):
        run = self.new()
        cx.set_state(run, "error", error="old failure")
        cx.set_state(run, "running", pid=123)
        self.assertFalse(cx.state(run).get("error"))

    def test_wait_only_polls_the_existing_worker(self):
        self.new()
        with patch.object(cx, "is_running", return_value=False), patch.object(cx, "execute") as worker:
            self.assertEqual(self.invoke("wait", "--seconds", "1"), 0)
            worker.assert_not_called()

    def test_successful_orphan_can_be_recovered_without_another_model_call(self):
        run = self.complete_fixture()
        spec = cx.read(run / ".cx/inputs.json")
        cx.set_state(run, "interrupted")
        cx.write(run / ".cx/worker.json", {"id": "attempt-one"})
        cx.write(run / ".cx/worker-success.json", {"worker_id": "attempt-one", "pack_sha256": spec["pack_sha256"],
                  "result_sha256": cx.sha(run / ".cx/result.json"), "artifacts": cx.output_hashes(run)})
        with patch.object(cx, "execute") as worker:
            self.assertEqual(self.invoke("run"), 0)
            worker.assert_not_called()
        self.assertEqual(self.invoke("finish"), 0)
        cx.set_state(run, "interrupted")
        (run / "final.obj").write_text("tampered")
        with patch.object(cx, "execute") as worker:
            self.assertEqual(self.invoke("run"), 1)
            worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
