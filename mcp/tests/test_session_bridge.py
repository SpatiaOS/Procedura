import argparse
import base64
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mcp"))
import workflow as cx
import bridge


class SessionBridgeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="cx-session-test-")
        self.addCleanup(temp.cleanup)
        self.runs = Path(temp.name) / "runs"
        for p in [patch.object(cx, "RUNS", self.runs), patch.object(cx, "ACTIVE", self.runs / "ACTIVE_V2"),
                  patch.object(cx, "toolchain", return_value={"test": True}), patch.object(cx, "check_tools"),
                  patch.object(bridge, "MIN_PAGE_SECONDS", 0)]:  # tests read pages without model turns
            p.start(); self.addCleanup(p.stop)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cx.main(["new", "fixture", "--brief", "small object", "--text-only"]), 0)
        self.run = self.runs / "fixture"
        self.lock = cx.run_lock(self.run); self.lock.__enter__()
        self.addCleanup(lambda: self.lock.__exit__(None, None, None))
        cx.set_state(self.run, "running")
        cx.write(self.run / ".cx/worker.json", {"id": "attempt-one"})

    def packet(self, name="request-one", order=1, text="Generate SCAD.", kind="artifact"):
        directory = self.run / ".cx/calls" / name
        directory.mkdir(parents=True)
        cx.write(directory / "request.json", {"id": name, "transport": "session", "attempt_id": "attempt-one", "order": order,
            "kind": kind, "deadline_at": time.time() * 1000 + 60000,
            "system": [{"text": "Exact system prompt"}], "messages": [{"role": "user", "content": [{"kind": "text", "text": text}]}]})
        cx.write(directory / "ready.json", {"request_sha256": cx.sha(directory / "request.json")})
        return directory

    def read_all(self, request_id=None):
        content=[];cursor=0
        while True:
            result=bridge.next_request(wait_seconds=0, request_id=request_id, cursor=cursor)
            metadata=json.loads(result["content"][0]["text"])
            content.extend(result["content"][2:])
            if metadata["next_cursor"] is None:
                return metadata["request_id"],content
            request_id=metadata["request_id"];cursor=metadata["next_cursor"]

    def test_complete_text_is_preserved_across_pages_and_submission_is_bound(self):
        body="module detailed() {\n" + "// geometry and context\n" * 1700 + "}\n"
        directory=self.packet(text=body)
        with self.assertRaisesRegex(ValueError,"every request page"):
            bridge.submit_artifact("request-one",artifact_text="cube(1);")
        request_id,parts=self.read_all()
        joined="".join(p["text"] for p in parts)
        self.assertIn(body,joined)
        self.assertIn("Exact system prompt",joined)
        self.assertGreater(json.loads(bridge.next_request(0,request_id,0)["content"][0]["text"])["page_count"],1)
        artifact='```openscad\nmodule detailed() { cube([10,20,30]); }\n```'
        bridge.submit_artifact(request_id,artifact_text=artifact)
        saved=cx.read(directory/"submission.json")
        self.assertEqual(saved["artifact_text"],artifact)
        self.assertEqual(saved["request_sha256"],cx.sha(directory/"request.json"))
        self.assertEqual(json.loads(bridge.submit_artifact(request_id,artifact_text=artifact)["content"][0]["text"])["status"],"already_submitted")
        with self.assertRaisesRegex(ValueError,"already submitted"):
            bridge.submit_artifact(request_id,artifact_text="other")

    def test_images_are_mcp_image_blocks_in_original_order(self):
        directory=self.packet()
        req=cx.read(directory/'request.json')
        req['messages'][0]['content']=[{'kind':'text','text':'before'}, {'kind':'image','mimeType':'image/png','data':base64.b64encode(b'first').decode()}, {'kind':'text','text':'between'}, {'kind':'image','mimeType':'image/png','data':base64.b64encode(b'second').decode()}]
        cx.write(directory/'request.json',req);cx.write(directory/'ready.json',{'request_sha256':cx.sha(directory/'request.json')})
        _,pieces=self.read_all()
        images=[base64.b64decode(x['data']) for x in pieces if x['type']=='image']
        self.assertEqual(images,[b'first',b'second'])  # undecodable fixture bytes pass through unchanged
        self.assertEqual([x.get('text', 'IMAGE') for x in pieces][-4:],['before','IMAGE','between','IMAGE'])

    def test_delivered_images_are_original_pngs_and_jpeg_stays_diagnostic(self):
        import io
        from PIL import Image
        def png(color, mode='RGB'):
            buffer=io.BytesIO(); Image.new(mode,(640,480),color).save(buffer,format='PNG'); return buffer.getvalue()
        originals=[png((200,30,30)), png((30,60,220,255),'RGBA')]
        directory=self.packet()
        req=cx.read(directory/'request.json')
        req['messages'][0]['content']=[{'kind':'text','text':'views'}]+[{'kind':'image','mimeType':'image/png','data':base64.b64encode(x).decode()} for x in originals]
        cx.write(directory/'request.json',req);cx.write(directory/'ready.json',{'request_sha256':cx.sha(directory/'request.json')})
        _,pieces=self.read_all()
        delivered=[x for x in pieces if x['type']=='image']
        self.assertEqual([x['mimeType'] for x in delivered],['image/png','image/png'])  # production: original bytes
        self.assertEqual([base64.b64decode(x['data']) for x in delivered],originals)
        with patch.object(bridge,'IMAGE_TRANSPORT_FORMAT','JPEG'):  # diagnostic path only; invisible on the current route
            _,pieces=self.read_all()
        delivered=[x for x in pieces if x['type']=='image']
        self.assertEqual([x['mimeType'] for x in delivered],['image/jpeg','image/jpeg'])
        for original,block,expected in zip(originals,delivered,[(200,30,30),(30,60,220)]):
            image=Image.open(io.BytesIO(base64.b64decode(block['data'])))
            self.assertEqual((image.format,image.size),('JPEG',(640,480)))
            self.assertTrue(all(abs(a-b)<=3 for a,b in zip(image.getpixel((320,240)),expected)))
        self.assertEqual(cx.read(directory/'request.json')['messages'][0]['content'][1]['mimeType'],'image/png')  # evidence keeps the PNG
        buffer=io.BytesIO()
        Image.effect_noise((512,512),80).convert('RGB').save(buffer,format='PNG')
        reference=buffer.getvalue()
        with patch.object(bridge,'IMAGE_TRANSPORT_FORMAT','JPEG'):
            shrunk=bridge.transport_image({'data':base64.b64encode(reference).decode(),'mimeType':'image/png'})
        self.assertLess(len(base64.b64decode(shrunk['data'])),len(reference)//3)

    def part_request(self, text_chars=48000, images=8):
        """A part request shaped like the real ones: long prompt, then labelled build views."""
        directory=self.packet(text="P"*text_chars)
        req=cx.read(directory/'request.json')
        req['system']=[{'text':'S'*8000}]
        labels=[f'build view {i}:' for i in range(images)]
        for i,label in enumerate(labels):
            req['messages'][0]['content']+=[{'kind':'text','text':label},{'kind':'image','mimeType':'image/png','data':base64.b64encode(f'image-{i}'.encode()).decode()}]
        cx.write(directory/'request.json',req);cx.write(directory/'ready.json',{'request_sha256':cx.sha(directory/'request.json')})
        return directory,labels

    def test_part_request_packs_into_few_bounded_pages_in_original_order(self):
        _,labels=self.part_request()
        pages=bridge.pages_for(cx.read(self.run/'.cx/calls/request-one/request.json'))
        self.assertEqual(len(pages),3)
        for page in pages:
            self.assertLessEqual(sum(len(x.get('text','')) for x in page),bridge.PAGE_CHARS)
            self.assertLessEqual(sum(x['type']=='image' for x in page),bridge.PAGE_IMAGES)
        self.assertEqual(sum(x['type']=='image' for x in pages[-1]),8)
        _,pieces=self.read_all()
        joined="".join(p.get('text','') for p in pieces)
        self.assertEqual(joined,"PIPELINE SYSTEM INSTRUCTIONS (task content):\n"+"S"*8000+"PIPELINE MESSAGE ROLE: user"+"P"*48000+"".join(labels))
        order=[('image',base64.b64decode(p['data']).decode()) if p['type']=='image' else ('text',p['text']) for p in pieces if p['type']=='image' or p['text'] in labels]
        self.assertEqual(order,[x for i in range(8) for x in (('text',labels[i]),('image',f'image-{i}'))])

    def test_each_page_needs_its_own_tool_call(self):
        self.part_request()
        with patch.object(bridge,'MIN_PAGE_SECONDS',0.4):
            first=json.loads(bridge.next_request(0)['content'][0]['text'])
            with self.assertRaisesRegex(ValueError,'one page per tool call'):
                bridge.next_request(0,first['request_id'],1)
            self.assertEqual(cx.read(self.run/'.cx/calls/request-one/delivery.json')['pages_served'],[0])
            with self.assertRaisesRegex(ValueError,'every request page'):
                bridge.submit_artifact(first['request_id'],artifact_text='cube(1);')
            time.sleep(0.45)
            self.assertEqual(json.loads(bridge.next_request(0,first['request_id'],1)['content'][0]['text'])['cursor'],1)
            with self.assertRaisesRegex(ValueError,'one page per tool call'):
                bridge.next_request(0,first['request_id'],2)
            time.sleep(0.45)
            self.assertIsNone(json.loads(bridge.next_request(0,first['request_id'],2)['content'][0]['text'])['next_cursor'])
            self.assertEqual(json.loads(bridge.next_request(0,first['request_id'],0)['content'][0]['text'])['cursor'],0)  # re-reading is allowed
            bridge.submit_artifact(first['request_id'],artifact_text='cube(1);')

    def test_stale_out_of_order_changed_and_expired_requests_are_refused(self):
        first=self.packet();self.packet('request-two',order=2)
        with self.assertRaisesRegex(ValueError,"oldest"):
            bridge.next_request(0,'request-two')
        self.read_all()
        original=(first/'request.json').read_bytes()
        (first/'request.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,"changed"):
            bridge.submit_artifact('request-one',artifact_text='x')
        (first/'request.json').write_bytes(original)
        cx.write(self.run/'.cx/worker.json',{'id':'new-attempt'})
        with self.assertRaisesRegex(ValueError,"Stale"):
            bridge.submit_artifact('request-one',artifact_text='x')
        cx.write(self.run/'.cx/worker.json',{'id':'attempt-one'})
        req=cx.read(first/'request.json');req['deadline_at']=0
        cx.write(first/'request.json',req);cx.write(first/'ready.json',{'request_sha256':cx.sha(first/'request.json')})
        with self.assertRaisesRegex(ValueError,"deadline"):
            bridge.submit_artifact('request-one',artifact_text='x')

    def test_reference_image_submission_preserves_png_bytes(self):
        directory=self.packet(kind='reference_image')
        self.read_all()
        png=self.runs.parent/'generated.png';png.write_bytes(b'\x89PNG\r\n\x1a\npng-fixture')
        with self.assertRaisesRegex(ValueError,'saved PNG'):
            bridge.submit_artifact('request-one',artifact_text='pretend image')
        bridge.submit_artifact('request-one',image_path=str(png))
        self.assertEqual((directory/'submitted.png').read_bytes(),png.read_bytes())

    def test_stopped_worker_cannot_accept_an_artifact(self):
        self.packet();self.read_all()
        with patch.object(cx,'is_running',return_value=False):
            with self.assertRaisesRegex(ValueError,'stopped'):
                bridge.submit_artifact('request-one',artifact_text='x')

    def test_stdio_mcp_negotiation_advertises_only_the_two_controlled_tools(self):
        messages=[{'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2024-11-05'}},
                  {'jsonrpc':'2.0','method':'notifications/initialized'}, {'jsonrpc':'2.0','id':2,'method':'tools/list'}]
        result=subprocess.run([sys.executable,str(ROOT/'mcp/server.py')],input='\n'.join(json.dumps(x) for x in messages)+'\n',capture_output=True,text=True,check=True)
        replies=[json.loads(x) for x in result.stdout.splitlines()]
        self.assertEqual(len(replies),2)
        self.assertEqual([x['name'] for x in replies[1]['result']['tools']],['next_request','submit_artifact'])
        self.assertIn('this session',replies[0]['result']['instructions'])


if __name__=='__main__':unittest.main()
