"""纯函数单元测试：领域词库、ASR 配置、transcript 校验、章节归一化。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from yiren_ingest.asr_pipeline.config import (
    FUNASR_DEFAULTS,
    WHISPER_DEFAULTS,
    get_engine_config,
    resolve_device,
)
from yiren_ingest.domain_terms import (
    GAME_DOMAIN_TERMS,
    GAME_HOTWORDS,
    TERM_CORRECTIONS,
)
from yiren_ingest.ingest_orchestrator.orchestrator import (
    _DEFAULT_CONFIG,
    _classify,
    _make_asr_config,
    _validate_transcript,
    generate_transcript_id,
    series_id_to_compact,
)
from yiren_ingest.launcher import LauncherWindow
from yiren_ingest.summarizer.chapter_summary import _normalize_chapters


class TestIds(unittest.TestCase):
    def test_compact_and_transcript_id(self):
        self.assertEqual(series_id_to_compact("zhangchulan_guide"), "zhangchulanguide")
        self.assertEqual(
            generate_transcript_id("zhangchulan_guide", 7),
            "TR_zhangchulanguide_007",
        )


class TestClassify(unittest.TestCase):
    def test_media_types(self):
        self.assertEqual(_classify("a.mp4"), "video")
        self.assertEqual(_classify("a.MKV"), "video")
        self.assertEqual(_classify("a.wav"), "audio")
        self.assertEqual(_classify("a.pdf"), "unknown")
        self.assertEqual(_classify("a.txt"), "unknown")


class TestConfig(unittest.TestCase):
    def test_game_domain_terms_wired(self):
        self.assertEqual(WHISPER_DEFAULTS["initial_prompt"], GAME_DOMAIN_TERMS)
        self.assertEqual(FUNASR_DEFAULTS["hotwords"], GAME_HOTWORDS)
        self.assertIn("身外身", GAME_HOTWORDS)
        self.assertIn("强制脱出", GAME_HOTWORDS)
        self.assertEqual(TERM_CORRECTIONS["炁"], ["气"])

    def test_none_does_not_override_engine_defaults(self):
        # 回归：model=None 曾覆盖 paraformer-zh 导致 FunASR 启动失败
        cfg = _make_asr_config({"asr_backend": "funasr"})
        self.assertNotIn("model", cfg)
        merged = get_engine_config("funasr", cfg)
        self.assertEqual(merged["model"], "paraformer-zh")

    def test_default_config_is_video_only(self):
        self.assertTrue(_DEFAULT_CONFIG["enable_summary"])
        self.assertNotIn("save_audio_samples", _DEFAULT_CONFIG)
        self.assertNotIn("doc_backend", _DEFAULT_CONFIG)

    def test_device_defaults_are_portable(self):
        self.assertEqual(_DEFAULT_CONFIG["asr_device"], "auto")
        self.assertEqual(_DEFAULT_CONFIG["asr_compute_type"], "auto")

    def test_resolve_device(self):
        self.assertEqual(resolve_device("cpu"), "cpu")
        self.assertEqual(resolve_device("cuda"), "cuda")
        self.assertIn(resolve_device("auto"), ("cpu", "cuda"))

    def test_auto_device_falls_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), "cpu")


class TestLauncherState(unittest.TestCase):
    class _DeadThread:
        def is_alive(self):
            return False

    def _launcher(self, result_status):
        launcher = LauncherWindow()
        launcher.running = True
        launcher.last_status = "running"
        launcher.pipeline_thread = self._DeadThread()
        launcher.pipeline_win = SimpleNamespace(result={"status": result_status})
        return launcher

    def test_is_done_releases_running(self):
        launcher = self._launcher("success")
        self.assertTrue(launcher.is_done())
        self.assertFalse(launcher.running)
        self.assertEqual(launcher.last_status, "complete")

    def test_is_done_marks_error(self):
        launcher = self._launcher("error")
        self.assertTrue(launcher.is_done())
        self.assertFalse(launcher.running)
        self.assertEqual(launcher.last_status, "error")


class TestTranscriptValidation(unittest.TestCase):
    def _valid(self):
        return {
            "transcript_id": "TR_x_001",
            "series_id": "x",
            "source_audio": "a.mp4",
            "language": "zh",
            "segments": [
                {"segment_id": "seg_0001", "text": "垫步", "start": 0, "end": 1}
            ],
        }

    def test_valid(self):
        self.assertEqual(_validate_transcript(self._valid()), [])

    def test_missing_series_id(self):
        data = self._valid()
        del data["series_id"]
        self.assertIn("series_id", _validate_transcript(data))

    def test_missing_segment_text(self):
        data = self._valid()
        del data["segments"][0]["text"]
        self.assertIn("segments[0].text", _validate_transcript(data))


class TestChapterNormalization(unittest.TestCase):
    def test_sort_clamp_and_ids(self):
        parsed = {
            "chapters": [
                {"title": "第二章", "summary": "后半段", "start": 60, "end": 200,
                 "key_points": ["b"]},
                {"title": "第一章", "summary": "前半段", "start": -5, "end": 60,
                 "key_points": ["a", ""]},
                {"title": "", "summary": "无标题", "start": 0, "end": 10},
                {"title": "空章", "summary": "", "start": 10, "end": 20},
            ]
        }
        chapters = _normalize_chapters(parsed, 0.0, 120.0)
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0]["chapter_id"], "ch_001")
        self.assertEqual(chapters[0]["title"], "第一章")
        self.assertEqual(chapters[0]["start"], 0.0)
        self.assertEqual(chapters[0]["key_points"], ["a"])
        self.assertEqual(chapters[1]["end"], 120.0)

    def test_invalid_chapters_returns_empty(self):
        self.assertEqual(_normalize_chapters({"chapters": "bad"}, 0, 10), [])


if __name__ == "__main__":
    unittest.main()
