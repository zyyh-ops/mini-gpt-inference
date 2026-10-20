"""Import smoke checks for the public package entry points."""
import unittest

class TestPackageImports(unittest.TestCase):
    def test_core_modules_importable(self):
        from minivllm.mini_runtime import MiniRuntime
        from minivllm.scheduler import Scheduler
        from minivllm.model import MiniGPT
        self.assertTrue(callable(MiniRuntime))
        self.assertTrue(callable(Scheduler))
        self.assertTrue(callable(MiniGPT))

    def test_experiment_modules_importable(self):
        from experiments.bench_preemption import make_runtime
        from experiments.trace_preemption import run_trace
        self.assertTrue(callable(make_runtime))
        self.assertTrue(callable(run_trace))

    def test_legacy_minigpt_still_importable(self):
        from minigpt.model import MiniGPT
        self.assertTrue(callable(MiniGPT))
