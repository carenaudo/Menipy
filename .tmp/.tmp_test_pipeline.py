from src.menipy.pipelines.base import PipelineBase


class TestPipeline(PipelineBase):
    name = "test"

    def do_acquisition(self, ctx):
        return ctx

    def do_preprocessing(self, ctx):
        return ctx


p = TestPipeline()
ctx = p.run()
print("Done run")
