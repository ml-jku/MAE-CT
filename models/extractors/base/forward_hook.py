class ForwardHook:
    def __init__(self, outputs: dict, output_name: str, allow_multiple_outputs: bool = False):
        self.outputs = outputs
        self.output_name = output_name
        self.allow_multiple_outputs = allow_multiple_outputs
        self.enabled = True

    def __call__(self, _, __, output):
        if not self.enabled:
            return

        if self.allow_multiple_outputs:
            # e.g. contrastive heads have multiple forward passes
            output_name = f"{self.output_name}.{len(self.outputs)}"
            assert output_name not in self.outputs
            self.outputs[output_name] = output
        else:
            assert self.output_name not in self.outputs, "clear output before next forward pass to avoid memory leaks"
            self.outputs[self.output_name] = output
