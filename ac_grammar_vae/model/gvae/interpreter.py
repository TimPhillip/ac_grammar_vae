import torch


class TorchEquationInterpreter:

    def __init__(self):
        pass

        self._semantics = {
            'sin' : 'torch.sin',
            'cos' : 'torch.cos',
            'exp' : 'torch.exp',
            '1' : 'torch.as_tensor(1, dtype=torch.float)',
            '2': 'torch.as_tensor(2, dtype=torch.float)',
            '3': 'torch.as_tensor(3, dtype=torch.float)',

        }

    def evaluate(self, expression_str, x):

        def replace_with_semantics(symbol):
            return self._semantics[symbol] if symbol in self._semantics else symbol

        expression_str = "".join(map(replace_with_semantics, expression_str))
        loc = {}
        exec("import torch\ny = " + expression_str,{'x': x}, loc)
        y = loc['y']

        # repeat the result if only constants are involved
        if y.shape != x.shape:
            y = y.repeat(x.shape[-1])

        return y