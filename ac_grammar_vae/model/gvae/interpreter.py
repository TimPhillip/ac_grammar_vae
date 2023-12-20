import torch


class TorchEquationInterpreter:

    def __init__(self):
        pass

        self._semantics = {
            'sin' : 'torch.sin',
            'cos' : 'torch.cos',
            'exp' : 'torch.exp',
            'log': 'torch.log',
            'sqrt': 'torch.sqrt',
            '1' : 'torch.as_tensor(1, dtype=torch.float)',
            '2': 'torch.as_tensor(2, dtype=torch.float)',
            '3': 'torch.as_tensor(3, dtype=torch.float)',
            'theta': 'torch.as_tensor(3, dtype=torch.float)',
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


class ExpressionWithParameters(torch.nn.Module):

    def __init__(self, expr):
        super(ExpressionWithParameters, self).__init__()

        self._semantics = {
            'sin': 'torch.sin',
            'cos': 'torch.cos',
            'exp': 'torch.exp',
            'log': 'torch.log',
            'sqrt': 'torch.sqrt'
        }

        self._expr = expr
        param_indices = [i for i, val in enumerate(expr) if val == "theta"]
        num_params = len(param_indices)
        self._params = torch.nn.Parameter(data=torch.randn(num_params), requires_grad=True)

        def replace_with_semantics(symbol):
            return self._semantics[symbol] if symbol in self._semantics else symbol

        for i, idx in enumerate(param_indices):
            expr[idx] = f"theta[{ i }]"

        self._expression_str = "".join(map(replace_with_semantics, expr))

    def optimize_parameters(self, X, Y, num_opt_steps=2000):

        optim = torch.optim.Adam(lr=1e-2, params=self.parameters())

        for _ in range(num_opt_steps):
            optim.zero_grad()
            Y_pred = eval(''.join(self._expr), {'x': X, 'theta': self._params})
            rmse = torch.sqrt(torch.mean(torch.square(Y_pred - Y)))
            rmse.backward()
            optim.step()

    def __call__(self, X):
        with torch.no_grad():
            return eval(''.join(self._expr), {'x': X, 'theta': self._params})


if __name__ == "__main__":

    expr = ExpressionWithParameters([
        'theta', '*', 'x',  '*', 'x', '+', 'theta'
    ])

    xx = torch.linspace(-5, 5, 100)
    yy = xx * xx

    y_pred = expr(xx)
    error = torch.mean((y_pred -yy)**2)
    print(error.item())

    expr.optimize_parameters(xx,yy)

    y_pred = expr(xx)
    error = torch.mean((y_pred - yy) ** 2)
    print(error.item())

