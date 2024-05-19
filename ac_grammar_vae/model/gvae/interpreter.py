import torch
import copy
import math


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


def real_complexity(r):
    eps= 10**(-9)
    x = r / eps
    return math.log2(1 + x**2)


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

        self._expr = copy.copy(expr)
        param_indices = [i for i, val in enumerate(expr) if val == "theta"]
        self.num_params = len(param_indices)
        self._params = torch.nn.Parameter(data=torch.randn(self.num_params), requires_grad=True)

        def replace_with_semantics(symbol):
            return self._semantics[symbol] if symbol in self._semantics else symbol

        for i, idx in enumerate(param_indices):
            expr[idx] = f"theta[{ i }]"

        self._expression_str = "".join(map(replace_with_semantics, expr))

    def complexity(self):
        return self.parameter_complexity() + self.structure_complexity()

    def parameter_complexity(self):
        comp = 0
        for param in self._params:
            comp += real_complexity(param.item())
        return comp

    def structure_complexity(self):
        num_ops = 0
        num_unique_ops = 0

        # count the different operations
        for sym in ["+", "-", "/", "*", "**", "sin", "cos", "exp", "log", "sqrt", "x"]:
            sym_count = len(list(filter(lambda s: s == sym, self._expr)))
            if sym_count > 0:
                num_ops += sym_count
                num_unique_ops += 1

        return num_ops * math.log2(num_unique_ops)

    def optimize_parameters(self, X, Y, num_opt_steps=100):

        if self.num_params == 0:
            return

        #optim = torch.optim.Adam(lr=1e-2, params=self.parameters())
        optim = torch.optim.LBFGS(lr=1e-2, params=self.parameters())

        def closure():
            optim.zero_grad()
            Y_pred = eval(self._expression_str, {'torch': torch, 'x': X, 'theta': self._params})
            rmse = torch.sqrt(torch.mean(torch.square(Y_pred - Y)))
            rmse.backward()
            return rmse.item()

        for _ in range(num_opt_steps):
            optim.step(closure=closure)

    def __call__(self, X):
        with torch.no_grad():
            y = eval(self._expression_str, {'torch': torch, 'x': X, 'theta': self._params})

            # repeat the result if only constants are involved
            if y.shape != X.shape:
                y = y.repeat(X.shape[-1])

            return y

    def __str__(self):

        param_indices = [i for i, val in enumerate(self._expr) if val == "theta"]
        readable_expression = copy.copy(self._expr)

        for i, idx in enumerate(param_indices):
            readable_expression[idx] = f"{ self._params[i].item() : .2f}"

        return "".join(readable_expression)


if __name__ == "__main__":

    expr = ExpressionWithParameters([
        'theta', '*', 'x',  '*', 'x', '+', 'exp', '(', 'theta', ')'
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
    print(expr)

