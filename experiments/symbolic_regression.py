

def main():
    interpreter = TorchEquationInterpreter()

    model.eval()
    X = torch.linspace(-5, 5, steps=20)
    Y = torch.square(X) + 1.0 + torch.randn(20) * 1e-3
    expr = model.find_expression_for(X=X, Y=Y, num_opt_steps=250)

    plot_expression(expr, interpreter)