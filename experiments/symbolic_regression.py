import torch
from matplotlib import pyplot as plt
from ac_grammar_vae.model.gvae.interpreter import TorchEquationInterpreter


def plot_expression(expression, interpreter, domain=(-5.0, 5.0), steps=250):

    plt.figure()
    xx = torch.linspace(*domain, steps=steps)
    yy = interpreter.evaluate(expression, x=xx)
    plt.plot(xx.numpy(), yy.numpy())
    plt.title("".join(expression))


def sanity_check_experiment():

    noise_level = 1e-3
    n_data = 20
    n_experiment_runs = 10
    n_opt_steps = 250
    true_expr = "(X * X) + 1"
    X = torch.linspace(-5, 5, steps=n_data)
    Y = torch.square(X) + 1.0 + torch.randn(n_data) * noise_level

    interpreter = TorchEquationInterpreter()

    # load the model from file
    model = torch.load("results/gvae_pretrained.pth")

    for i in range(n_experiment_runs):

        model.eval()
        expr = model.find_expression_for(X=X, Y=Y, num_opt_steps=n_opt_steps)

        plot_expression(expr, interpreter)
        plt.savefig(f"results/solution_{ i }.pdf")

        with open(f"results/solution.json") as f:
            f.writelines([
                f"True Expression= {true_expr}",
                f"Solution Found= { ''.join(expr) }",
                f"RMSE= { torch.sqrt(torch.mean(torch.square(interpreter.evaluate(expr, X) - Y))) }"
            ])


def main():
    sanity_check_experiment()


if __name__ == "__main__":
    main()
