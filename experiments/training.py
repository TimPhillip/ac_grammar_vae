import matplotlib.pyplot as plt
import torch
import tqdm
import logging
from torch.utils.data import DataLoader

from ac_grammar_vae.data import CFGEquationDataset
from ac_grammar_vae.data.transforms import MathTokenEmbedding, ToTensor, Compose, PadSequencesToSameLength, GrammarParseTreeEmbedding
from ac_grammar_vae.model.gvae import GrammarVariationalAutoencoder


def setup_dataset(n_samples, n_validation_samples=0, expressions_with_parameters=False):

    training = CFGEquationDataset(n_samples=n_samples, use_original_grammar=not expressions_with_parameters)
    validation = CFGEquationDataset(n_samples=n_validation_samples, use_original_grammar=not expressions_with_parameters) if n_validation_samples > 0 else None

    embedding = GrammarParseTreeEmbedding(training.pcfg, pad_to_length=training.max_grammar_productions)
    training.transform = Compose([
            embedding,
            ToTensor(dtype=torch.int64)
        ])

    if validation:
        validation.transform = Compose([
            embedding,
            ToTensor(dtype=torch.int64)
        ])

    if not validation:
        return training, embedding
    else:
        return training, validation, embedding


def main():

    # Hyperparameters
    num_epochs = 20
    early_stopping_patience = 3
    expression_with_parameters = True
    export_file = "results/gvae_pretrained.pth" if not expression_with_parameters else "results/gvae_pretrained_parametric.pth"
    batch_size = 600
    val_batch_size = 2048

    # create the dataset
    training, validation, embedding = setup_dataset(n_samples=10**5, n_validation_samples=10**4, expressions_with_parameters=expression_with_parameters)
    training_loader = DataLoader(dataset=training,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 collate_fn=PadSequencesToSameLength())

    validation_loader = DataLoader(dataset=validation,
                                   batch_size=val_batch_size,
                                   shuffle=False,
                                   collate_fn=PadSequencesToSameLength())

    # build the model
    model = GrammarVariationalAutoencoder(
        len(training.pcfg.productions()) + 1,
        training.max_grammar_productions,
        latent_dim=16,
        rule_embedding=embedding,
        expressions_with_parameters=expression_with_parameters
    )

    optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())
    learning_rate_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs//2, eta_min=1e-8)


    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Training Grammar Variational Autoencoder (GVAE) for { num_epochs } epochs.")


    no_improvement_for_epochs = 0
    best_validation_loss = torch.inf

    for epoch in range(num_epochs):

        # Start of a new epoch
        epoch_steps = tqdm.tqdm(training_loader, desc=f"Epoch {epoch}: ", unit=" Batches")

        model.train()
        for X in epoch_steps:

            # compute the loss
            loss = model.negative_elbo(X)

            # optimize
            loss.backward()
            optimizer.step()

            epoch_steps.set_postfix({
                'loss': loss.detach().item(),
                'lr': learning_rate_schedule.get_lr()[0]
            })

        # update the learning rate according to the scheduler
        learning_rate_schedule.step()

        # run the validation
        with torch.no_grad():
            model.eval()

            validation_loss = 0
            for X in validation_loader:
                val_loss = model.negative_elbo(X)
                validation_loss += val_loss

            validation_loss = validation_loss / len(validation_loader)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                no_improvement_for_epochs = 0

                logging.info(f"Validation {epoch}: loss={validation_loss}")
            else:
                no_improvement_for_epochs += 1
                logging.info(f"Validation {epoch}: loss={validation_loss} (no improvement for { no_improvement_for_epochs } epochs)")

        # early stopping
        if no_improvement_for_epochs > early_stopping_patience:
            return

        # save the weights to file
        torch.save(model, export_file)



if __name__ == "__main__":
    main()