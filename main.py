import os

import click

from mcmc import run, set_seed

# Set the default path to the current directory
PATH = os.path.dirname(os.path.realpath(__file__)) + "/"


@click.command()
@click.option("--seed", default=42, help="Set the random seed.")
@click.option(
    "--language",
    type=click.Choice(["hebrew", "english"]),
    required=True,
    help="Language of the message (hebrew or english).",
)
@click.option(
    "--text_file",
    default="war-and-peace.txt",
    required=True,
    help="Language of the message (hebrew or english).",
)
@click.option("--message", required=True, help="The message to decrypt.")
@click.option(
    "--plot/--no-plot", default=False, help="Enable/disable plotting the results."
)
@click.option(
    "--iterations", default=10000, help="Number of iterations for the MCMC algorithm."
)
def main(seed, language, text_file, message, plot, iterations):
    set_seed(seed)
    run(PATH, text_file, language, message, plot=plot, iterations=iterations)


if __name__ == "__main__":
    main()
