"""Command-line interface."""
import typer

app = typer.Typer()


@app.command()
def main():
    """Pandas Profiling Lab."""
    print("""Project: pandas-profiling-lab.""")


if __name__ == "__main__":
    app()
