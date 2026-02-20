import nox
import glob


@nox.session
def test(session):
    session.install(".[dev]")
    session.run("pytest", "-v", "--cov", "--cov-report", "term-missing")


@nox.session
def lint(session):
    session.install("ruff")
    session.run("ruff", "check")


@nox.session
def format(session):
    session.install("ruff")
    session.run("ruff", "format")


@nox.session
def ltf(session):
    session.install(".[dev]")
    session.run("ruff", "format")
    session.run("ruff", "check")
    session.run("pyrefly", "check")


@nox.session
def typecheck(session):
    session.install(".[dev]")
    session.run("pyrefly", "check")


@nox.session
def build_docs(session):
    session.install(".[dev, ray]")
    session.run("rm", "-rf", "docs/build")
    session.run("rm", "-rf", "docs/source/generated")
    for rst_file in glob.glob("docs/source/*.rst"):
        session.run("sphinx-autogen", "-o", "docs/source/generated", rst_file)
    session.run("sphinx-build", "-M", "html", "docs/source", "docs/build", "--fail-on-warning")
