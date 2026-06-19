import nox
import glob


@nox.session(python=["3.13.7"])
def test(session):
    if not isinstance(session.virtualenv, nox.virtualenv.PassthroughEnv):
        session.install("-e", ".[dev,all]")
    session.run("pytest", "-v", "-s", "--cov", "--cov-report", "term-missing")


@nox.session(python=["3.13.7"])
def lint(session):
    if not isinstance(session.virtualenv, nox.virtualenv.PassthroughEnv):
        session.install("ruff")
    session.run("ruff", "check")


@nox.session(python=["3.13.7"])
def format(session):
    if not isinstance(session.virtualenv, nox.virtualenv.PassthroughEnv):
        session.install("ruff")
    session.run("ruff", "format")


@nox.session(python=["3.13.7"])
def typecheck(session):
    if not isinstance(session.virtualenv, nox.virtualenv.PassthroughEnv):
        session.install("-e", ".[dev,all]")
    session.run("python", "--version")
    session.run("pyrefly", "--version")
    session.run("pyrefly", "check")


@nox.session(python=["3.13.7"])
def ltf(session):
    if not isinstance(session.virtualenv, nox.virtualenv.PassthroughEnv):
        session.install("-e", ".[dev,all]")
    session.run("ruff", "format")
    session.run("ruff", "check")
    session.run("pyrefly", "check")


@nox.session(python=["3.13.7"])
def build_docs(session):
    if not isinstance(session.virtualenv, nox.virtualenv.PassthroughEnv):
        session.install("-e", ".[dev,ray]")
    session.run("rm", "-rf", "docs/build")
    session.run("rm", "-rf", "docs/source/generated")
    for rst_file in glob.glob("docs/source/*.rst"):
        session.run("sphinx-autogen", "-o", "docs/source/generated", rst_file)
    session.run("sphinx-build", "-M", "html", "docs/source", "docs/build", "--fail-on-warning")
