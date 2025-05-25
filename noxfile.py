import nox


@nox.session(
    python=["3.8", "3.9", "3.10", "3.11", "3.12", "3.13"],
)
def tests(session: nox.Session) -> None:
    """Build the package."""
    session.install("cmake>=3.18", "setuptools", "cmake", "wheel")  # Build tools first
    session.install("pytest",)
    session.install(".")
    # session.run("pip", "install", ".", "--use-pep517")
    # Run all utils tests
    session.run(
        "pytest",
        "tests/test_utils/",
        "tests/test_simobjects/",
        "-v",          # Show individual test names
        "--tb=long",   # Full error tracebacks
        "--color=yes", # Colorized output
    )
    