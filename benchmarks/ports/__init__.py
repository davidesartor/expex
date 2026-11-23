"""One file per port. Each defines:

    NAME       the port's name (examples/<NAME>/)
    ORIGINALS  the entries of originals.toml it needs
    CLAIM      what README.md says about its speed, {measure name: claim}, for the report
    CONFIGS    {config name: dict}; "quick" is the smoke-test config
    DEFAULTS   {"quick" | "cpu" | "gpu": [config names]} run when no --config is given
    setup(cfg) -> list[harness.Measure]: the original's and the port's callables, on the same inputs
"""
