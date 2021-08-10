# Specufex Preprocessing Development

## Testing

There is now a standard testing framework started.
Whenever you make a change to the code that you want to push, run the tests to see if any errors are generated.
For now, there is a single dataset used, but we can add more in the future for morre robust testing.
To run the tests, you must install the `pytest` library

```bash
>>> pip install pytest
```

After installation, to run the tests, simply cd to the "tests" directory and run

```bash
>>> pytest
```

If everything works, it will tell you that everything passed.
