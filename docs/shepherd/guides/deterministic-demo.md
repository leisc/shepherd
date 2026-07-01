# Deterministic demo

> Page status: release-ready
> Source state: checked-example
> Applies to: Shepherd 0.1
> Owner: @docs-system-owner (TBD)
> Validation: pytest docs_src/shepherd/quickstart/ docs_src/shepherd/tutorials/

*How-to guide. New to Shepherd? Start with the tutorial. For exact APIs, see the reference.*

**Job.** Run a Shepherd example with no credentials and no network, and get
**byte-identical output every time**, the deterministic offline provider that
the docs and CI run against.

**Prerequisites.** The quickstart or tutorial environment. No API key, no
account, nothing billed.

## Steps

1. **Use the offline provider, it is the default for every documented
   example.** Calls are answered from a recorded transcript, so the run is
   deterministic and offline. The quickstart program is the smallest case:

    ```python
    --8<-- "quickstart/hello.py:hello"
    ```

    The workspace pins `claude("sonnet-4-5")`, but against the offline provider
    the answer is replayed, not generated, no credential is read and no request
    leaves the machine.

2. **Run it twice.**

    ```bash
    python hello.py
    python hello.py
    ```

    **Expected output (both runs, identical)**

    ```text
    - Shepherd turns typed Python functions into model-backed tasks.
    - The docstring is the instruction; the return type is the contract.
    - Runs are recorded, so behavior is debuggable after the fact.
    ```

## Expected result

The two runs print the same three bullets, character for character, the same
output CI asserts in `docs_src/shepherd/quickstart/test_hello.py`. Determinism is the
point: a page that drifts from its code is caught by that test, not by a reader.

## If it fails

- **Output differs between runs?** You are not on the offline provider. The
  documented examples select it by default; check you did not swap in a live
  model.
- **`shp.DeliveryFailed`?** On this example, against the offline provider, that
  signals a broken install, reinstall and rerun.
- **`RuntimeError` about a workspace?** The task was called outside
  `with shp.workspace(...)`; see [Debug your first run](debug-your-first-run.md).
