# Koharu Project Rules

Document only durable, repository-specific constraints here. Do not record current file layouts, temporary paths, model inventories, helper names, or other implementation details that may change during a refactor. Normal Rust, TypeScript, testing, formatting, and Git practices are assumed.

## Change Policy

- Never add backward compatibility. When an API, schema, configuration, or ownership boundary changes, update every in-repository consumer and remove the replaced form.
- Prefer a coherent ownership redesign over aliases, forwarding layers, compatibility parsers, or cosmetic renaming.
- Keep responsibilities self-contained. Defaults and provider-specific behavior belong to the component that owns them rather than a central list of special cases.
- Remove dead abstractions and one-use helpers when direct code is clearer.
- Make the smallest change that solves the problem. Do not edit unrelated configuration, documentation, generated files, or workflows in the same change. A feature does not land beside drive-by cleanup.
- If the current owner has moved, rebase onto that owner. Do not port a patch onto a deleted tree.
- When two changes implement the same workflow, keep one and extend the existing command or configuration. Do not land both.
- Translation credentials come from the environment variables `koharu-server --help` documents. Never persist credentials in the repository.
- Once a subsystem is removed, delete its calls and dependency together. Do not leave a stub, alias, or disabled flag behind.

## Ownership

- Scene owns what a project means. Storage owns opaque payloads and immutable blobs. Analysis, content, and presentation stay separate: detection geometry is not a visible layer, translation does not drop OCR provenance, and typography does not rewrite semantic text.
- A patch binds to a project and a base revision, records preconditions, and carries inverses. Reject a stale patch. Derived work rebases explicitly and fails if anything it observed has changed.
- Publish missing blobs first, write the inactive state slot, then publish atomically. Opening uses the newest valid slot and can fall back. Renderer and pipeline consume snapshots and submit patches. They do not write storage files.
- A pipeline stage runs on one page: detection, then OCR and translation, or detection, then inpainting. Upscaling needs no earlier stage. Pages enter in project order. Completed detection makes that page's later stages eligible immediately. At most one job per stage runs at a time. Each committed result is its own undo step. Do not add a second translation prompt path.
- Accelerator inference uses one admission lane, including remote translation. On out-of-memory, release the lane, unload the other models, wait for a fresh resource sample, and retry once. Any other failure unloads the failing stage. Stopping is cooperative and discards an uncommitted result.
- One prepared frame feeds every rendered output. Translation is the only visible text. Incremental reuse must match a full render.

## Source Boundaries

- Keep safe public APIs separate from unsafe FFI, dynamic loading, and build integration.
- Do not hand-edit generated or derived source. Change its authoritative input and run the generator.
- Do not commit credentials, model weights, datasets, generated outputs, or machine-specific artifacts.
- Native llama.cpp and diffusion runtimes come from prebuilt packages. Do not compile those CUDA trees inside the app build. Release and publish jobs do a fresh build.
- Hardware discovery stays separate from managed runtime package activation. Do not load a cached accelerator runtime during the probe.

## ML Architecture

- Keep a consistent public lifecycle across models while allowing model-specific inputs and outputs.
- Separate network ownership and weight loading from preprocessing, postprocessing, slicing, and public result types.
- Avoid pass-through types and layers that do not own a real responsibility.
- Accept a device abstraction at the model boundary, convert it once, and avoid unnecessary transfers or synchronization.
- Use the established runtime and variable-store loading paths unless they are proven insufficient.
- Disable gradient tracking during inference.
- Create generated token IDs and index tensors on the model device. A CPU tensor used against accelerator embeddings is a defect, not a later optimization.
- Invalid model JSON fails closed. Do not add a silent repair layer that can change meaning.
- LLM providers enforce the translation output schema through their native structured-output API. A failing engine fails immediately, and translation moves on to the next engine in the requested order.
- Translation instructions are optional guidance. Engines that cannot follow them ignore them.

## Upstream Alignment

- Keep ports structurally traceable to a commit-pinned authoritative implementation.
- Preserve checkpoint-affecting names, construction order, parameter paths, tensor layouts, execution order, and postprocessing semantics.
- Treat missing or unexpected weights as an architecture or parameter-name mismatch before changing the loader.
- Explain intentional divergences next to the affected code.
- Compare ports on identical inputs using structured outputs such as shapes, ranges, boxes, scores, masks, and ordering.

## Performance

- Optimize and benchmark the actual target device with representative inputs.
- Remove redundant transfers, synchronization, allocations, and per-pixel host loops before adding concurrency or caching.
- Account for asynchronous accelerator execution when timing work.
- Load assets and warm models outside measured regions.
- Report the device, input size, baseline, result, and correctness difference.
- For manga bubble detection and inpainting, consult the
  [hard-page evaluation repository](https://github.com/perennialtech/koharu-hard-page-eval)
  for difficult test pages, model comparisons, and reproduction steps.

## Verification

- Before declaring a change ready, run `./scripts/check.sh`. CI runs the same command.
- Optimize for fast development and iteration. By default, run the smallest relevant check or focused test once using the debug profile.
- Do not run full test suites, repeatedly rerun unchanged tests or builds, or build and test profiles other than debug unless the user explicitly requests it.
- Run end-to-end tests only when the user explicitly asks for them.
- Add a test when it protects a real regression boundary. Do not lock a translated phrase, a language display name, a registration string, or a one-line behavior the owner can read directly. Do not add a second harness for behavior the crate's existing tests already cover. Do not assert a compatibility shim.
- A model port needs a structured comparison on identical inputs. A green build is not that comparison.
- Keep the pull-request text short: problem, change, and verification.

## Documentation

- Comments should explain ownership, invariants, upstream mapping, or deliberate divergence; do not narrate straightforward code.
- Keep this file focused on long-lived decision rules rather than the current implementation.
