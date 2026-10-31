# Random window-order control (Freebase 1900–1909)

Keep ten RGCN42 models frozen, GraphMask initialization42, validation100042, report42; budget six joint epochs, 531 batches/window/epoch and the same 5%/10% fidelity selection. No temporal penalty.

Four new runs: fixed ascending order and independently reshuffled rounds with order seeds42/43/44. A round contains exactly one batch from every window. `window_order` uses local Python RNG keyed by order seed, epoch and step; it never consumes the global sampler/gate RNG. The common new `per_window_batch_v1` protocol keys training RNG by GraphMask seed, epoch, batch index and window index using NumPy SeedSequence. Loader initialization and each window batch are reseeded independently. The fixed-order run also uses this protocol, providing the matched order-only comparator. Earlier seed42 fixed-order results used a global sequential RNG and remain a historical comparison, not the order-only control.

The original configs omit these optional fields and keep their old global RNG and fixed order. Existing outputs are untouched. Every run has a separate lock, logs, per-window test reports, epoch-boundary recovery and configuration/code revision records. Inspect selected epoch/layers and fidelity/retention before comparing curve variation; do not tune by smoothness.
