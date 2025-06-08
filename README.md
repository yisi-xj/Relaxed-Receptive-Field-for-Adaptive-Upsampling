# RRFU: Relaxed Receptive Field for Adaptive Upsampling

RRFU is a lightweight plug-and-play upsampling operator with task-agnostic applicability. RRFU establishes the feasibility of adaptive receptive fields for region-sensitive and detail-critical dense prediction tasks. RRFU achieves state-of-the-art performance while maintaining exceptional efficiency, imposing negligible computational overhead on real-time inference without requiring custom CUDA implementations. Crucially, as a single-feature upsampler, it attains high-frequency contour accuracy comparable to methods leveraging high-resolution guidance features, demonstrating that adaptive receptive fields hold significant promise in current upsampling applications.

## Highlights

- **Easy to use:** RRFU does not rely on any extra CUDA packages installed;
- **Simplicity:**: RRFU's code is concise and easy to modify;
- **precise:** RRFU can efficiently sample both internal objects and contours.

