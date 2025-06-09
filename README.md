# FRFU: Flexible Receptive Field for Upsampling

FRFU is a lightweight plug-and-play upsampling operator with task-agnostic applicability. FRFU establishes the feasibility of adaptive receptive fields for region-sensitive and detail-critical dense prediction tasks. FRFU achieves state-of-the-art performance while maintaining exceptional efficiency, imposing negligible computational overhead on real-time inference without requiring custom CUDA implementations. Crucially, as a single-feature upsampler, it attains high-frequency contour accuracy comparable to methods leveraging high-resolution guidance features, demonstrating that adaptive receptive fields hold significant promise in current upsampling applications.

:warning:Note: In the experimental logs and weight files of the project, the old name "rrfu" is used.

## Highlights

- **Simple code:** FRFU's code is concise and easy to modify;
- **Easy to use:** FRFU does not rely on any extra CUDA packages installed;
- **precise:** FRFU can efficiently sample both internal objects and contours.

