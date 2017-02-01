EVA
===========
An AI research project which explores audio synthesis.

Currently, PixelCNN is implemented with Keras, WaveNet is on the horizon.

Things to check:
- [ ] Channel masking is probably retarded beyond all recognition.
- [ ] - Switch to sparse.
- [ ] - Why dafuq the output has shape of `[channels, 1, pixels]` wtf is this 1 doing? I never found a reference to this.
- [ ] - Don't pass random to inference.
