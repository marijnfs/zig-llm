Zig LLM implementation with Mach backend

Author: Marijn Stollenga
License: See LICENSE file

Zig/Mach based Language model
============
This repo is an implementation of the language model used in e.g. https://github.com/karpathy/llama2.c and https://github.com/cgbur/llama2.zig, but it uses WGSL shaders with the great Mach backend, allowing it to use your gpu.

State of repo
=============
This is preliminary work that needs a bunch of work to stabilize, but it can run various LLM models.
Things that need to be implemented:
- n_kv_heads is not supported right now, we expect query heads to be same size as key/value heads.
- sampling is still MAP sampling, temp sampling needs to be implemented.
- there is still leaking memory in variuos parts.
- the model.bin file is very shaky, a better model binarization is on the roadmap

Build
=====
Build it using zig version 0.11.0. If this is your default installed zig compiler, you can use `make` or `make release` which are shortcuts to the appropriate build commands (to `zig build` or `zig build -Doptimize=ReleaseSafe`). Otherwise, have a look at `Makefile` and adapt as needed.

Use
===
Provide a model and tokenizer file. The llama2 repo tokenizer from karpathy is added in this repository. One of the matching models can be grabbed using:
`wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin`

Then you can run inference using:

`./zig-out/bin/llm --model stories15M.bin --tokenizer models/tokenizer.bin --prompt "Once upon a time there was a donkey"`

If you run out of memory, you can limit the token response size with the `--length` parameter.