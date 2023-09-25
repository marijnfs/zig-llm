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
- sampling is still MAP sampling, temp sampling needs to be implemented.
- there is still leaking memory in various parts.
- the model.bin file is very shaky, a better model binarization is on the roadmap

Build
=====
There are two branches:

# Stable
`git checkout stable`

Build it using zig version 0.11.0. If this is your default installed zig compiler, you can use `make` or `make release` which are shortcuts to the appropriate build commands (to `zig build` or `zig build -Doptimize=ReleaseSafe`). Otherwise, have a look at `Makefile` and adapt as needed.

# Master
This is the development branch, which will (mostly) track the latest zig and Mach versions. This is inherently more unstable, but will have the latest features.

Use
===
Provide a model and tokenizer file. The llama2 repo tokenizer from karpathy is added in this repository. One of the matching models can be grabbed using:
`wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin`

or 

`wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin`

Then you can run inference by providing the models (currently full paths are needed) using:

`./zig-out/bin/llm --model /full/path/to/stories110M.bin --tokenizer /full/path/to/models/tokenizer.bin --prompt "Once upon a time there was a donkey"`

Example output:

`Tokenized:
Once- upon- a- time- there- was- a- don-key-
Prediction:
Once upon a time there was a donkey called Daisy. Daisy was very happy and loved to play in the fields. One day, Daisy was walking in the fields when she saw a big, juicy carrot. She was so excited that she started to eat it right away.
Suddenly, Daisy heard a loud noise. She looked up and saw a big, scary wolf. The wolf was very angry and he wanted to eat Daisy. Daisy was so scared that she started to run away.
The wolf chased Daisy and she ran as fast as she could. She ran and ran until she reached a big, tall tree. Daisy climbed up the tree and the wolf couldn't reach her.
The wolf was so angry that he started to bark and howl. Daisy was so scared that she stayed in the tree until the wolf went away.
When the wolf was gone, Daisy climbed down from the tree and ran back home. She was so happy to be safe and she never forgot the big, scary wolf.
<s>`

If you run out of memory, you can limit the token response size with the `--length` parameter.
