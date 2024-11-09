# Dwarf In The Flask

![Dwarf](https://i.sstatic.net/sUyGg.jpg)

This is a flask server that holds all of the AI stuff I want to do in order to make it accessible to other applications. It was started for my Obsidian plugin RhizObsidian but I want it to be usable outside of that context too. It's not very polished at the moment because I made it for me but I wanted to make it available, and not have it exclusively bundled with the Obsidian plugin (I might experiment with other interfaces for it).

Add your anthropic api key to the .envsample file and rename it to .env. If you aren't going to use the /chat route then you don't need to do this, it will run without and you can still use the ColBERT, Qwen, and Whisper stuff.

After that 'uv run flask_server.py' should be all you need. If you can run flash attention you should also run 'uv pip install flash-attn --no-build-isolation'. This one might take a while, flash attn take a long time to build. It's supposed to be fast with ninja but idk. You can use the requirements.txt if you don't want to use uv, but you really should just use uv.

Currently the chat functionality uses claudette, but I want to add openrouter and local chat. ColBERT embeddings and QwenVL based pdf metadata extraction assume a local gpu. I use a 3090, haven't tested on anything else. The claude metadata extraction function quickly used up all my api tokens for a day so I would recommend using the default Qwen function for that part but I left in the Claude function for reference.

The Whisper stuff I haven't messed around with enough to find perfect settings, it still fucks up on long files like podcasts.

To restart your ColBERT index completely, remove the .byaldi folder and the state.json file.

Roadmap:
- Cloud/local options for everything
- Better Whisper
- Docs
