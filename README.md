# Compact AI Language Model (CALM)
In this project, we will design a lightweight language model called **CALM**, short for **Compact AI Language Model** (pretty neat, right?). Like other language models, CALM can interact with users by receiving a sequence of words, characters, or byte-pairs (depending on the tokenization method) and generating a probabilistic response that best fits the input.

This project is intended as a complete, end-to-end tutorial, covering everything from the fundamentals to full implementation of a compact language model. By the end, we will also build a user interface (UI) that enables direct interaction with CALM.

Overall, this tutorial is designed to be a practical, hands-on learning journey. While it requires some time, energy, and dedication, it promises to be both educational and fun.

## Table of Contents
Here is the table of contents for this tutorial. You can jump directly to the topic of your choice, or use the navigation panel at the bottom of the document to move step by step through the tutorial.

|     | Topic                                                                          | Description                                   |
|-----|--------------------------------------------------------------------------------|-----------------------------------------------|
| 1   | [Introduction](/documentation/markdowns/Introduction.md)                       | Quick intro on what we are doing Here.        |
| 2   | [Tools](/documentation/markdowns/tools.md)                                     | Introduction to tools and python environment. |
| 3   | [Byte-Pair Encoding (BPE)](/documentation/markdowns/tokenization.md)           | Introduction to Byte-Pair Encoding.           |
| 4   | [Data Handler](/documentation/markdowns/data_handler.md)                       | Preprocessing the DailyDialog Dataset.        |
| 5   | [Compact Ai Language Model (CALM)](/documentation/markdowns/language_model.md) | Boilerplate for the language model.           |
| 6   | [Transformer Decoder](/documentation/markdowns/transformer_decoder.md)         | Transformer Decoder for language tasks.       |
| 7   | [Training and Wrapup](/documentation/markdowns/training_and_wrapup.md)         | Training the model and creating a simple UI.  |

## Acknowledgments

In the preparation of this project, the following resources were used:

- [DailyDialog Dataset](http://yanran.li/dailydialog) — a high-quality multi-turn dialog dataset used as the training corpus.

- [Video by Andrej Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE&t=7271s) and [minBPE repository](https://github.com/karpathy/minbpe) — excellent resources for understanding and implementing Byte-Pair Encoding (BPE).

- [Video by Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [ng-video-lecture repository](https://github.com/karpathy/ng-video-lecture) — a practical walkthrough of building a language model. The original uses character-level tokenization, which I replaced with BPE.

- [Transformer-Neural-Network repository](https://github.com/ajhalthor/Transformer-Neural-Network) — the decoder architecture in this project is based on this work, with modifications to apply the pre-norm approach (LayerNorm before multi-head attention and feed-forward layers).

## Navigation Panel
[Proceed to the next section: Byte-Pair Encoding (BPE)](/documentation/markdowns/tokenization.md)<br>