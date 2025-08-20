# Introduction
Language models like ChatGPT have gained immense popularity in recent years. They help us perform everyday tasks more efficiently, and their capabilities continue to improve at a rapid pace. While the inner workings of ChatGPT are complex and not easy to fully grasp, the core concept is actually quite simple: given a sequence of words, the model predicts the next word. Think of it like that friend who always finishes your sentences—helpful, yet sometimes a little annoying.

In this project, we will implement our own language model, a simplified version of the production grade models you see today. Our model will predict the next word **autoregressively**, which means it generates one word at a time based on the previous sequence. While this alone will not create a full chatbot, it will allow the model to keep "talking" when given a starting sequence of words.

We will start by introducing **Byte-Pair Encoding (BPE)** and explain how it bridges the gap between human language and how machines understand text. Next, we will explore the dataset we will use, including preprocessing and proper tokenization. Then, we will dive into the core of nearly all modern language models: the **transformer decoder**. Using this building block, we will create our model and finally wrap it in a simple application with a user-friendly interface. Our creation will be called the **Compact AI Language Model (CALM)**.

Without further ado, let's get started!

## Navigation Panel
Use this navigation panel to move forward or backward through the tutorial, or jump straight to the homepage whenever you like.<br>
[Proceed to the next section: Tools](./tools.md)<br>
[Back to the table of contents](/)<br>