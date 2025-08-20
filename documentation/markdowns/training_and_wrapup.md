# Training and Wrapup
The following JSON file contains the configuration used to train the model. The weights provided are from an incomplete training phase, as I was unable to finish due to the heavy computational requirements. With enough epochs and a more complex architecture, the results can be significantly better than what is shown here.
```json
{
  "device": "cpu",
  "val_iter": 1,
  "val_freq": 200,
  "save_path": "./weights/",
  "num_heads": 16,
  "model_embd": 256,
  "vocab_size": 1024,
  "batch_size": 256,
  "block_size": 256,
  "hidden": 1024,
  "dropout_p": 0.1,
  "num_layers": 6
}
```

A few notes about the current model: it is designed to autoregressively predict the most probable next token, where a token is not necessarily a word but a byte-pair. In its current state, the model cannot reliably answer specific questions without additional processing and fine-tuning. To work around this limitation, we applied a small trick.

In future training, it is best to include an `<eos>` token so the model knows when to stop generating. For our dataset of conversations, we instead used a preprocessing step: each sentence was prefixed with either `<speaker1>` or `<speaker2>`. When prompting the model, we prepend a `<speaker1>` tag to the input, which increases the likelihood that the response begins with `<speaker2>`. The model then continues generating text, sometimes even adding extra `<speaker1>` segments. To handle this, we simply extract the first response that appears between `<speaker2>` and `<speaker1>`.

With more training, this approach will produce much stronger results. For now, the model can sometimes generate coherent sentences or context-aware responses. For example, when asked about pizza, a half-trained version of the model produced an answer about food. While not precise, it shows clear progress compared to earlier outputs.

![CALM Chat Screen Shot](/documentation/visuals/CALM_V1_ScreenShot.png)

And now here you can find the code that is used to create the UI, we are using FLET package here, which is very easy to use, and you can also create exe files from it, so when you are done training your models, you can really turn it into an offline app and have fun with it.
```python
import flet as ft
import calm
import json
import torch

# Load parameters
with open("./weights/params.json", "r") as f:
  params = json.load(f)

# Load tokenizer
tokenizer = calm.tokenization.Tokenizer()
tokenizer.load_state("./data/vocab/tokenizer_state.pkl")

# Load model
model = calm.CompactAiLanguageModel(parameters=params)
model.load_weights(path="./weights/weights.pth")

BUBBLE_MAX_WIDTH_RATIO = 0.6  # 60% of page width

def main(page: ft.Page):
  page.title = "CompactAiLanguageModel (CALM)"
  page.vertical_alignment = "stretch"

  chat = ft.ListView(
    expand=True,
    spacing=10,
    auto_scroll=True,
  )

  input_box = ft.TextField(
    hint_text="Type your message...",
    expand=True,
    autofocus=True,
    multiline=False,
  )

  # Add message bubble
  def add_message(text, is_user=True):
    max_width = int(page.width * BUBBLE_MAX_WIDTH_RATIO) if page.width else 400
    estimated_width = len(text) * 8 + 20
    bubble_width = estimated_width if estimated_width < max_width else max_width

    bubble = ft.Container(
      content=ft.Text(
        text,
        selectable=True,
        size=16,
        no_wrap=False,
      ),
      padding=10,
      border_radius=10,
      bgcolor="lightblue" if is_user else "#e6f2ff",  # subtle pastel for machine
      margin=5,
    )

    if estimated_width > max_width:
      bubble.width = max_width

    row = ft.Row(
      controls=[bubble],
      alignment="end" if is_user else "start",
    )
    chat.controls.append(row)
    page.update()

  # Send handler
  def send_message(e=None):
    user_text = input_box.value.strip()
    if not user_text:
      return

    add_message(user_text, is_user=True)

    sentence = "<speaker1>" + user_text
    X = torch.tensor([tokenizer.encoder(sentence)])
    response = tokenizer.decoder(model.generate(X, 64).tolist()[0])

    # Remove the initial user input prefix if present
    if response.startswith(sentence):
      response = response[len(sentence):]

    # Only show text before next <speaker1> if present
    if "<speaker1>" in response:
      response = response.split("<speaker1>")[1]

    # Remove all <speaker2> tokens
    response = response.replace("<speaker2>", "")

    add_message(response.strip(), is_user=False)

    input_box.value = ""
    page.update()
    input_box.focus()

  send_button = ft.ElevatedButton("Send", on_click=send_message)
  input_box.on_submit = send_message  # press Enter submits

  input_row = ft.Row(
    controls=[input_box, send_button],
    alignment="spaceBetween",
  )

  page.add(
    ft.Column(
      controls=[chat, input_row],
      expand=True,
    )
  )

ft.app(target=main)
```

## Navigation Panel
Use this navigation panel to move forward or backward through the tutorial, or jump straight to the homepage whenever you like.<br>
[Return to the previous section: Transformer Decoder](/documentation/markdowns/transformer_decoder.md)<br>
[Back to the table of contents](/)<br>