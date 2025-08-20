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
    response = tokenizer.decoder(model.generate(X, 256).tolist()[0])

    # Remove the initial user input prefix if present
    if response.startswith(sentence):
      response = response[len(sentence):]

    # Only show text before next <speaker1> if present
    if "<speaker1>" in response:
      response = response.split("<speaker1>")[0]

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
