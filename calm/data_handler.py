########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import re
import pandas as pd

########################################################################################################################
## -- data handler class -- ############################################################################################
########################################################################################################################
class DataHandler():
  def __init__(self):
    super(DataHandler, self).__init__()

  def convert_dailydialog_to_txt(self, csv_path, output_path):
    df = pd.read_csv(csv_path)
    pat = re.compile(r"(?:'([^']+)'|\"([^\"]+)\")")

    with open(output_path, "w", encoding="utf-8") as file:
      for dialog in df['dialog']:
        dialog = dialog.replace("’", "'").replace("‘", "'")
        utterances = [m.group(1) or m.group(2) for m in pat.finditer(dialog)]
        utterances = [u.strip() for u in utterances if u.strip()]
        conversation = []
        for i, utt in enumerate(utterances):
          speaker = "<speaker1>" if i % 2 == 0 else "<speaker2>"
          conversation.append(f"{speaker} {utt}")
        file.write("\n".join(conversation) + "\n\n")

    return

  def load_data(self, path):
    with open(path, "r", encoding = "utf-8") as file:
      text = file.read()
    return text