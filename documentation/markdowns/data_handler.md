# Data Handler
The data handler module is designed to load and preprocess data. The data used in this project is the [DailtDialog dataset](https://www.kaggle.com/datasets/thedevastator/dailydialog-unlock-the-conversation-potential-in), which consists of three csv files for training, testing and validation. Our first step is to turn it all into a text file, and add additional speaker tokens to them for enhanced context, it is important to note that usually the entire `<speaker1>` or `<speaker2>` is added as a token, however production grade tokenization is out of the scope of this project, so we will treat that entire sequnece of characters as it is, without assigning a dedicated token to it. Below is the code we will be using for our data handling, you can find a jupyter notebook that tests the code provided below [here](/development/data_handler_test.ipynb).

```python
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
```

## Navigation Panel
Proceed to the next section: Positional Encoding<br>
[Return to the previous section: Byte-Pair Encoding (BPE)](/documentation/markdowns/tokenization.md)<br>
[Back to the table of contents](/)<br>