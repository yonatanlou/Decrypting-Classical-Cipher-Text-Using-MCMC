import os
from cipher_decryption import run, set_seed
import click


set_seed(42)


PATH = os.path.dirname(os.path.realpath(__file__)) + "/"
TEXT_FILE = "haaretz.txt"

message_heb = "המלחמה ברצועת עזה נגמרה לגמרי ושלום עולמי קיים בארץ ישראל אף על פי כך, נדקר טיפוס אחד, המצב בשווקים הידרדר משמעותית בחודש האחרון בעקבות המצב החמור במושבה מאדים"
message_eng = "What is important are the rights of man, emancipation from prejudices, and equality of citizenship, and all these ideas Napoleon has retained in full force"

run(PATH, TEXT_FILE, "hebrew", message_heb, plot=False, iterations=150000)
