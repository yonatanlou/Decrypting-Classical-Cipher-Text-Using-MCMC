import os
from cipher_decryption import run, set_seed



set_seed(42)


PATH = os.path.dirname(os.path.realpath(__file__))+"/"
TEXT_FILE = 'haaretz.txt'

iterations = [20000, 60000, 100000]
text_files = ["wikipedia.txt", "haaretz.txt"]

message_heb = "המלחמה ברצועת עזה נגמרה לגמרי ושלום עולמי קיים בארץ ישראל אף על פי כך, נדקר טיפוס אחד, המצב בשווקים הידרדר משמעותית בחודש האחרון בעקבות המצב החמור במושבה מאדים"
message_eng = "What is important are the rights of man, emancipation from prejudices, and equality of citizenship, and all these ideas Napoleon has retained in full force"
PICKLE = False
IS_HEBREW = False
results = []

run(PATH, "text_files/war-and-peace.txt", IS_HEBREW, PICKLE, message_eng, plot=False, iterations=100000)

# for txt in text_files:
#
#     SEED_VALUE = 24
#     random.seed(SEED_VALUE)
#     np.random.seed(SEED_VALUE)
#     os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
#
#     res = run(PATH, txt, IS_HEBREW, PICKLE, message_heb, 100000)
#     res.update({"corpus": [txt]*100000})
#     results.append(pd.DataFrame(res))
#
# df_results = pd.concat([results[0],results[1]])
# df_results.to_csv("df_results_seed_24_iters_100000.csv")

