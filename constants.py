PATTERN_HEBREW = "[^\u0590-\u05FF\uFB1D-\uFB4F ]"
PATTERN_ENGLISH = "[^A-Z ]"
HEB_ALPHABET = "אבגדהוזחטיכךלמםנןסעפףצץקרשת "
EN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "

LANGUAGES = {
    "hebrew": {
        "pattern": "[^\u0590-\u05FF\uFB1D-\uFB4F ]",
        "alphabet": "אבגדהוזחטיכךלמםנןסעפףצץקרשת ",
    },
    "english": {"pattern": "[^A-Z ]", "alphabet": "ABCDEFGHIJKLMNOPQRSTUVWXYZ "},
}
