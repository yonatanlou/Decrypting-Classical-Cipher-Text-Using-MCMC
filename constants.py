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
BAD_HEBREW_CHARS = [
    1524,
    1523,
    1522,
    1521,
    1520,
    1518,
    1515,
    1480,
    1472,
    1475,
    1478,
]
