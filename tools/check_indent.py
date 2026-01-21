import tokenize
from pathlib import Path

p = Path("D:/programacion/Menipy/src/menipy/gui/mainwindow.py")
with p.open("rb") as f:
    try:
        for tok in tokenize.tokenize(f.readline):
            if tok.type in (
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.NL,
                tokenize.NEWLINE,
            ):
                print(tok)
    except Exception as e:
        print("EXC", e)
