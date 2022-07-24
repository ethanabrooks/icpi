import tomli
import tomli_w


with open("poetry.lock", "rb") as f1, open("typing-extensions.lock", "rb") as f2:
    poetry_lock = tomli.load(f1)
    typing_extensions_lock = tomli.load(f2)


names = [p["name"] for p in poetry_lock["package"]]
poetry_lock["package"].append(typing_extensions_lock)
with open("poetry.lock", "wb") as f:
    tomli_w.dump(poetry_lock, f)
