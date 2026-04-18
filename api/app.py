def health_check() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    print(health_check())
