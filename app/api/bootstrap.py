from __future__ import annotations

from app.api.schemas import TrainRequest
from app.api.service import PlatformService


def main() -> None:
    service = PlatformService()
    actor = {"username": "bootstrap", "role": "admin"}
    report = service.train(TrainRequest(), actor)
    print(report)


if __name__ == "__main__":
    main()

