"""Download the photos Vector took himself ("Hey Vector, take a picture") into captures/photos/."""

from pathlib import Path

from kin import robot as kin_robot
from kin.robot import call

out = Path(__file__).resolve().parent.parent / "captures" / "photos"
out.mkdir(parents=True, exist_ok=True)

with kin_robot.connect() as robot:
    call(robot.photos.load_photo_info)
    infos = list(robot.photos.photo_info)
    print(f"{len(infos)} photo(s) on the robot")
    for info in infos:
        path = out / f"vector_{info.photo_id}.jpg"
        if path.exists():
            continue
        # ~400 KB each; over Vector's Wi-Fi this has taken up to three minutes.
        path.write_bytes(call(robot.photos.get_photo, info.photo_id, timeout=300).image)
        print(f"saved {path}")
