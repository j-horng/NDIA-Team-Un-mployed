import csv
from typing import Iterator
from common.types import ImuDelta
def imu_from_csv(path: str) -> Iterator[ImuDelta]:
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            yield ImuDelta(ts=row.get("ts",""), dt=float(row.get("dt","0.01")),
                           gyro=(float(row.get("gx","0")),float(row.get("gy","0")),float(row.get("gz","0"))),
                           accel=(float(row.get("ax","0")),float(row.get("ay","0")),float(row.get("az","0"))))
