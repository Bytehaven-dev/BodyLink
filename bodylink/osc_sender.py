from __future__ import annotations

import socket
from collections.abc import Sequence
from dataclasses import dataclass

from pythonosc.osc_bundle_builder import IMMEDIATELY, OscBundleBuilder
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.udp_client import SimpleUDPClient

from bodylink.geometry import TrackerPose
from bodylink.face import FaceOutput, FaceParameter, neutral_face_output


@dataclass(frozen=True, slots=True)
class OscStats:
    packets_sent: int
    messages_sent: int


def validate_target(host: str, port: int) -> tuple[str, int]:
    clean_host = host.strip()
    if not clean_host:
        raise ValueError("OSC 地址不能为空")
    if not 1 <= int(port) <= 65535:
        raise ValueError("OSC 端口必须在 1 到 65535 之间")
    socket.getaddrinfo(clean_host, int(port), type=socket.SOCK_DGRAM)
    return clean_host, int(port)


def _message(address: str, values: Sequence[FaceParameter]):
    builder = OscMessageBuilder(address=address)
    for value in values:
        if isinstance(value, bool):
            builder.add_arg(value, arg_type="T" if value else "F")
        else:
            builder.add_arg(float(value), arg_type="f")
    return builder.build()


class VRChatOscSender:
    def __init__(self, host: str, port: int) -> None:
        self._host, self._port = validate_target(host, port)
        self._client = SimpleUDPClient(self._host, self._port)
        self._packets_sent = 0
        self._messages_sent = 0

    @property
    def target(self) -> tuple[str, int]:
        return self._host, self._port

    @property
    def stats(self) -> OscStats:
        return OscStats(self._packets_sent, self._messages_sent)

    def configure(self, host: str, port: int) -> None:
        if (host.strip(), int(port)) == self.target:
            return
        target = validate_target(host, port)
        if target == self.target:
            return
        self._close_client()
        self._host, self._port = target
        self._client = SimpleUDPClient(self._host, self._port)

    def _close_client(self) -> None:
        sock = getattr(self._client, "_sock", None)
        if sock is not None:
            sock.close()

    def close(self) -> None:
        self._close_client()

    def send_trackers(
        self,
        trackers: list[TrackerPose],
        head_position_m: Sequence[float] | None = None,
        head_yaw_deg: float | None = None,
    ) -> None:
        if not trackers and head_position_m is None and head_yaw_deg is None:
            return

        bundle = OscBundleBuilder(IMMEDIATELY)
        messages = 0
        for tracker in trackers:
            base = f"/tracking/trackers/{tracker.tracker_id}"
            bundle.add_content(_message(f"{base}/position", tracker.position_m.tolist()))
            bundle.add_content(_message(f"{base}/rotation", tracker.euler_deg.tolist()))
            messages += 2

        if head_position_m is not None:
            bundle.add_content(
                _message("/tracking/trackers/head/position", list(head_position_m))
            )
            messages += 1
        if head_yaw_deg is not None:
            bundle.add_content(
                _message(
                    "/tracking/trackers/head/rotation",
                    [0.0, float(head_yaw_deg), 0.0],
                )
            )
            messages += 1

        self._client.send(bundle.build())
        self._packets_sent += 1
        self._messages_sent += messages

    def send_face(self, output: FaceOutput) -> None:
        messages: list[tuple[str, Sequence[FaceParameter]]] = []
        if output.native_eye is not None:
            eye = output.native_eye
            messages.extend(
                (
                    (
                        "/tracking/eye/LeftRightPitchYaw",
                        [
                            eye.left_pitch_deg,
                            eye.left_yaw_deg,
                            eye.right_pitch_deg,
                            eye.right_yaw_deg,
                        ],
                    ),
                    ("/tracking/eye/EyesClosedAmount", [eye.closed_amount]),
                )
            )
        messages.extend(
            (f"/avatar/parameters/{name}", [value])
            for name, value in output.parameters.items()
        )

        for start in range(0, len(messages), 16):
            chunk = messages[start : start + 16]
            bundle = OscBundleBuilder(IMMEDIATELY)
            for address, values in chunk:
                bundle.add_content(_message(address, values))
            self._client.send(bundle.build())
            self._packets_sent += 1
            self._messages_sent += len(chunk)

    def send_face_reset(self, native_eyes: bool) -> None:
        self.send_face(neutral_face_output(native_eyes))

    def send_yaw_alignment(self) -> None:
        self._client.send(_message("/tracking/trackers/head/rotation", [0.0, 0.0, 0.0]))
        self._packets_sent += 1
        self._messages_sent += 1
