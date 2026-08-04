from __future__ import annotations

import socket
import unittest

import numpy as np

from bodylink.geometry import TrackerPose
from bodylink.face import face_output
from bodylink.osc_sender import VRChatOscSender, validate_target
from pythonosc.osc_packet import OscPacket


class FakeClient:
    def __init__(self) -> None:
        self.sent: list[object] = []

    def send(self, content: object) -> None:
        self.sent.append(content)


class OscSenderTests(unittest.TestCase):
    @staticmethod
    def _sent_messages(client: FakeClient) -> dict[str, list[object]]:
        messages: dict[str, list[object]] = {}
        for content in client.sent:
            packet = OscPacket(content.dgram)  # type: ignore[attr-defined]
            for timed_message in packet.messages:
                message = timed_message.message
                messages[message.address] = list(message.params)
        return messages

    def test_bundle_contains_position_and_rotation_addresses(self) -> None:
        sender = VRChatOscSender("127.0.0.1", 9000)
        client = FakeClient()
        sender.close()
        sender._client = client  # type: ignore[attr-defined]
        tracker = TrackerPose(
            role="hip",
            tracker_id=1,
            label="腰",
            position_m=np.array([0.1, 1.0, -0.2]),
            euler_deg=np.array([0.0, 15.0, 0.0]),
            confidence=1.0,
        )
        sender.send_trackers([tracker])
        self.assertEqual(sender.stats.packets_sent, 1)
        self.assertEqual(sender.stats.messages_sent, 2)
        payload = client.sent[0].dgram  # type: ignore[attr-defined]
        self.assertIn(b"/tracking/trackers/1/position", payload)
        self.assertIn(b"/tracking/trackers/1/rotation", payload)

    def test_alignment_uses_official_head_rotation_address(self) -> None:
        sender = VRChatOscSender("127.0.0.1", 9000)
        client = FakeClient()
        sender.close()
        sender._client = client  # type: ignore[attr-defined]
        sender.send_yaw_alignment()
        payload = client.sent[0].dgram  # type: ignore[attr-defined]
        self.assertIn(b"/tracking/trackers/head/rotation", payload)

    def test_vr_head_reference_shares_bundle_with_body_trackers(self) -> None:
        sender = VRChatOscSender("127.0.0.1", 9000)
        client = FakeClient()
        sender.close()
        sender._client = client  # type: ignore[attr-defined]
        tracker = TrackerPose(
            role="hip",
            tracker_id=1,
            label="腰",
            position_m=np.array([0.0, 1.0, 0.0]),
            euler_deg=np.zeros(3),
            confidence=1.0,
        )
        sender.send_trackers(
            [tracker],
            head_position_m=np.array([0.1, 1.65, -0.2]),
            head_yaw_deg=32.0,
        )
        self.assertEqual(len(client.sent), 1)
        messages = self._sent_messages(client)
        self.assertIn("/tracking/trackers/1/position", messages)
        np.testing.assert_allclose(
            messages["/tracking/trackers/head/position"],
            [0.1, 1.65, -0.2],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            messages["/tracking/trackers/head/rotation"],
            [0.0, 32.0, 0.0],
            atol=1e-6,
        )

    def test_target_validation(self) -> None:
        self.assertEqual(validate_target("127.0.0.1", 9000), ("127.0.0.1", 9000))
        with self.assertRaises(ValueError):
            validate_target("", 9000)
        with self.assertRaises(ValueError):
            validate_target("127.0.0.1", 70000)

    def test_bundle_is_delivered_over_udp(self) -> None:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(1.0)
        sender = VRChatOscSender("127.0.0.1", receiver.getsockname()[1])
        tracker = TrackerPose(
            role="hip",
            tracker_id=1,
            label="腰",
            position_m=np.array([0.0, 1.0, 0.0]),
            euler_deg=np.zeros(3),
            confidence=1.0,
        )
        try:
            sender.send_trackers([tracker])
            payload, _ = receiver.recvfrom(4096)
        finally:
            sender.close()
            receiver.close()
        self.assertTrue(payload.startswith(b"#bundle"))
        self.assertIn(b"/tracking/trackers/1/position", payload)

    def test_face_output_contains_native_eyes_vrcft_and_boolean_flags(self) -> None:
        sender = VRChatOscSender("127.0.0.1", 9000)
        client = FakeClient()
        sender.close()
        sender._client = client  # type: ignore[attr-defined]
        sender.send_face(
            face_output(
                {"jawOpen": 0.8, "eyeBlinkLeft": 0.4, "eyeBlinkRight": 0.2},
                native_eyes=True,
            )
        )
        messages = self._sent_messages(client)
        self.assertIn("/tracking/eye/LeftRightPitchYaw", messages)
        self.assertIn("/tracking/eye/EyesClosedAmount", messages)
        self.assertAlmostEqual(messages["/avatar/parameters/v2/JawOpen"][0], 0.8)
        self.assertIs(messages["/avatar/parameters/ExpressionTrackingActive"][0], True)
        self.assertGreater(sender.stats.packets_sent, 1)

    def test_face_reset_uses_vrcft_eye_neutral_and_false_flags(self) -> None:
        sender = VRChatOscSender("127.0.0.1", 9000)
        client = FakeClient()
        sender.close()
        sender._client = client  # type: ignore[attr-defined]
        sender.send_face_reset(native_eyes=False)
        messages = self._sent_messages(client)
        self.assertNotIn("/tracking/eye/LeftRightPitchYaw", messages)
        self.assertEqual(messages["/avatar/parameters/v2/EyeLidLeft"][0], 0.75)
        self.assertEqual(messages["/avatar/parameters/v2/EyeOpenLeft"][0], 1.0)
        self.assertIs(messages["/avatar/parameters/EyeTrackingActive"][0], False)



if __name__ == "__main__":
    unittest.main()
