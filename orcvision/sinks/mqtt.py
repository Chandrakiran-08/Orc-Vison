"""MQTT sink — publishes each PerceptionEvent as a JSON payload."""

from __future__ import annotations

from orcvision.events import PerceptionEvent


class MQTTSink:
    """Publish JSON perception events to an MQTT topic via paho-mqtt."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        topic: str = "orcvision/events",
        qos: int = 0,
        client_id: str = "orcvision",
        keepalive: int = 60,
    ) -> None:
        import paho.mqtt.client as mqtt

        self.topic = topic
        self.qos = qos
        # Prefer the v2 callback API when available (paho-mqtt >= 2.0).
        try:
            self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        except (AttributeError, TypeError):  # pragma: no cover - older paho
            self._client = mqtt.Client(client_id=client_id)
        self._client.connect(host, port, keepalive)
        self._client.loop_start()

    def emit(self, event: PerceptionEvent) -> None:
        self._client.publish(self.topic, event.to_json(), qos=self.qos)

    def close(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
