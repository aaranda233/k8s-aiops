"""
Tests del bus de eventos pipeline→WebSocket (web/event_bus.py).

Verifica el replay buffer (clientes tardíos reciben el historial), la publicación
sin loop (solo buffer), y la baja de suscriptores.
"""

import pytest

from web.event_bus import EventBus, PipelineEvent


@pytest.mark.unit
def test_publish_without_loop_only_buffers():
    bus = EventBus()
    bus.publish(PipelineEvent("anomaly", {"score": 0.9}))
    # Sin loop no hay entrega, pero el evento queda en el replay buffer
    q = bus.subscribe()
    assert q.qsize() == 1


@pytest.mark.unit
def test_late_subscriber_gets_replay_history():
    bus = EventBus()
    bus.publish(PipelineEvent("a", {"i": 1}))
    bus.publish(PipelineEvent("b", {"i": 2}))
    q = bus.subscribe()
    assert q.qsize() == 2
    import json
    first = json.loads(q.get_nowait())
    assert first["type"] == "a"
    assert first["data"] == {"i": 1}


@pytest.mark.unit
def test_replay_buffer_bounded():
    bus = EventBus(replay_buffer_size=3)
    for i in range(5):
        bus.publish(PipelineEvent("e", {"i": i}))
    q = bus.subscribe()
    assert q.qsize() == 3  # solo los 3 últimos


@pytest.mark.unit
def test_unsubscribe_removes_queue():
    bus = EventBus()
    q = bus.subscribe()
    bus.unsubscribe(q)
    assert q not in bus._queues
    # Doble unsubscribe no rompe
    bus.unsubscribe(q)
