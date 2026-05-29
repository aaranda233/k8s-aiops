"""
Bus de eventos entre el pipeline (hilo sync) y los WebSockets (async).

Incluye replay buffer: cualquier cliente que conecte tarde recibe
todos los eventos almacenados desde el arranque.
"""

import asyncio
import json
from collections import deque
from dataclasses import dataclass


@dataclass
class PipelineEvent:
    type: str
    data: dict


class EventBus:
    def __init__(self, replay_buffer_size: int = 2000):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queues: list[asyncio.Queue] = []
        # Buffer de replay: todos los clientes que conecten reciben esto primero
        self._replay: deque[str] = deque(maxlen=replay_buffer_size)

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        """
        Crea una cola para un nuevo cliente WebSocket.
        Le entrega el replay buffer completo antes de eventos nuevos.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=5000)
        # Rellenar con el historial acumulado
        for msg in self._replay:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                break
        self._queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._queues.remove(q)
        except ValueError:
            pass

    def publish(self, event: PipelineEvent) -> None:
        payload = json.dumps({"type": event.type, "data": event.data})
        self._replay.append(payload)

        if not self._loop:
            return
        for q in self._queues:
            self._loop.call_soon_threadsafe(self._put_nowait, q, payload)

    @staticmethod
    def _put_nowait(q: asyncio.Queue, payload: str) -> None:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
                q.put_nowait(payload)
            except Exception:
                pass


bus = EventBus()
