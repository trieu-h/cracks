import { useEffect, useRef, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

export function useWebSocket(
  path: string | null,
  onMessage: (data: any) => void
) {
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    if (!path) return;

    const socket = io(`${WS_URL}${path}`);
    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('WebSocket connected:', path);
    });

    socket.on('message', onMessage);

    socket.on('disconnect', () => {
      console.log('WebSocket disconnected:', path);
    });

    return () => {
      socket.disconnect();
    };
  }, [path, onMessage]);

  const sendMessage = useCallback((data: any) => {
    if (socketRef.current) {
      socketRef.current.emit('message', data);
    }
  }, []);

  return { sendMessage };
}
