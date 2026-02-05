/// <reference types="vite/client" />

// Extend Window interface for webkit prefix
interface Window {
  webkitAudioContext?: typeof AudioContext;
}
