import { useEffect, useRef, useState } from "react";
import { Download, Pause, Play, RotateCcw, RotateCw } from "lucide-react";
import { fileUrl } from "../../api";
import { clock } from "../../components/ui";

export type Track = {
  file: string;
  label: string;
  autoplay?: boolean;
  v?: number;
  seek?: number;      // jump straight to this second (e.g. clicking a paragraph)
  seekNonce?: number; // bumped so clicking the same paragraph twice re-seeks cleanly
  chapter?: number;   // chapter index when playing a chapter WAV (for live paragraph highlight)
} | null;

/** The book's player along the bottom: back 15 s, play/pause, forward 15 s, and where it is. */
export function Player({ track, onSave, onTime, empty }: {
  track: Track;
  onSave?: (file: string) => void;
  onTime?: (seconds: number, playing: boolean) => void;
  empty: string;
}) {
  const el = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [length, setLength] = useState(0);
  const pendingSeek = useRef<number | null>(null);
  const onTimeRef = useRef(onTime);
  onTimeRef.current = onTime;

  const src = track ? fileUrl(track.file, track.v) : "";

  // Source changed: reset counters and apply initial seek/autoplay once metadata is ready
  useEffect(() => {
    const a = el.current;
    if (!a) return;
    setTime(track?.seek ?? 0);
    setLength(0);
    setPlaying(false);
    pendingSeek.current = track?.seek ?? null;
    if (track?.autoplay) {
      if (track.seek != null && track.seek > 0 && a.readyState < 1) {
        // Wait for onLoadedMetadata to seek first so audio never blips from 0:00 before jumping
        return;
      }
      if (track.seek != null && a.readyState >= 1) {
        a.currentTime = track.seek;
        pendingSeek.current = null;
      }
      a.play().catch(() => {});
    }
  }, [src]); // eslint-disable-line react-hooks/exhaustive-deps

  // Same source, new paragraph seek: jump immediately without reloading audio
  useEffect(() => {
    const a = el.current;
    if (!a || !track || track.seek == null) return;
    if (a.readyState >= 1) {
      a.currentTime = track.seek;
      setTime(track.seek);
      onTimeRef.current?.(track.seek, !a.paused);
      if (track.autoplay) a.play().catch(() => {});
    } else {
      pendingSeek.current = track.seek;
      if (track.autoplay) a.play().catch(() => {});
    }
  }, [track?.seek, track?.seekNonce]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = () => { const a = el.current; if (!a || !track) return; if (a.paused) a.play().catch(() => {}); else a.pause(); };
  const skip = (s: number) => {
    const a = el.current;
    if (!a || !track) return;
    const next = Math.max(0, Math.min(a.duration || 0, a.currentTime + s));
    a.currentTime = next;
    setTime(next);
    onTimeRef.current?.(next, !a.paused);
  };

  return (
    <div className="player" aria-label="Player">
      {track && <audio key={src} ref={el} src={src} preload="metadata"
        onPlay={() => { setPlaying(true); onTimeRef.current?.(el.current?.currentTime ?? 0, true); }}
        onPause={() => { setPlaying(false); onTimeRef.current?.(el.current?.currentTime ?? 0, false); }}
        onEnded={() => { setPlaying(false); onTimeRef.current?.(el.current?.currentTime ?? 0, false); }}
        onTimeUpdate={(e) => { const t = e.currentTarget.currentTime; setTime(t); onTimeRef.current?.(t, !e.currentTarget.paused); }}
        onLoadedMetadata={(e) => {
          const a = e.currentTarget;
          setLength(a.duration);
          if (pendingSeek.current != null) {
            a.currentTime = Math.min(pendingSeek.current, Math.max(0, a.duration - 0.05));
            setTime(a.currentTime);
            onTimeRef.current?.(a.currentTime, !a.paused);
            pendingSeek.current = null;
          }
          if (track.autoplay && a.paused) a.play().catch(() => {});
        }} />}
      <div className="player-line">
        <span className="small muted tnum">{clock(time)}</span>
        <input type="range" min={0} max={length || 1} step={0.1} value={Math.min(time, length || 1)} disabled={!track}
          aria-label="Position" onChange={(e) => {
            const a = el.current;
            const v = Number(e.target.value);
            if (a) a.currentTime = v;
            setTime(v);
            onTimeRef.current?.(v, playing);
          }} />
        <span className="small muted tnum">{clock(length)}</span>
      </div>
      <div className="player-controls">
        <span className="player-label small muted">{track ? track.label : empty}</span>
        <button className="tp" onClick={() => skip(-15)} disabled={!track} aria-label="Back 15 seconds" title="Back 15 seconds"><RotateCcw size={18} /></button>
        <button className="tp main" onClick={toggle} disabled={!track} aria-label={playing ? "Pause" : "Play"}>
          {playing ? <Pause size={22} /> : <Play size={22} style={{ marginLeft: 2 }} />}
        </button>
        <button className="tp" onClick={() => skip(15)} disabled={!track} aria-label="Forward 15 seconds" title="Forward 15 seconds"><RotateCw size={18} /></button>
        <span className="player-label right">
          {track && onSave && <button className="btn sm ghost" onClick={() => onSave(track.file)} title="Save this file"><Download size={14} />Save</button>}
        </span>
      </div>
    </div>
  );
}
