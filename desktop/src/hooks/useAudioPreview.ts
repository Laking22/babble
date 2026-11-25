import { useEffect, useRef, useState } from "react";
import { fileUrl, followTask, voiceApi, type Task } from "../api";
import { useToast } from "../components/ui";

/** Shared voice-preview player used by Voices and VoicePicker: plays one preview at a time and cleans up completed task listeners. */
export function useVoicePreview(opts: { active?: boolean; sampleText?: string; language?: string } = {}) {
  const { active = true, sampleText, language } = opts;
  const toast = useToast();
  const el = useRef<HTMLAudioElement | null>(null);
  const [on, setOn] = useState<string | null>(null);
  const [making, setMaking] = useState<Set<string>>(new Set());
  const stops = useRef<Set<() => void>>(new Set());
  const activeRef = useRef(active);
  activeRef.current = active;
  const reqId = useRef(0);

  useEffect(() => {
    if (!active) {
      reqId.current += 1;
      el.current?.pause();
      setOn(null);
    }
  }, [active]);

  useEffect(() => {
    reqId.current += 1;
    if (el.current) {
      el.current.pause();
      setOn(null);
    }
  }, [sampleText]);

  useEffect(() => () => {
    reqId.current += 1;
    el.current?.pause();
    stops.current.forEach((s) => s());
    stops.current.clear();
  }, []);

  const playFile = (voice: string, path: string, id?: number) => {
    if (!activeRef.current) return;
    if (id !== undefined && id !== reqId.current) return;
    if (!el.current) {
      el.current = new Audio();
      el.current.onended = () => setOn(null);
    }
    el.current.pause();
    el.current.src = fileUrl(path);
    el.current.play().then(() => {
      if (id === undefined || id === reqId.current) setOn(voice);
    }, () => toast("Couldn't play the preview.", true));
  };

  const done = (voice: string) => setMaking((m) => {
    const n = new Set(m);
    n.delete(voice);
    return n;
  });

  const toggle = async (voice: string) => {
    const id = ++reqId.current;
    if (on === voice) {
      el.current?.pause();
      setOn(null);
      return;
    }
    el.current?.pause();
    setOn(null);
    if (making.has(voice)) return;
    try {
      const r = await voiceApi.preview(voice, sampleText, language);
      if (id !== reqId.current) return;
      if (r.audio) return playFile(voice, r.audio, id);
      setMaking((m) => new Set(m).add(voice));
      let stopFn: (() => void) | null = null;
      stopFn = followTask(r as Task, () => {}, (t) => {
        done(voice);
        if (stopFn) stops.current.delete(stopFn);
        if (t.state === "done" && t.result?.audio) playFile(voice, t.result.audio, id);
        else if (t.state === "failed" && id === reqId.current) toast(t.error || "Couldn't make the preview.", true);
      });
      stops.current.add(stopFn);
    } catch (e) {
      done(voice);
      if (id === reqId.current) toast((e as Error).message, true);
    }
  };

  return { on, making, toggle };
}

/** Shared single-element audio file preview player with automatic pause on unmount. */
export function useFilePreview() {
  const el = useRef<HTMLAudioElement | null>(null);
  const [on, setOn] = useState<string | null>(null);
  const toast = useToast();

  useEffect(() => () => {
    el.current?.pause();
  }, []);

  const play = (path: string) => {
    if (!el.current) {
      el.current = new Audio();
      el.current.onended = () => setOn(null);
    }
    if (on === path) {
      el.current.pause();
      setOn(null);
      return;
    }
    el.current.src = fileUrl(path);
    el.current.play().then(() => setOn(path), () => toast("Couldn't play that sound.", true));
  };

  return { on, play };
}
