// Audiobooks being made, followed for the whole app: a book keeps being made (and its shelf card shows how far)
// after its page is closed, and a window reload picks the task up again.
import { createContext, ReactNode, useCallback, useContext, useEffect, useRef, useState } from "react";
import { api, followTask, projectApi, type Task } from "./api";
import { useToast } from "./components/ui";

type Jobs = Record<string, Task>; // project id -> its latest Create (or Preview) task
type Ctx = {
  jobs: Jobs;
  start: (book: string, name: string, begin: () => Promise<Task>) => Promise<void>;
  stop: (book: string) => void;
  /** Called when a book's text changed while it was made (the AI added sound tags) or it finished. */
  onChange: (fn: (book: string, t: Task) => void) => () => void;
};
const JobsCtx = createContext<Ctx>({ jobs: {}, start: async () => {}, stop: () => {}, onChange: () => () => {} });
export const useBookJobs = () => useContext(JobsCtx);

const KEY = "book-tasks";
const saved = (): Record<string, string> => { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch { return {}; } };
const remember = (m: Record<string, string>) => { try { localStorage.setItem(KEY, JSON.stringify(m)); } catch { /* private window */ } };

export function BookJobsHost({ children }: { children: ReactNode }) {
  const [jobs, setJobs] = useState<Jobs>({});
  const closers = useRef<Record<string, () => void>>({});
  const listeners = useRef(new Set<(book: string, t: Task) => void>());
  const lastStory = useRef<Record<string, string>>({});
  const toast = useToast();

  const emit = (book: string, t: Task) => listeners.current.forEach((fn) => fn(book, t));

  const follow = useCallback((book: string, name: string, t: Task) => {
    closers.current[book]?.();
    setJobs((j) => ({ ...j, [book]: t }));
    const update = (cur: Task) => {
      setJobs((j) => ({ ...j, [book]: cur }));
      const r = cur.result;
      if (r.story && r.story !== lastStory.current[book]) { // the AI added sound tags: keep them in the book's text
        lastStory.current[book] = r.story;
        projectApi.save(book, { story: r.story, original: r.original ?? "" }).catch(() => {});
        emit(book, cur);
      }
    };
    closers.current[book] = followTask(t, update, (final) => {
      update(final);
      delete closers.current[book];
      const m = saved(); delete m[book]; remember(m);
      if (final.state === "done" && final.result.audio && !final.title.startsWith("Book preview")) {
        projectApi.save(book, { file: final.result.audio }).catch(() => {});
        toast(`“${name}” is ready.`);
      } else if (final.state === "failed") toast(final.error || `Couldn't make “${name}”.`, true);
      emit(book, final);
    });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { // after a reload: follow the books that were still being made
    const m = saved();
    Object.entries(m).forEach(([book, id]) => {
      api.task(id).then((t) => (t.state === "running" ? follow(book, t.title.replace(/^Audiobook: /, ""), t) : null))
        .catch(() => { const n = saved(); delete n[book]; remember(n); });
    });
    const all = closers.current;
    return () => Object.values(all).forEach((c) => c());
  }, [follow]);

  const start = useCallback(async (book: string, name: string, begin: () => Promise<Task>) => {
    const placeholder: Task = { id: "", kind: "book", title: name, state: "running", status: "Starting…", progress: null, result: {}, error: null };
    setJobs((j) => ({ ...j, [book]: placeholder }));
    try {
      const t = await begin();
      remember({ ...saved(), [book]: t.id });
      follow(book, name, t);
    } catch (e) {
      setJobs((j) => ({ ...j, [book]: { ...placeholder, state: "failed", error: (e as Error).message } }));
    }
  }, [follow]);

  const stop = useCallback((book: string) => {
    const t = jobs[book];
    if (t?.id) api.stopTask(t.id).catch(() => {});
  }, [jobs]);

  const onChange = useCallback((fn: (book: string, t: Task) => void) => {
    listeners.current.add(fn);
    return () => { listeners.current.delete(fn); };
  }, []);

  return <JobsCtx.Provider value={{ jobs, start, stop, onChange }}>{children}</JobsCtx.Provider>;
}
