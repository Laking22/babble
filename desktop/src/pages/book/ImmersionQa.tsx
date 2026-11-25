import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Play, RefreshCw, ShieldCheck, Sparkles } from "lucide-react";
import {
  bookApi, directorApi, useLoad, useTask,
  type BookRequest, type QaIssue, type QaReport,
} from "../../api";
import { useCostConfirm } from "../../components/CostConfirm";
import { Starting, StopButton, clock, useStarting, useToast } from "../../components/ui";

const SEV_CLS: Record<QaIssue["severity"], string> = {
  high: "chip warn",
  medium: "chip accent",
  low: "chip",
};

/** Two-stage Immersion QA ('Finish-up') modal:
 *  - Stage 1 (Local, free): checks script tags/cast/missing sounds + audio continuity DSP (volume jumps, clipping, dead air, splice clicks).
 *  - Stage 2 (Optional Cloud AI): reviews narrative & cue continuity using the book's configured Director provider/model with upfront cost confirmation. */
export function ImmersionQaDialog({
  folder,
  story,
  directorWhere,
  directorModels,
  request,
  busy,
  onJump,
  onRegen,
  onClose,
}: {
  folder: string;
  story: string;
  directorWhere: string;
  directorModels: Record<string, string>;
  request: () => BookRequest;
  busy: boolean;
  onJump: (chapter: number, para: number, start?: number | null) => void;
  onRegen: (chapter: number, para: number) => void;
  onClose: () => void;
}) {
  const toast = useToast();
  const confirmCost = useCostConfirm();
  const providers = useLoad(directorApi.providers);
  const [report, setReport] = useState<QaReport | null>(null);
  const [est, setEst] = useState<{ cost: string; free_tier?: boolean; chapters: number } | null>(null);

  const ready = (providers.data || []).filter((s) => s.has_key);
  const svc = ready.find((s) => s.id === directorWhere) || ready[0] || null;
  const model = svc ? (directorModels[svc.id] || svc.model || svc.models?.[0]?.id || "") : "";

  useEffect(() => {
    bookApi.qaGet(folder).then((r) => setReport(r?.checked ? r : null)).catch(() => {});
  }, [folder]);

  useEffect(() => {
    if (!svc || !model || !story.trim()) { setEst(null); return; }
    bookApi.qaEstimate(story, svc.id, model).then(setEst).catch(() => setEst(null));
  }, [svc?.id, model, story.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const qaTask = useTask((t) => {
    if (t.state === "failed") return toast(t.error || "Immersion QA failed.", true);
    if (t.result?.qa) setReport(t.result.qa);
    if (t.status) toast(t.status);
  }, `qa-${folder}`);

  const runStage1 = () => {
    qaTask.start(() => bookApi.qaRun({ ...request(), stage2: false }));
  };

  const stage2 = useStarting();
  const runStage2 = async () => {
    if (!svc || !model) {
      toast("Connect an AI provider (and pick a model in Director) to run Stage 2 review.", true);
      return;
    }
    const e = est || await bookApi.qaEstimate(story, svc.id, model).catch(() => null);
    const ok = await confirmCost({
      title: `Run Stage 1 + Stage 2 Immersion QA with ${svc.name}?`,
      confirm: "Run QA",
      cost: e?.cost || "price unknown",
      free: !!e?.free_tier,
      details: [
        `Model: ${model}`,
        e ? `${e.chapters} chapter${e.chapters === 1 ? "" : "s"} reviewed for speaker, ambience, anachronism & mood continuity` : "",
      ],
    });
    if (!ok) return;
    qaTask.start(() => bookApi.qaRun({ ...request(), stage2: true, where: svc.id, model }));
  };

  const issues = report?.issues || [];

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal line-check" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Immersion QA">
        <div className="row" style={{ marginBottom: 8 }}>
          <ShieldCheck size={18} style={{ color: "var(--accent)" }} />
          <h2 style={{ margin: 0 }}>Finish-up Immersion QA</h2>
          <span className="spacer" />
          <button className="btn sm ghost" onClick={onClose}>Close</button>
        </div>

        <div className="modal-body stack" style={{ gap: 12 }}>
          <p className="small muted" style={{ margin: 0 }}>
            <b>Stage 1 (Free, local)</b> checks script tags, cast, missing sounds, and rendered audio continuity
            (volume jumps, clipping, dead air, splice clicks). <b>Stage 2 (Optional AI)</b> reviews story &amp; cue continuity across chapters.
          </p>

          <div className="row wrap" style={{ gap: 8 }}>
            <button className="btn sm primary" disabled={busy || qaTask.running} onClick={runStage1}>
              <ShieldCheck size={14} />Stage 1 · Local audit (Free)
            </button>
            <button className="btn sm" disabled={busy || qaTask.running || stage2.starting || !svc || !model} onClick={() => stage2.run(runStage2)}
              title={svc ? `Also run Stage 2 with ${svc.name} (${model})` : "Connect an AI service in Connections first"}>
              {stage2.starting ? <Starting /> : <><Sparkles size={14} />Stage 1 + 2 · AI review{est?.cost ? ` (${est.cost})` : ""}</>}
            </button>
            {qaTask.running && <StopButton key={qaTask.task?.id} onStop={qaTask.stop} stopping={qaTask.task?.stopping} />}
          </div>

          {qaTask.running && (
            <div className="progress">
              <div className={`bar ${qaTask.task?.progress == null ? "indeterminate" : ""}`}>
                <i style={{ width: `${Math.round((qaTask.task?.progress ?? 0) * 100)}%` }} />
              </div>
              <span className="small muted">{qaTask.task?.status || "Running immersion checks…"}</span>
            </div>
          )}

          {report && !qaTask.running && (
            <div className="stack" style={{ gap: 8 }}>
              <div className="row">
                {issues.length === 0 ? (
                  <span className="small ok-text row" style={{ gap: 6 }}>
                    <CheckCircle2 size={14} />No immersion issues found across {report.rendered_chapters} rendered chapter{report.rendered_chapters === 1 ? "" : "s"}.
                  </span>
                ) : (
                  <span className="small muted">
                    Found <b>{issues.length}</b> issue{issues.length === 1 ? "" : "s"} ({report.rendered_chapters} of {report.total_chapters} chapters have rendered audio).
                  </span>
                )}
              </div>

              {issues.length > 0 && (
                <div className="line-check-list">
                  {issues.map((it) => (
                    <div key={it.id} className="line-check-row">
                      <AlertTriangle size={15} style={{ marginTop: 3, flex: "none", color: it.severity === "high" ? "var(--warn)" : "var(--muted)" }} />
                      <div className="line-check-text grow">
                        <div className="row wrap" style={{ gap: 6 }}>
                          <strong>{it.title}</strong>
                          <span className={SEV_CLS[it.severity]}>{it.severity}</span>
                          <span className="chip">{it.stage === 1 ? "Local" : "AI"}</span>
                          {it.start != null && <span className="small faint tnum">{clock(it.start)}</span>}
                        </div>
                        <div className="small muted">{it.detail}</div>
                      </div>
                      <div className="row" style={{ gap: 4, flex: "none" }}>
                        <button className="btn sm ghost" onClick={() => { onJump(it.chapter, it.para, it.start); onClose(); }}
                          title="Jump to this paragraph in Listen & Direct">
                          <Play size={12} />Jump
                        </button>
                        {it.can_regen && (
                          <button className="btn sm ghost" disabled={busy}
                            onClick={() => { onRegen(it.chapter, it.para); onClose(); }}
                            title="Regenerate just this paragraph">
                            <RefreshCw size={12} />Redo
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
