import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { spendingApi, type Spending } from "../api";
import { ConfirmDialog } from "./ui";

/** What an action will cost, for the confirmation shown before anything is sent. */
export type CostInfo = {
  title: string;          // "Direct with OpenRouter?"
  cost: string;           // "$0.03–$0.06", "free: 0.2 to 1.0 of your weekly Kaggle GPU hours"
  details?: string[];     // what it does and where: model, scenes, GPU, time
  free?: boolean;         // uses free allowances only (no money)
  confirm?: string;       // the button: "Direct", "Send", "Start training"
};
/** A question with up to two answers besides Cancel: "confirm" or "other", null for Cancel (named by `cancel`). */
export type AskInfo = { title: string; lines: string[]; confirm: string; other?: string; cancel?: string };
type Answer = "confirm" | "other" | null;
type Open = { kind: "cost"; info: CostInfo } | { kind: "ask"; info: AskInfo };
type Ctx = { cost: (info: CostInfo) => Promise<boolean>; ask: (info: AskInfo) => Promise<Answer> };
const Ctx = createContext<Ctx>({ cost: async () => true, ask: async () => "confirm" });

/** Before anything that costs money or a service's allowance: what it will cost, then Confirm or Cancel. */
export const useCostConfirm = () => useContext(Ctx).cost;
/** A plain question in the same dialog ("Every chapter is made already. Remake the audiobook?"). */
export const useAsk = () => useContext(Ctx).ask;

export function CostConfirmHost({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState<Open | null>(null);
  const [spend, setSpend] = useState<Spending | null>(null);
  const answer = useRef<(a: Answer) => void>(() => {});
  const cost = useCallback((info: CostInfo) => new Promise<boolean>((resolve) => {
    answer.current = (a) => resolve(a === "confirm");
    setSpend(null);
    setOpen({ kind: "cost", info });
  }), []);
  const ask = useCallback((info: AskInfo) => new Promise<Answer>((resolve) => {
    answer.current = resolve;
    setOpen({ kind: "ask", info });
  }), []);
  useEffect(() => {
    if (open?.kind === "cost" && !open.info.free) spendingApi.get().then(setSpend).catch(() => {});
  }, [open]);
  const done = (a: Answer) => { setOpen(null); answer.current(a); };
  return (
    <Ctx.Provider value={{ cost, ask }}>
      {children}
      {open && (
        <ConfirmDialog title={open.info.title} confirm={open.info.confirm || "Confirm"} onConfirm={() => done("confirm")} onCancel={() => done(null)}
          other={open.kind === "ask" ? open.info.other : undefined} onOther={() => done("other")}
          cancel={open.kind === "ask" ? open.info.cancel : undefined}>
          {open.kind === "cost" ? (
            <>
              <p style={{ margin: "0 0 8px" }}>{open.info.free ? "Uses" : "Estimated cost"}: <b>{open.info.cost}</b></p>
              {(open.info.details || []).filter(Boolean).map((d) => <p key={d} className="small muted" style={{ margin: "2px 0" }}>{d}</p>)}
              {spend && (
                <p className="small muted" style={{ margin: "8px 0 0" }}>
                  Spent today: ${spend.today.toFixed(2)}{spend.limit ? ` of your $${spend.limit.toFixed(2)} limit` : ""}. Estimates; the service's bill is what counts.
                </p>
              )}
            </>
          ) : open.info.lines.filter(Boolean).map((d, i) => (
            <p key={d} className={i ? "small muted" : undefined} style={{ margin: i ? "2px 0" : "0 0 8px" }}>{d}</p>
          ))}
        </ConfirmDialog>
      )}
    </Ctx.Provider>
  );
}
