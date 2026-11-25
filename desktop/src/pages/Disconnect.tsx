import { useState } from "react";
import { Unplug } from "lucide-react";
import { ConfirmDialog, useToast } from "../components/ui";

/** Disconnect a connection (after asking): forget its sign-in, key or address on this Mac. */
export function Disconnect({ name, note, run }: { name: string; note?: string; run: () => Promise<unknown> }) {
  const toast = useToast();
  const [asking, setAsking] = useState(false);
  const [busy, setBusy] = useState(false);
  const go = async () => {
    setAsking(false);
    setBusy(true);
    try { await run(); toast(`Disconnected ${name}.`); }
    catch (e) { toast((e as Error).message, true); }
    finally { setBusy(false); }
  };
  return (
    <>
      <button className="btn sm ghost" disabled={busy} onClick={() => setAsking(true)}><Unplug size={13} />Disconnect</button>
      {asking && (
        <ConfirmDialog title={`Disconnect ${name}?`} confirm="Disconnect" danger onCancel={() => setAsking(false)} onConfirm={go}>
          {note && <p>{note}</p>}
        </ConfirmDialog>
      )}
    </>
  );
}
