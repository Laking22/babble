import type { ReactNode } from "react";
import { ChevronRight } from "lucide-react";
import { openExternal } from "../native";

/** A link inside a how-to step: opens that exact page in the browser. */
export function Site({ href, children }: { href: string; children: ReactNode }) {
  return <button className="link-btn" onClick={() => openExternal(href)}>{children}</button>;
}

/** Step-by-step help under a connection, for people who've never made an API key: which site to open, where to
 *  click, what to copy. Opens by itself while the connection still needs setting up. */
export function HowTo({ title, open, steps, note, path }: {
  title: string; open?: boolean; steps: ReactNode[]; note?: ReactNode; path?: ClickPath[];
}) {
  return (
    <details className="howto" open={open}>
      <summary><ChevronRight size={13} />{title}</summary>
      {path && path.length > 0 && (
        <div className="howto-paths">
          {path.map(([label, ...clicks]) => (
            <div key={label} className="howto-path">
              <span className="howto-path-label">{label}</span>
              {clicks.map((c, i) => <span key={i} className="howto-click">{i > 0 && <ChevronRight size={11} />}<b>{c}</b></span>)}
            </div>
          ))}
        </div>
      )}
      <ol>{steps.map((s, i) => <li key={i}>{s}</li>)}</ol>
      {note && <p className="howto-note">{note}</p>}
    </details>
  );
}

/** [what it's for, then each place to click in order]: shown as "API key: Settings › API Keys › Create". */
type ClickPath = [string, ...string[]];

/** Where to click on each service's site, at a glance (checked against its own help pages, November 2026). */
export const PATHS: Record<string, ClickPath[]> = {
  kaggle: [["Phone", "Profile picture", "Settings", "Phone verification"],
           ["Token (if sign-in fails)", "kaggle.com/settings", "API Tokens", "Generate New Token"]],
  modal: [["Sign in", "Sign in with Modal (here)", "Pick your workspace", "Approve"]],
  runpod: [["Credit", "Runpod console", "Billing", "Add credit"],
           ["API key", "Runpod console", "Credentials", "API Keys", "Create API Key"]],
  vast: [["Credit", "cloud.vast.ai", "Billing", "Add credit"],
         ["API key", "cloud.vast.ai", "Keys", "API Keys", "+New"]],
  thunder: [["Card", "Thunder console", "Settings", "Billing", "Add a card"],
            ["Token", "Thunder console", "Settings", "Tokens", "Generate"]],
  lambda: [["API key", "Lambda Cloud console", "API keys (left menu)", "Generate API key"]],
  lightning: [["Keys", "Profile picture", "Settings", "Keys"]],
  beam: [["Credit", "Beam dashboard", "Settings", "Billing"],
         ["API key", "Beam dashboard", "Settings", "API Keys", "Create"]],
  cerebrium: [["Token", "Cerebrium dashboard", "API Keys", "Create Service Account"]],
  hfjobs: [["Credit", "Profile picture", "Settings", "Billing"],
           ["Token", "Profile picture", "Settings", "Access Tokens", "Create new token", "Write"]],
  huggingface: [["Token", "Profile picture", "Settings", "Access Tokens", "Create new token", "Read"]],
  ssh: [["Vast.ai", "Instances", "Your machine", "Connect"],
        ["Runpod", "Pods", "Your pod", "Connect"],
        ["Your Windows PC", "Settings", "System", "Optional features", "OpenSSH Server"]],
  orpheus: [["In this app", "Models", "Orpheus 3B", "Run Orpheus on", "API server"]],
};

/** Each service's steps, checked against its own help pages (November 2026). */
export const HOWTO = {
  kaggle: [
    <>Make a free account at <Site href="https://www.kaggle.com/account/login?phase=startRegisterTab">kaggle.com</Site>.</>,
    <>Verify your phone number: click your profile picture (top right) → <b>Settings</b> → <b>Phone verification</b>. Kaggle only lends GPUs to verified accounts.</>,
    <>Come back here and click <b>Sign in with Kaggle</b>. Your browser opens a Kaggle page; approve it, and this card turns to Connected.</>,
    <>Sign-in not working? Open <Site href="https://www.kaggle.com/settings/api">Kaggle's API settings</Site>, click <b>Generate New Token</b> under <b>API Tokens</b>, copy the token, and paste it under <b>Or paste an API token</b>.</>,
  ],
  modal: [
    <>Make an account at <Site href="https://modal.com/signup">modal.com</Site> (you can sign up with Google or GitHub). The free plan includes $30 of GPU time each month.</>,
    <>Come back here and click <b>Sign in with Modal</b>.</>,
    <>Your browser opens a Modal page: pick your workspace and approve this computer. There's no key to copy; the app keeps the sign-in by itself.</>,
  ],
  runpod: [
    <>Make an account at <Site href="https://console.runpod.io/signup">runpod.io</Site>.</>,
    <>Add credit on <Site href="https://console.runpod.io/user/billing">Runpod's Billing page</Site>: pick an amount and pay. Runpod's smallest top-up is $10; the app only spends it while a book is being made.</>,
    <>Open <Site href="https://console.runpod.io/user/credentials">Runpod's Credentials page</Site>, choose the <b>API Keys</b> tab, and click <b>Create API Key</b>.</>,
    <>Give it any name (like “audiobooks”), set the permission to <b>All</b>, and click <b>Create</b>.</>,
    <>Click the new key to copy it, paste it into the box above, and click <b>Save</b>. Runpod shows each key only once, so if you lose it, just make another.</>,
  ],
  vast: [
    <>Make an account at <Site href="https://cloud.vast.ai/">cloud.vast.ai</Site>.</>,
    <>Add credit on <Site href="https://cloud.vast.ai/billing/">Vast's Billing page</Site>. The smallest top-up is $5; the app only spends it while a book is being made.</>,
    <>Open <Site href="https://cloud.vast.ai/manage-keys/">Vast's Keys page</Site>, and on its <b>API Keys</b> part click <b>New</b>. Give it any name and keep the permissions it suggests.</>,
    <>Copy the key, paste it into the box above, and click <b>Save</b>.</>,
  ],
  thunder: [
    <>Make an account at <Site href="https://console.thundercompute.com/signup">console.thundercompute.com</Site> (an email address or a Google account).</>,
    <>Add a card on <Site href="https://console.thundercompute.com/settings/billing">Thunder's Billing page</Site>: machines can't start without one.</>,
    <>Open <Site href="https://console.thundercompute.com/settings/tokens">Thunder's Tokens page</Site> and generate a new token. Any name will do.</>,
    <>Copy the token, paste it into the box above, and click <b>Save</b>.</>,
  ],
  lambda: [
    <>Make an account at <Site href="https://cloud.lambda.com/">cloud.lambda.com</Site> and add a card: Lambda bills it for what you use.</>,
    <>In the Lambda Cloud dashboard, open <b>API keys</b> (in the menu on the left) and click <b>Generate API key</b>. Any name will do.</>,
    <>Copy the key (Lambda shows it only once), paste it into the box above, and click <b>Save</b>.</>,
  ],
  lightning: [
    <>Make a free account at <Site href="https://lightning.ai/sign-up">lightning.ai</Site>, and verify your phone number when it asks: that unlocks the free credits.</>,
    <>Open <Site href="https://lightning.ai/account/settings">your Lightning settings</Site> (profile picture → <b>Settings</b>) and go to <b>Keys</b>.</>,
    <>It shows two values: your <b>user ID</b> and your <b>API key</b> (sometimes as two lines starting <code>LIGHTNING_USER_ID</code> and <code>LIGHTNING_API_KEY</code>). Copy both, paste them into the box above together, and click <b>Save</b>.</>,
  ],
  beam: [
    <>Make an account at <Site href="https://platform.beam.cloud/">platform.beam.cloud</Site>.</>,
    <>Add a card or credit on <Site href="https://platform.beam.cloud/settings/billing">Beam's Billing page</Site>.</>,
    <>Open <Site href="https://platform.beam.cloud/settings/api-keys">Beam's API Keys page</Site> (<b>Settings</b> → <b>API Keys</b>) and create a key.</>,
    <>Copy it, paste it into the box above, and click <b>Save</b>.</>,
  ],
  cerebrium: [
    <>Make an account at <Site href="https://dashboard.cerebrium.ai/register">dashboard.cerebrium.ai</Site>, and add a card in the dashboard if it asks.</>,
    <>In the dashboard, open <b>API Keys</b> and click <b>Create Service Account</b>. Give it any name (like “audiobooks”), choose how long it should last, and click <b>Create</b>.</>,
    <>Copy the token it shows, paste it into the box above, and click <b>Save</b>.</>,
  ],
  hfjobs: [
    <>Make a free account at <Site href="https://huggingface.co/join">huggingface.co</Site>.</>,
    <>Add credit on <Site href="https://huggingface.co/settings/billing">Hugging Face's Billing page</Site>: Jobs only run while your balance is above zero (a PRO plan's monthly credit counts).</>,
    <>Open <Site href="https://huggingface.co/settings/tokens">Access Tokens</Site>, click <b>Create new token</b>, choose <b>Write</b>, give it any name, and click <b>Create token</b>.</>,
    <>Copy the token (it starts with <code>hf_</code>), paste it into the box above, and click <b>Save</b>. It's a different token from the read-only one under Downloads.</>,
  ],
  huggingface: [
    <>Make a free account at <Site href="https://huggingface.co/join">huggingface.co</Site>.</>,
    <>Open <Site href="https://huggingface.co/settings/tokens">Access Tokens</Site> (your profile picture → <b>Settings</b> → <b>Access Tokens</b>) and click <b>Create new token</b>.</>,
    <>Choose <b>Read</b>, give it any name, and click <b>Create token</b>.</>,
    <>Copy the token (it starts with <code>hf_</code>), paste it into the box above, and click <b>Save</b>. Hugging Face shows it only once.</>,
  ],
  ssh: [
    <><b>A rented GPU</b> (Vast.ai, a Runpod pod…): first add your Mac's SSH key on the rental site's SSH keys page. Start a machine, open its <b>Connect</b> panel, and copy the SSH command it shows, for example <code>ssh root@203.0.113.5 -p 22044</code>.</>,
    <>Put the part after <code>ssh</code> in <b>Address</b> (<code>root@203.0.113.5</code>) and the number after <code>-p</code> in <b>Port</b> (<code>22044</code>). If the command has <code>-i</code> and a file, put that file in <b>SSH key</b>.</>,
    <><b>Your own NVIDIA PC</b>: turn on its SSH server, then use your user name on it and its address on your network, like <code>you@192.168.1.20</code>, with port 22.</>,
  ],
  orpheus: [
    <>The address and key come from whoever runs the server. If someone set one up for you, ask them for both.</>,
    <>If you run it yourself (for example Orpheus-TTS-FastAPI on a GPU machine), the address is that machine's address with the port the server listens on, like <code>http://192.168.1.20:8880</code>.</>,
    <>Leave the key empty unless the server was set up with one. Click <b>Test</b> to check it answers.</>,
  ],
};
