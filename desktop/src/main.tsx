import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { ToastHost } from "./components/ui";
import { CostConfirmHost } from "./components/CostConfirm";
import "./styles.css";
import { markDesktop } from "./native";
import { keepTextSelectable } from "./textDrag";
import { applyTheme, readTheme } from "./uiPrefs";

applyTheme(readTheme());
markDesktop();
keepTextSelectable();

// Mount immediately so Splash renders on frame 1 (engineReady runs inside App once /api/health answers).
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ToastHost>
      <CostConfirmHost>
        <App />
      </CostConfirmHost>
    </ToastHost>
  </StrictMode>,
);
