import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import "./styles/global.css";
import "./i18n";

document.addEventListener("contextmenu", (e) => e.preventDefault());
document.addEventListener("click", (e) => {
  const target = e.target as HTMLElement;
  console.log("[UTAI] Click:", target.tagName, target.className, target.textContent?.slice(0, 20));
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
