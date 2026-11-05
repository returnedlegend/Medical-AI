import { useEffect, useState } from "react";
import {
  Activity,
  BookOpen,
  Box,
  CheckCircle2,
  ChevronRight,
  Database,
  FlaskConical,
  FolderOpen,
  Monitor,
  PanelLeft,
  Terminal,
} from "lucide-react";
import Study from "./Study";
import Physics from "./Physics";
import Procedure from "./Procedure";
import ModelDocumentation from "./ModelDocumentation";
import "./academic.css";
type Workspace = "procedure" | "study" | "physics" | "methods";
export default function App() {
  const [workspace, setWorkspace] = useState<Workspace>("procedure"),
    [online, setOnline] = useState(false),
    [tree, setTree] = useState(true);
  const checkConnection = () =>
    fetch("/api/health")
      .then((r) => setOnline(r.ok))
      .catch(() => setOnline(false));
  useEffect(() => {
    let active = true;
    const check = () =>
      fetch("/api/health")
        .then((r) => {
          if (active) setOnline(r.ok);
        })
        .catch(() => {
          if (active) setOnline(false);
        });
    void check();
    const timer = window.setInterval(check, 15000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);
  const views = [
    { key: "procedure", label: "Procedure laboratory", icon: Activity },
    { key: "study", label: "Cohort experiments", icon: Database },
    { key: "physics", label: "SOFA diagnostics", icon: Box },
    { key: "methods", label: "Model documentation", icon: BookOpen },
  ] as const;
  return (
    <div className="desktop-app">
      <header className="desktop-titlebar">
        <div>
          <span className="desktop-app-icon">
            <FlaskConical size={17} />
          </span>
          <strong>Retraction Laboratory</strong>
          <span className="desktop-document">
            Computational surgical research
          </span>
        </div>
        <span className="desktop-mode">
          <Monitor size={12} />
          Local research application
        </span>
      </header>
      <div className="desktop-menubar">
        <button onClick={() => setWorkspace("procedure")}>Laboratory</button>
        <button onClick={() => setWorkspace("study")}>Experiments</button>
        <button onClick={() => setWorkspace("physics")}>Physics</button>
        <button onClick={() => setWorkspace("methods")}>Methods</button>
        <button
          className="desktop-layout-button"
          onClick={() => setTree(!tree)}
          aria-label="Toggle project navigator"
          aria-pressed={tree}
        >
          <PanelLeft size={14} />
        </button>
      </div>
      <nav className="desktop-tabbar" aria-label="Research workspaces">
        {views.map((view) => (
          <button
            key={view.key}
            onClick={() => setWorkspace(view.key)}
            className={workspace === view.key ? "active" : ""}
            aria-pressed={workspace === view.key}
          >
            <view.icon size={14} />
            {view.label}
          </button>
        ))}
      </nav>
      <div className="desktop-body">
        {tree && (
          <aside className="project-tree">
            <h2>PROJECT NAVIGATOR</h2>
            <div className="project-root">
              <FolderOpen size={14} />
              Retraction research
            </div>
            {views.map((view) => (
              <button
                key={view.key}
                className={workspace === view.key ? "active" : ""}
                onClick={() => setWorkspace(view.key)}
              >
                <ChevronRight size={10} />
                <view.icon size={13} />
                <span>{view.label}</span>
              </button>
            ))}
            <div className="project-model-info">
              <span>MODEL CONTEXT</span>
              <p>Exploratory synthetic mechanics</p>
              <p>Illustrative model units</p>
              <p>Versioned, seeded episodes</p>
              <div>
                <Terminal size={12} />
                No clinical deployment
              </div>
            </div>
          </aside>
        )}
        <main className="desktop-workspace">
          {workspace === "procedure" ? (
            <Procedure />
          ) : workspace === "physics" ? (
            <div className="academic-document">
              <div className="academic-page-title">
                <h1>SOFA scene diagnostics</h1>
                <span>Upstream runtime inspection</span>
              </div>
              <Physics />
            </div>
          ) : workspace === "study" ? (
            <Study />
          ) : (
            <ModelDocumentation />
          )}
        </main>
      </div>
      <footer className="desktop-statusbar">
        <span>
          <span className={`connection-light ${online ? "online" : ""}`} />
          <button
            className="connection-refresh"
            aria-label="Refresh engine connection"
            onClick={() => void checkConnection()}
          >
            {online ? "Local engine connected" : "Local engine unavailable"}
          </button>
        </span>
        <span>
          {workspace === "procedure"
            ? "Stateful synthetic environment"
            : workspace === "physics"
              ? "SOFA 24.06 · diagnostic only"
              : "Analytic surrogate · exploratory study"}
        </span>
        <span className="desktop-status-right">
          <CheckCircle2 size={11} />
          Research use only
        </span>
      </footer>
    </div>
  );
}
