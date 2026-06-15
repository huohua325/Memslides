import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CheckCircle2,
  Download,
  Eye,
  FileArchive,
  FileText,
  FolderOpen,
  KeyRound,
  Layers3,
  Loader2,
  MemoryStick,
  Plus,
  RefreshCw,
  Save,
  Send,
  Settings2,
  Sparkles,
  Trash2,
  Upload,
  Wand2,
} from "lucide-react";

type SessionSummary = {
  session_id: string;
  display_name?: string;
  instruction_summary?: string;
  state?: string;
  phase?: string;
  message?: string;
  language?: "en" | "zh";
  slide_count?: number;
  has_exports?: boolean;
  updated_at?: string;
  created_at?: string;
  last_operation?: { state?: string; phase?: string; message?: string; operation_type?: string } | null;
  queued_operation?: string;
  memory_intent?: string;
  memory_profile_id?: string;
  service_profile_id?: string;
};

type ServiceProfile = {
  profile_id: string;
  display_name?: string;
  model?: string;
  base_url?: string;
  enabled?: boolean;
  is_default?: boolean;
  max_concurrent?: number;
  required_ready?: boolean;
  validation_status?: string;
  validation_message?: string;
  services?: Record<string, any>;
  llm?: Record<string, any>;
  pdf?: Record<string, any>;
  embedding?: Record<string, any>;
  search?: Record<string, any>;
  image_generation?: Record<string, any>;
};

type MemoryProfile = {
  memory_profile_id: string;
  name?: string;
  intent?: string;
  profile?: Record<string, any>;
  is_default?: boolean;
  updated_at?: string;
};

type TemplateSummary = {
  template_id?: string;
  id?: string;
  name?: string;
  slide_count?: number;
  layout_count?: number;
  image_count?: number;
  updated_at?: number;
};

type SlideItem = {
  name: string;
  path: string;
  url: string;
  updated_at?: number;
};

type ArtifactFile = {
  name: string;
  path: string;
  kind: string;
  size?: number;
  updated_at?: number;
  download_url?: string;
};

type ArtifactPayload = {
  current_deck?: { files?: ArtifactFile[]; status?: string; html_slides_count?: number };
  version_history?: Array<{ version_label?: string; files?: ArtifactFile[] }>;
  inputs?: Array<{ label?: string; files?: ArtifactFile[] }>;
  files?: ArtifactFile[];
};

type Tab = "create" | "revise" | "templates" | "memory" | "services" | "files";
type Toast = { tone: "info" | "success" | "error"; message: string };

const emptyArtifacts: ArtifactPayload = { current_deck: { files: [] }, version_history: [], inputs: [], files: [] };

function apiUrl(path: string): string {
  return path;
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(init?.headers || {}),
    },
  });
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(payload.detail || payload.message || `Request failed (${response.status})`);
  }
  return payload as T;
}

function App() {
  const [tab, setTab] = useState<Tab>("create");
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [current, setCurrent] = useState<SessionSummary | null>(null);
  const [slides, setSlides] = useState<SlideItem[]>([]);
  const [activeSlide, setActiveSlide] = useState("");
  const [artifacts, setArtifacts] = useState<ArtifactPayload>(emptyArtifacts);
  const [serviceProfiles, setServiceProfiles] = useState<ServiceProfile[]>([]);
  const [memoryProfiles, setMemoryProfiles] = useState<MemoryProfile[]>([]);
  const [templates, setTemplates] = useState<TemplateSummary[]>([]);
  const [workingMemory, setWorkingMemory] = useState<Record<string, any> | null>(null);
  const [health, setHealth] = useState<Record<string, any> | null>(null);
  const [toast, setToast] = useState<Toast | null>(null);
  const [busy, setBusy] = useState(false);

  const [instruction, setInstruction] = useState("");
  const [feedback, setFeedback] = useState("");
  const [numPages, setNumPages] = useState("8");
  const [language, setLanguage] = useState<"en" | "zh">("en");
  const [selectedService, setSelectedService] = useState("");
  const [selectedMemory, setSelectedMemory] = useState("");
  const [selectedTemplate, setSelectedTemplate] = useState("");
  const [memoryEnabled, setMemoryEnabled] = useState(true);
  const [memoryIntent, setMemoryIntent] = useState("default");

  const [serviceForm, setServiceForm] = useState(defaultServiceForm());
  const [memoryForm, setMemoryForm] = useState(defaultMemoryForm());
  const [templateName, setTemplateName] = useState("");
  const [profileJson, setProfileJson] = useState("{}");

  const attachmentsRef = useRef<HTMLInputElement | null>(null);
  const referenceTemplateRef = useRef<HTMLInputElement | null>(null);
  const inductTemplateRef = useRef<HTMLInputElement | null>(null);

  const notify = useCallback((tone: Toast["tone"], message: string) => {
    setToast({ tone, message });
    window.setTimeout(() => setToast(null), 4200);
  }, []);

  const refreshLists = useCallback(async () => {
    const [healthPayload, sessionsPayload, servicePayload, memoryPayload, templatePayload] = await Promise.all([
      jsonFetch<Record<string, any>>("/api/health"),
      jsonFetch<{ sessions: SessionSummary[] }>("/api/sessions"),
      jsonFetch<{ profiles: ServiceProfile[] }>("/api/service-profiles"),
      jsonFetch<{ profiles: MemoryProfile[] }>("/api/memory-profiles"),
      jsonFetch<{ templates: TemplateSummary[] }>("/api/templates"),
    ]);
    setHealth(healthPayload);
    setSessions(sessionsPayload.sessions || []);
    setServiceProfiles(servicePayload.profiles || []);
    setMemoryProfiles(memoryPayload.profiles || []);
    setTemplates(templatePayload.templates || []);
    if (!selectedService) {
      const preferred = (servicePayload.profiles || []).find((item) => item.is_default) || servicePayload.profiles?.[0];
      if (preferred?.profile_id) setSelectedService(preferred.profile_id);
    }
    if (!selectedMemory) {
      const preferred = (memoryPayload.profiles || []).find((item) => item.is_default) || memoryPayload.profiles?.[0];
      if (preferred?.memory_profile_id) setSelectedMemory(preferred.memory_profile_id);
    }
  }, [selectedMemory, selectedService]);

  const refreshSession = useCallback(
    async (sessionId = current?.session_id || "") => {
      if (!sessionId) return;
      const [statusPayload, slidesPayload, artifactPayload] = await Promise.all([
        jsonFetch<SessionSummary>(`/api/sessions/${sessionId}/status`),
        current?.session_id === sessionId
          ? jsonFetch<{ slides: SlideItem[] }>(`/api/sessions/${sessionId}/slides`)
          : Promise.resolve({ slides: [] as SlideItem[] }),
        current?.session_id === sessionId
          ? jsonFetch<ArtifactPayload>(`/api/sessions/${sessionId}/artifacts`)
          : Promise.resolve(emptyArtifacts),
      ]);
      setCurrent(statusPayload);
      setSlides(slidesPayload.slides || []);
      setArtifacts(artifactPayload || emptyArtifacts);
      setActiveSlide((previous) => {
        const nextSlides = slidesPayload.slides || [];
        if (previous && nextSlides.some((slide) => slide.url === previous)) return previous;
        return nextSlides[0]?.url || "";
      });
    },
    [current?.session_id],
  );

  useEffect(() => {
    refreshLists().catch((error) => notify("error", error.message));
  }, [notify, refreshLists]);

  useEffect(() => {
    if (!current?.session_id) return;
    const timer = window.setInterval(() => {
      const active = isSessionActive(current);
      refreshSession(current.session_id).catch(() => undefined);
      if (!active) refreshLists().catch(() => undefined);
    }, isSessionActive(current) ? 1800 : 5000);
    return () => window.clearInterval(timer);
  }, [current, refreshLists, refreshSession]);

  const createSession = useCallback(async () => {
    setBusy(true);
    try {
      const payload = await jsonFetch<SessionSummary>("/api/sessions", {
        method: "POST",
        body: JSON.stringify({
          language,
          service_profile_id: selectedService,
          memory_profile_id: selectedMemory,
          memory_enabled: memoryEnabled,
          memory_intent: memoryIntent,
        }),
      });
      setCurrent(payload);
      setSlides([]);
      setArtifacts(emptyArtifacts);
      setActiveSlide("");
      await refreshLists();
      notify("success", "New local session created.");
      return payload;
    } finally {
      setBusy(false);
    }
  }, [language, memoryEnabled, memoryIntent, notify, refreshLists, selectedMemory, selectedService]);

  const ensureSession = useCallback(async () => current || (await createSession()), [createSession, current]);

  const runGenerate = useCallback(async () => {
    if (!instruction.trim()) {
      notify("error", "Enter a prompt before generating.");
      return;
    }
    if (!selectedService) {
      setTab("services");
      notify("error", "Add a local service profile before generating.");
      return;
    }
    setBusy(true);
    try {
      const session = await ensureSession();
      const form = new FormData();
      form.append("instruction", instruction.trim());
      form.append("num_pages", numPages);
      form.append("language", language);
      form.append("service_profile_id", selectedService);
      form.append("memory_profile_id", selectedMemory);
      form.append("memory_enabled", String(memoryEnabled));
      form.append("memory_intent", memoryIntent);
      form.append("template_id", selectedTemplate);
      for (const file of Array.from(attachmentsRef.current?.files || [])) {
        form.append("files", file);
      }
      const templateFile = referenceTemplateRef.current?.files?.[0];
      if (templateFile) form.append("reference_template", templateFile);
      const accepted = await jsonFetch<SessionSummary>(`/api/sessions/${session.session_id}/generate`, {
        method: "POST",
        body: form,
      });
      setCurrent(accepted);
      notify("success", "Generation started.");
      await refreshSession(session.session_id);
      await refreshLists();
    } catch (error) {
      notify("error", error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }, [
    ensureSession,
    instruction,
    language,
    memoryEnabled,
    memoryIntent,
    notify,
    numPages,
    refreshLists,
    refreshSession,
    selectedMemory,
    selectedService,
    selectedTemplate,
  ]);

  const runRevise = useCallback(async () => {
    if (!current?.session_id) {
      notify("error", "Open a session before revising.");
      return;
    }
    if (!feedback.trim()) {
      notify("error", "Enter feedback before revising.");
      return;
    }
    setBusy(true);
    try {
      const accepted = await jsonFetch<SessionSummary>(`/api/sessions/${current.session_id}/revise`, {
        method: "POST",
        body: JSON.stringify({
          feedback: feedback.trim(),
          service_profile_id: selectedService,
          memory_profile_id: selectedMemory,
          memory_intent: memoryIntent,
        }),
      });
      setCurrent(accepted);
      notify("success", "Revision started.");
      await refreshSession(current.session_id);
      await refreshLists();
    } catch (error) {
      notify("error", error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }, [current?.session_id, feedback, memoryIntent, notify, refreshLists, refreshSession, selectedMemory, selectedService]);

  const openSession = useCallback(
    async (session: SessionSummary) => {
      setCurrent(session);
      setBusy(true);
      try {
        const opened = await jsonFetch<SessionSummary>("/api/sessions/open", {
          method: "POST",
          body: JSON.stringify({ session_id: session.session_id }),
        });
        setCurrent(opened);
        const [slidesPayload, artifactPayload] = await Promise.all([
          jsonFetch<{ slides: SlideItem[] }>(`/api/sessions/${session.session_id}/slides`),
          jsonFetch<ArtifactPayload>(`/api/sessions/${session.session_id}/artifacts`),
        ]);
        setSlides(slidesPayload.slides || []);
        setArtifacts(artifactPayload || emptyArtifacts);
        setActiveSlide((previous) => {
          const nextSlides = slidesPayload.slides || [];
          if (previous && nextSlides.some((slide) => slide.url === previous)) return previous;
          return nextSlides[0]?.url || "";
        });
      } catch (error) {
        notify("error", error instanceof Error ? error.message : String(error));
      } finally {
        setBusy(false);
      }
    },
    [notify, refreshSession],
  );

  const deleteSession = useCallback(
    async (sessionId: string) => {
      await jsonFetch(`/api/sessions/${sessionId}`, { method: "DELETE" });
      if (current?.session_id === sessionId) {
        setCurrent(null);
        setSlides([]);
        setArtifacts(emptyArtifacts);
        setActiveSlide("");
      }
      await refreshLists();
      notify("success", "Session deleted.");
    },
    [current?.session_id, notify, refreshLists],
  );

  const saveServiceProfile = useCallback(async () => {
    try {
      const payload = servicePayloadFromForm(serviceForm);
      const saved = await jsonFetch<ServiceProfile>("/api/service-profiles", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setSelectedService(saved.profile_id);
      setServiceForm(defaultServiceForm());
      await refreshLists();
      notify("success", "Service profile saved locally.");
    } catch (error) {
      notify("error", error instanceof Error ? error.message : String(error));
    }
  }, [notify, refreshLists, serviceForm]);

  const editServiceProfile = useCallback((profile: ServiceProfile) => {
    const services = profile.services || {};
    setServiceForm({
      profile_id: profile.profile_id,
      display_name: profile.display_name || "",
      model: serviceValue(services.llm || profile.llm, "model", profile.model || ""),
      base_url: serviceValue(services.llm || profile.llm, "base_url", profile.base_url || "https://api.openai.com/v1"),
      api_key: "",
      pdf_api_key: "",
      pdf_api_url: serviceValue(services.pdf || profile.pdf, "api_url", ""),
      embedding_enabled: Boolean((services.embedding || profile.embedding)?.enabled),
      embedding_model: serviceValue(services.embedding || profile.embedding, "model", "text-embedding-3-small"),
      embedding_base_url: serviceValue(services.embedding || profile.embedding, "base_url", "https://api.openai.com/v1"),
      embedding_api_key: "",
      image_enabled: Boolean((services.image_generation || profile.image_generation)?.enabled),
      image_model: serviceValue(services.image_generation || profile.image_generation, "model", ""),
      image_base_url: serviceValue(services.image_generation || profile.image_generation, "base_url", "https://api.openai.com/v1"),
      image_api_key: "",
      search_enabled: Boolean((services.search || profile.search)?.enabled),
      search_api_key: "",
      is_default: Boolean(profile.is_default),
      max_concurrent: String(profile.max_concurrent || 2),
    });
  }, []);

  const validateServiceProfile = useCallback(
    async (profileId: string) => {
      setBusy(true);
      try {
        const result = await jsonFetch<{ ok: boolean; message?: string }>(`/api/service-profiles/${profileId}/validate`, {
          method: "POST",
        });
        await refreshLists();
        notify(result.ok ? "success" : "error", result.message || (result.ok ? "Service profile is ready." : "Validation failed."));
      } catch (error) {
        notify("error", error instanceof Error ? error.message : String(error));
      } finally {
        setBusy(false);
      }
    },
    [notify, refreshLists],
  );

  const deleteServiceProfile = useCallback(
    async (profileId: string) => {
      await jsonFetch(`/api/service-profiles/${profileId}`, { method: "DELETE" });
      if (selectedService === profileId) setSelectedService("");
      await refreshLists();
      notify("success", "Service profile deleted.");
    },
    [notify, refreshLists, selectedService],
  );

  const saveMemoryProfile = useCallback(async () => {
    try {
      const profile = profileJson.trim() ? JSON.parse(profileJson) : {};
      const saved = await jsonFetch<MemoryProfile>("/api/memory-profiles", {
        method: "POST",
        body: JSON.stringify({ ...memoryForm, profile }),
      });
      setSelectedMemory(saved.memory_profile_id);
      setMemoryForm(defaultMemoryForm());
      setProfileJson("{}");
      await refreshLists();
      notify("success", "Memory profile saved.");
    } catch (error) {
      notify("error", error instanceof Error ? error.message : String(error));
    }
  }, [memoryForm, notify, profileJson, refreshLists]);

  const editMemoryProfile = useCallback((profile: MemoryProfile) => {
    setMemoryForm({
      memory_profile_id: profile.memory_profile_id,
      name: profile.name || "",
      intent: profile.intent || "default",
      is_default: Boolean(profile.is_default),
    });
    setProfileJson(JSON.stringify(profile.profile || {}, null, 2));
  }, []);

  const deleteMemoryProfile = useCallback(
    async (profileId: string) => {
      await jsonFetch(`/api/memory-profiles/${profileId}`, { method: "DELETE" });
      if (selectedMemory === profileId) setSelectedMemory("");
      await refreshLists();
      notify("success", "Memory profile deleted.");
    },
    [notify, refreshLists, selectedMemory],
  );

  const loadWorkingMemory = useCallback(async () => {
    if (!current?.session_id) {
      notify("error", "Open a session first.");
      return;
    }
    try {
      const payload = await jsonFetch<Record<string, any>>(`/api/sessions/${current.session_id}/memory/working`);
      setWorkingMemory(payload);
      notify("success", "Working memory loaded.");
    } catch (error) {
      notify("error", error instanceof Error ? error.message : String(error));
    }
  }, [current?.session_id, notify]);

  const inductTemplate = useCallback(async () => {
    const file = inductTemplateRef.current?.files?.[0];
    if (!file) {
      notify("error", "Choose a PPTX template first.");
      return;
    }
    setBusy(true);
    try {
      const form = new FormData();
      form.append("template_file", file);
      form.append("output_name", templateName || file.name.replace(/\.pptx$/i, ""));
      const result = await jsonFetch<Record<string, any>>("/api/templates/induct", { method: "POST", body: form });
      await refreshLists();
      notify("success", `Template ready: ${result.template_id || "template"}`);
    } catch (error) {
      notify("error", error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }, [notify, refreshLists, templateName]);

  const currentDownloads = useMemo(() => {
    const live = artifacts.current_deck?.files || [];
    const versions = (artifacts.version_history || []).flatMap((group) => group.files || []);
    return [...live, ...versions].filter((item) => item.download_url && ["pptx", "pdf"].includes(item.kind));
  }, [artifacts]);

  const sessionStatus = current ? statusLabel(current) : "No session";

  return (
    <div className="app-shell">
      <aside className="left-panel">
        <div className="brand">
          <div className="brand-mark">M</div>
          <div>
            <h1>MemSlides</h1>
            <span>Local Studio</span>
          </div>
        </div>

        <nav className="tabs" aria-label="Studio sections">
          <TabButton active={tab === "create"} icon={<Sparkles size={16} />} label="Create" onClick={() => setTab("create")} />
          <TabButton active={tab === "revise"} icon={<Wand2 size={16} />} label="Revise" onClick={() => setTab("revise")} />
          <TabButton active={tab === "templates"} icon={<Layers3 size={16} />} label="Templates" onClick={() => setTab("templates")} />
          <TabButton active={tab === "memory"} icon={<MemoryStick size={16} />} label="Memory" onClick={() => setTab("memory")} />
          <TabButton active={tab === "services"} icon={<KeyRound size={16} />} label="Services" onClick={() => setTab("services")} />
          <TabButton active={tab === "files"} icon={<FolderOpen size={16} />} label="Files" onClick={() => setTab("files")} />
        </nav>

        <section className="panel-scroll">
          {tab === "create" && (
            <div className="stack">
              <Field label="Prompt">
                <textarea rows={8} value={instruction} onChange={(event) => setInstruction(event.target.value)} placeholder="Topic, audience, tone, source material..." />
              </Field>
              <div className="two">
                <Field label="Slides">
                  <input value={numPages} onChange={(event) => setNumPages(event.target.value)} inputMode="numeric" />
                </Field>
                <Field label="Language">
                  <select value={language} onChange={(event) => setLanguage(event.target.value as "en" | "zh")}>
                    <option value="en">English</option>
                    <option value="zh">Chinese</option>
                  </select>
                </Field>
              </div>
              <Selector label="Service" value={selectedService} onChange={setSelectedService} empty="Add service profile" items={serviceProfiles.map((item) => ({ value: item.profile_id, label: serviceLabel(item) }))} />
              <Selector label="Memory profile" value={selectedMemory} onChange={setSelectedMemory} empty="No memory profile" items={memoryProfiles.map((item) => ({ value: item.memory_profile_id, label: `${item.name || item.intent || item.memory_profile_id}` }))} />
              <Selector label="Template" value={selectedTemplate} onChange={setSelectedTemplate} empty="No template" items={templates.map((item) => ({ value: templateId(item), label: templateLabel(item) }))} />
              <Field label="Memory intent">
                <input value={memoryIntent} onChange={(event) => setMemoryIntent(event.target.value)} />
              </Field>
              <label className="toggle-row">
                <input type="checkbox" checked={memoryEnabled} onChange={(event) => setMemoryEnabled(event.target.checked)} />
                <span>Memory enabled</span>
              </label>
              <FileInput label="Files" helper="PDF, PPTX, images, Markdown, or text" inputRef={attachmentsRef} multiple />
              <FileInput label="Reference PPTX" helper="Optional style reference" inputRef={referenceTemplateRef} accept=".pptx" />
              <div className="action-row">
                <button className="secondary" onClick={createSession} disabled={busy}>
                  <Plus size={16} /> New
                </button>
                <button className="primary" onClick={runGenerate} disabled={busy}>
                  {busy ? <Loader2 className="spin" size={16} /> : <Send size={16} />} Generate
                </button>
              </div>
            </div>
          )}

          {tab === "revise" && (
            <div className="stack">
              <InfoLine icon={<FileText size={16} />} title={current?.display_name || "No session open"} detail={current?.session_id || "Open a recent session first"} />
              <Field label="Feedback">
                <textarea rows={10} value={feedback} onChange={(event) => setFeedback(event.target.value)} placeholder="Describe what should change in the current deck..." />
              </Field>
              <button className="primary full" onClick={runRevise} disabled={busy || !current}>
                {busy ? <Loader2 className="spin" size={16} /> : <Wand2 size={16} />} Apply revision
              </button>
            </div>
          )}

          {tab === "templates" && (
            <div className="stack">
              <FileInput label="Upload PPTX" helper="Create a reusable template profile" inputRef={inductTemplateRef} accept=".pptx" />
              <Field label="Template name">
                <input value={templateName} onChange={(event) => setTemplateName(event.target.value)} placeholder="Optional output name" />
              </Field>
              <button className="primary full" onClick={inductTemplate} disabled={busy}>
                {busy ? <Loader2 className="spin" size={16} /> : <Upload size={16} />} Induct template
              </button>
              <List title="Templates" count={templates.length}>
                {templates.map((item) => (
                  <button key={templateId(item)} className={`list-card ${selectedTemplate === templateId(item) ? "active" : ""}`} onClick={() => setSelectedTemplate(templateId(item))}>
                    <strong>{templateLabel(item)}</strong>
                    <span>{[item.slide_count && `${item.slide_count} slides`, item.layout_count && `${item.layout_count} layouts`, item.image_count && `${item.image_count} images`].filter(Boolean).join(" · ") || "Ready"}</span>
                  </button>
                ))}
              </List>
            </div>
          )}

          {tab === "memory" && (
            <div className="stack">
              <div className="two">
                <Field label="Name">
                  <input value={memoryForm.name} onChange={(event) => setMemoryForm({ ...memoryForm, name: event.target.value })} />
                </Field>
                <Field label="Intent">
                  <input value={memoryForm.intent} onChange={(event) => setMemoryForm({ ...memoryForm, intent: event.target.value })} />
                </Field>
              </div>
              <Field label="Profile JSON">
                <textarea rows={8} value={profileJson} onChange={(event) => setProfileJson(event.target.value)} />
              </Field>
              <label className="toggle-row">
                <input type="checkbox" checked={memoryForm.is_default} onChange={(event) => setMemoryForm({ ...memoryForm, is_default: event.target.checked })} />
                <span>Default memory profile</span>
              </label>
              <button className="primary full" onClick={saveMemoryProfile}>
                <Save size={16} /> Save memory profile
              </button>
              <button className="secondary full" onClick={loadWorkingMemory} disabled={!current}>
                <Eye size={16} /> Load working memory
              </button>
              <List title="Memory profiles" count={memoryProfiles.length}>
                {memoryProfiles.map((item) => (
                  <article key={item.memory_profile_id} className={`list-card ${selectedMemory === item.memory_profile_id ? "active" : ""}`}>
                    <button onClick={() => setSelectedMemory(item.memory_profile_id)}>
                      <strong>{item.name || item.intent || item.memory_profile_id}</strong>
                      <span>{item.intent || "default"}</span>
                    </button>
                    <div className="mini-actions">
                      <button onClick={() => editMemoryProfile(item)}>Edit</button>
                      <button onClick={() => deleteMemoryProfile(item.memory_profile_id)}><Trash2 size={14} /></button>
                    </div>
                  </article>
                ))}
              </List>
            </div>
          )}

          {tab === "services" && (
            <div className="stack">
              <div className="two">
                <Field label="Name">
                  <input value={serviceForm.display_name} onChange={(event) => setServiceForm({ ...serviceForm, display_name: event.target.value })} />
                </Field>
                <Field label="Model">
                  <input value={serviceForm.model} onChange={(event) => setServiceForm({ ...serviceForm, model: event.target.value })} />
                </Field>
              </div>
              <Field label="Base URL">
                <input value={serviceForm.base_url} onChange={(event) => setServiceForm({ ...serviceForm, base_url: event.target.value })} />
              </Field>
              <Field label="LLM key">
                <input type="password" value={serviceForm.api_key} onChange={(event) => setServiceForm({ ...serviceForm, api_key: event.target.value })} placeholder={serviceForm.profile_id ? "Leave blank to keep saved key" : "Paste key locally"} />
              </Field>
              <Field label="PDF parser key or URL">
                <input type="password" value={serviceForm.pdf_api_key} onChange={(event) => setServiceForm({ ...serviceForm, pdf_api_key: event.target.value })} placeholder={serviceForm.profile_id ? "Leave blank to keep saved key" : "MinerU key"} />
                <input value={serviceForm.pdf_api_url} onChange={(event) => setServiceForm({ ...serviceForm, pdf_api_url: event.target.value })} placeholder="Optional compatible endpoint" />
              </Field>
              <details>
                <summary><Settings2 size={15} /> Optional services</summary>
                <label className="toggle-row">
                  <input type="checkbox" checked={serviceForm.embedding_enabled} onChange={(event) => setServiceForm({ ...serviceForm, embedding_enabled: event.target.checked })} />
                  <span>Embedding endpoint</span>
                </label>
                <Field label="Embedding model">
                  <input value={serviceForm.embedding_model} onChange={(event) => setServiceForm({ ...serviceForm, embedding_model: event.target.value })} />
                </Field>
                <Field label="Embedding base URL">
                  <input value={serviceForm.embedding_base_url} onChange={(event) => setServiceForm({ ...serviceForm, embedding_base_url: event.target.value })} />
                </Field>
                <Field label="Embedding key">
                  <input type="password" value={serviceForm.embedding_api_key} onChange={(event) => setServiceForm({ ...serviceForm, embedding_api_key: event.target.value })} />
                </Field>
                <label className="toggle-row">
                  <input type="checkbox" checked={serviceForm.image_enabled} onChange={(event) => setServiceForm({ ...serviceForm, image_enabled: event.target.checked })} />
                  <span>Image generation</span>
                </label>
                <Field label="Image model">
                  <input value={serviceForm.image_model} onChange={(event) => setServiceForm({ ...serviceForm, image_model: event.target.value })} />
                </Field>
                <Field label="Image base URL">
                  <input value={serviceForm.image_base_url} onChange={(event) => setServiceForm({ ...serviceForm, image_base_url: event.target.value })} />
                </Field>
                <Field label="Image key">
                  <input type="password" value={serviceForm.image_api_key} onChange={(event) => setServiceForm({ ...serviceForm, image_api_key: event.target.value })} />
                </Field>
                <label className="toggle-row">
                  <input type="checkbox" checked={serviceForm.search_enabled} onChange={(event) => setServiceForm({ ...serviceForm, search_enabled: event.target.checked })} />
                  <span>Web search</span>
                </label>
                <Field label="Search key">
                  <input type="password" value={serviceForm.search_api_key} onChange={(event) => setServiceForm({ ...serviceForm, search_api_key: event.target.value })} />
                </Field>
              </details>
              <label className="toggle-row">
                <input type="checkbox" checked={serviceForm.is_default} onChange={(event) => setServiceForm({ ...serviceForm, is_default: event.target.checked })} />
                <span>Default service profile</span>
              </label>
              <button className="primary full" onClick={saveServiceProfile}>
                <Save size={16} /> Save service profile
              </button>
              <List title="Service profiles" count={serviceProfiles.length}>
                {serviceProfiles.map((item) => (
                  <article key={item.profile_id} className={`list-card ${selectedService === item.profile_id ? "active" : ""}`}>
                    <button onClick={() => setSelectedService(item.profile_id)}>
                      <strong>{item.display_name || item.profile_id}</strong>
                      <span>{serviceLabel(item)}</span>
                    </button>
                    <div className="mini-actions">
                      <button onClick={() => editServiceProfile(item)}>Edit</button>
                      <button onClick={() => validateServiceProfile(item.profile_id)}><CheckCircle2 size={14} /></button>
                      <button onClick={() => deleteServiceProfile(item.profile_id)}><Trash2 size={14} /></button>
                    </div>
                  </article>
                ))}
              </List>
            </div>
          )}

          {tab === "files" && (
            <div className="stack">
              <InfoLine icon={<FileArchive size={16} />} title="Downloads" detail={`${currentDownloads.length} current deck files`} />
              <ArtifactList files={currentDownloads} />
              <List title="Recent sessions" count={sessions.length}>
                {sessions.map((item) => (
                  <article key={item.session_id} className={`list-card ${current?.session_id === item.session_id ? "active" : ""}`}>
                    <button onClick={() => openSession(item)}>
                      <strong>{item.display_name || item.instruction_summary || item.session_id}</strong>
                      <span>{statusLabel(item)} · {formatTime(item.updated_at)}</span>
                    </button>
                    <div className="mini-actions">
                      <button onClick={() => deleteSession(item.session_id)}><Trash2 size={14} /></button>
                    </div>
                  </article>
                ))}
              </List>
            </div>
          )}
        </section>
      </aside>

      <main className="preview-panel">
        <header className="topbar">
          <div>
            <span className="eyebrow">Studio</span>
            <h2>{current?.display_name || "Ready for a local demo"}</h2>
          </div>
          <div className="topbar-actions">
            <button className="secondary" onClick={() => refreshSession()} disabled={!current}>
              <RefreshCw size={16} /> Refresh
            </button>
            {currentDownloads[0]?.download_url ? (
              <a className="primary link-button" href={currentDownloads[0].download_url} download>
                <Download size={16} /> Download
              </a>
            ) : null}
            <span className={`status-pill ${isSessionActive(current) ? "running" : ""}`}>{sessionStatus}</span>
          </div>
        </header>

        <section className="slide-stage">
          {activeSlide ? (
            <iframe title="Slide preview" src={activeSlide} />
          ) : (
            <div className="empty-stage">
              <Sparkles size={36} />
              <strong>{selectedService ? "Create a session and generate a deck" : "Add a service profile to begin"}</strong>
              <span>{health?.workspace_base ? `Workspace: ${health.workspace_base}` : "Local single-user Studio"}</span>
            </div>
          )}
        </section>

        <footer className="slide-strip">
          {slides.length ? (
            slides.map((slide, index) => (
              <button key={slide.path} className={activeSlide === slide.url ? "active" : ""} onClick={() => setActiveSlide(slide.url)}>
                <span>{index + 1}</span>
                <strong>{slide.name}</strong>
              </button>
            ))
          ) : (
            <span>No slide preview yet</span>
          )}
        </footer>
      </main>

      <aside className="right-panel">
        <section className="card">
          <div className="card-head">
            <strong>Current Session</strong>
            <span>{current?.session_id || "none"}</span>
          </div>
          <dl className="kv">
            <div><dt>Status</dt><dd>{sessionStatus}</dd></div>
            <div><dt>Slides</dt><dd>{slides.length || current?.slide_count || 0}</dd></div>
            <div><dt>Artifacts</dt><dd>{currentDownloads.length}</dd></div>
            <div><dt>Memory</dt><dd>{selectedMemory || memoryIntent}</dd></div>
          </dl>
        </section>

        <section className="card grow">
          <div className="card-head">
            <strong>Artifacts</strong>
            <span>{artifacts.current_deck?.status || "waiting"}</span>
          </div>
          <ArtifactList files={currentDownloads.slice(0, 8)} />
          {(artifacts.inputs || []).length ? (
            <div className="input-groups">
              {(artifacts.inputs || []).map((group) => (
                <div key={group.label}>
                  <strong>{group.label}</strong>
                  {(group.files || []).slice(0, 4).map((file) => (
                    <a key={file.path} href={file.download_url || "#"}>{file.name}</a>
                  ))}
                </div>
              ))}
            </div>
          ) : null}
        </section>

        <section className="card memory-card">
          <div className="card-head">
            <strong>Working Memory</strong>
            <button onClick={loadWorkingMemory} disabled={!current}>Load</button>
          </div>
          <pre>{workingMemory ? JSON.stringify(workingMemory, null, 2) : "No live snapshot loaded."}</pre>
        </section>
      </aside>

      {toast ? <div className={`toast ${toast.tone}`}>{toast.message}</div> : null}
    </div>
  );
}

function Field(props: { label: string; children: React.ReactNode }) {
  return (
    <label className="field">
      <span>{props.label}</span>
      {props.children}
    </label>
  );
}

function Selector(props: { label: string; value: string; onChange: (value: string) => void; empty: string; items: Array<{ value: string; label: string }> }) {
  return (
    <Field label={props.label}>
      <select value={props.value} onChange={(event) => props.onChange(event.target.value)}>
        <option value="">{props.empty}</option>
        {props.items.map((item) => (
          <option key={item.value} value={item.value}>{item.label}</option>
        ))}
      </select>
    </Field>
  );
}

function TabButton(props: { active: boolean; icon: React.ReactNode; label: string; onClick: () => void }) {
  return (
    <button className={props.active ? "active" : ""} onClick={props.onClick}>
      {props.icon}
      <span>{props.label}</span>
    </button>
  );
}

function FileInput(props: { label: string; helper: string; inputRef: React.RefObject<HTMLInputElement | null>; accept?: string; multiple?: boolean }) {
  return (
    <label className="file-input">
      <input ref={props.inputRef} type="file" accept={props.accept} multiple={props.multiple} />
      <Upload size={16} />
      <span>
        <strong>{props.label}</strong>
        <small>{props.helper}</small>
      </span>
    </label>
  );
}

function InfoLine(props: { icon: React.ReactNode; title: string; detail: string }) {
  return (
    <div className="info-line">
      {props.icon}
      <span>
        <strong>{props.title}</strong>
        <small>{props.detail}</small>
      </span>
    </div>
  );
}

function List(props: { title: string; count: number; children: React.ReactNode }) {
  return (
    <section className="list">
      <div className="list-head">
        <strong>{props.title}</strong>
        <span>{props.count}</span>
      </div>
      {props.count ? props.children : <div className="empty-list">Nothing here yet.</div>}
    </section>
  );
}

function ArtifactList(props: { files: ArtifactFile[] }) {
  if (!props.files.length) return <div className="empty-list">No downloadable artifacts yet.</div>;
  return (
    <div className="artifact-list">
      {props.files.map((file) => (
        <a key={`${file.path}:${file.kind}`} href={file.download_url || "#"} download>
          <Download size={14} />
          <span>{file.name}</span>
          <small>{file.kind.toUpperCase()}</small>
        </a>
      ))}
    </div>
  );
}

function defaultServiceForm() {
  return {
    profile_id: "",
    display_name: "Local OpenAI-compatible",
    model: "gpt-4.1",
    base_url: "https://api.openai.com/v1",
    api_key: "",
    pdf_api_key: "",
    pdf_api_url: "",
    embedding_enabled: false,
    embedding_model: "text-embedding-3-small",
    embedding_base_url: "https://api.openai.com/v1",
    embedding_api_key: "",
    image_enabled: false,
    image_model: "",
    image_base_url: "https://api.openai.com/v1",
    image_api_key: "",
    search_enabled: false,
    search_api_key: "",
    is_default: true,
    max_concurrent: "2",
  };
}

function defaultMemoryForm() {
  return { memory_profile_id: "", name: "Demo profile", intent: "default", is_default: true };
}

function servicePayloadFromForm(form: ReturnType<typeof defaultServiceForm>) {
  return {
    profile_id: form.profile_id,
    display_name: form.display_name,
    max_concurrent: Number(form.max_concurrent || 2),
    enabled: true,
    is_default: form.is_default,
    llm: {
      provider: "openai_compatible",
      base_url: form.base_url,
      model: form.model,
      api_key: form.api_key,
      enabled: true,
      vision_capable: true,
    },
    pdf: {
      provider: form.pdf_api_url.trim() ? "mineru_compatible" : "mineru_official",
      api_key: form.pdf_api_key,
      api_url: form.pdf_api_url,
      enabled: true,
    },
    embedding: {
      provider: "openai_compatible",
      base_url: form.embedding_base_url,
      model: form.embedding_model,
      api_key: form.embedding_api_key,
      enabled: form.embedding_enabled,
    },
    search: {
      provider: "tavily",
      api_key: form.search_api_key,
      enabled: form.search_enabled,
    },
    image_generation: {
      provider: "openai_compatible",
      base_url: form.image_base_url,
      model: form.image_model,
      api_key: form.image_api_key,
      enabled: form.image_enabled,
    },
  };
}

function serviceValue(service: Record<string, any> | undefined, key: string, fallback: string) {
  return String(service?.[key] || fallback || "");
}

function serviceLabel(profile: ServiceProfile) {
  const model = profile.model || profile.services?.llm?.model || profile.llm?.model || "model";
  const status = profile.required_ready ? "ready" : profile.validation_status || "setup";
  return `${profile.display_name || profile.profile_id} · ${model} · ${status}`;
}

function templateId(template: TemplateSummary) {
  return String(template.template_id || template.id || "");
}

function templateLabel(template: TemplateSummary) {
  return String(template.name || template.template_id || template.id || "Template");
}

function isSessionActive(session: SessionSummary | null) {
  if (!session) return false;
  const values = [session.state, session.phase, session.queued_operation, session.last_operation?.state, session.last_operation?.phase]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return /running|queued|cancelling/.test(values);
}

function statusLabel(session: SessionSummary) {
  const state = String(session.state || "idle");
  const phase = String(session.phase || "");
  if (session.message) return `${state} · ${session.message}`;
  return phase ? `${state} · ${phase}` : state;
}

function formatTime(value?: string) {
  if (!value) return "never";
  const time = Date.parse(value);
  return Number.isFinite(time) ? new Date(time).toLocaleString() : value;
}

export default App;
