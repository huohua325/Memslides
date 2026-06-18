-- Memory System Schema
-- 合并自 schema.sql + cognitive_schema.sql (Stage B 整合)

-- ═══════════════════════════════════════════════════════════════
-- Stage 12: Job/Round 执行层级表
-- ═══════════════════════════════════════════════════════════════

-- Job 表（一次多轮对话会话）
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    started_at TEXT NOT NULL,
    ended_at TEXT,
    intent TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(user_id, project_id);

-- Round 表（兼容层暂沿用历史 tasks 表名）
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    user_message TEXT NOT NULL,
    agent_response TEXT DEFAULT '',
    started_at TEXT NOT NULL,
    ended_at TEXT,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);
CREATE INDEX IF NOT EXISTS idx_tasks_job ON tasks(job_id);

-- UserProfile 表（用户偏好画像，按 intent 分区）
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT NOT NULL,
    intent TEXT NOT NULL DEFAULT 'default',
    profile_json TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    last_updated TEXT NOT NULL,
    PRIMARY KEY (user_id, intent)
);

-- UserCoreProfile 表（用户稳定 persona 与跨任务稳定偏好）
CREATE TABLE IF NOT EXISTS user_core_profiles (
    user_id TEXT PRIMARY KEY,
    core_persona TEXT NOT NULL DEFAULT '',
    profile_json TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    last_updated TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_user_core_profiles_persona
ON user_core_profiles(core_persona);

-- 会话表
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    status TEXT DEFAULT 'active',       -- active/ended/expired
    created_at TEXT NOT NULL,
    ended_at TEXT,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(user_id, project_id);

-- 经验轨迹表
CREATE TABLE IF NOT EXISTS experience_traces (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    task_description TEXT NOT NULL,
    reasoning_steps TEXT NOT NULL DEFAULT '[]',
    tools_used TEXT DEFAULT '[]',
    final_outcome TEXT NOT NULL DEFAULT '',
    lessons_learned TEXT DEFAULT '',
    applicable_scenarios TEXT DEFAULT '[]',
    confidence REAL DEFAULT 0.7,
    reuse_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active',
    superseded_by TEXT DEFAULT '',
    superseded_at TEXT DEFAULT '',
    merged_from_ids TEXT DEFAULT '[]',
    source_types TEXT DEFAULT '[]',
    consolidation_version TEXT DEFAULT '',
    template_id TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    -- Stage 12: 链级扩展字段（零损失转换）
    experience_type TEXT DEFAULT 'generic',
    chain_name TEXT DEFAULT '',
    tool_sequence TEXT DEFAULT '[]',
    anti_pattern TEXT DEFAULT '',
    applicable_when TEXT DEFAULT '',
    source_chain_ids TEXT DEFAULT '[]',
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_experience_session ON experience_traces(session_id);
CREATE INDEX IF NOT EXISTS idx_experience_outcome ON experience_traces(final_outcome);
CREATE INDEX IF NOT EXISTS idx_experience_type ON experience_traces(experience_type);
CREATE INDEX IF NOT EXISTS idx_experience_status ON experience_traces(status);
CREATE INDEX IF NOT EXISTS idx_experience_type_status ON experience_traces(experience_type, status);
CREATE INDEX IF NOT EXISTS idx_experience_chain_name ON experience_traces(chain_name);

-- experience_traces FTS5 全文索引 (Stage 3, 改造 3.5 + Stage 12 扩展)
CREATE VIRTUAL TABLE IF NOT EXISTS experience_traces_fts USING fts5(
    task_description, lessons_learned, applicable_scenarios, anti_pattern, applicable_when,
    content='experience_traces', content_rowid='rowid',
    tokenize='trigram'
);

-- experience_traces FTS 同步触发器
CREATE TRIGGER IF NOT EXISTS experience_traces_fts_ai AFTER INSERT ON experience_traces BEGIN
    INSERT INTO experience_traces_fts(rowid, task_description, lessons_learned, applicable_scenarios, anti_pattern, applicable_when)
    VALUES (new.rowid, new.task_description, new.lessons_learned, new.applicable_scenarios, new.anti_pattern, new.applicable_when);
END;
CREATE TRIGGER IF NOT EXISTS experience_traces_fts_ad AFTER DELETE ON experience_traces BEGIN
    INSERT INTO experience_traces_fts(experience_traces_fts, rowid, task_description, lessons_learned, applicable_scenarios)
    VALUES ('delete', old.rowid, old.task_description, old.lessons_learned, old.applicable_scenarios);
END;
CREATE TRIGGER IF NOT EXISTS experience_traces_fts_au AFTER UPDATE ON experience_traces BEGIN
    INSERT INTO experience_traces_fts(experience_traces_fts, rowid, task_description, lessons_learned, applicable_scenarios)
    VALUES ('delete', old.rowid, old.task_description, old.lessons_learned, old.applicable_scenarios);
    INSERT INTO experience_traces_fts(rowid, task_description, lessons_learned, applicable_scenarios)
    VALUES (new.rowid, new.task_description, new.lessons_learned, new.applicable_scenarios);
END;

-- SessionSnapshot表（跨Session续接）
CREATE TABLE IF NOT EXISTS session_snapshots (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    focus_context TEXT DEFAULT '{}',
    session_rules_summary TEXT DEFAULT '',
    last_episode_summary TEXT DEFAULT '',
    modification_summary TEXT DEFAULT '',
    unfinished_items TEXT DEFAULT '[]',
    total_episodes INTEGER DEFAULT 0,
    total_edit_segments INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_snapshots_user_project ON session_snapshots(user_id, project_id);

-- 满意度记录表
CREATE TABLE IF NOT EXISTS satisfaction_records (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    modification_id TEXT NOT NULL,
    level TEXT DEFAULT 'unknown',
    confidence REAL DEFAULT 0.0,
    signals TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_satisfaction_session ON satisfaction_records(session_id);
CREATE INDEX IF NOT EXISTS idx_satisfaction_modification ON satisfaction_records(modification_id);

-- ═══════════════════════════════════════════════════════════════
-- 认知记忆 Schema (原 cognitive_schema.sql, Stage B 合并)
--
-- 活跃表: design_episodes, atomic_preferences
-- ═══════════════════════════════════════════════════════════════

-- 设计情景记忆 (Episodic Memory)
CREATE TABLE IF NOT EXISTS design_episodes (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    source_round_id INTEGER DEFAULT 0,
    user_intent TEXT DEFAULT '',
    interpretation_gap TEXT DEFAULT '',
    action_outcome TEXT DEFAULT '',
    design_insight TEXT DEFAULT '',
    category TEXT DEFAULT '',
    confidence REAL DEFAULT 0.7,
    status TEXT DEFAULT 'active',
    used_for_profile_update INTEGER DEFAULT 0,
    used_for_rule_induction INTEGER DEFAULT 0,
    context TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_design_episodes_user ON design_episodes(user_id);
CREATE INDEX IF NOT EXISTS idx_design_episodes_session ON design_episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_design_episodes_status ON design_episodes(status);

CREATE VIRTUAL TABLE IF NOT EXISTS design_episodes_fts USING fts5(
    user_intent, design_insight, content='design_episodes', content_rowid='rowid',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS design_episodes_fts_ai AFTER INSERT ON design_episodes BEGIN
    INSERT INTO design_episodes_fts(rowid, user_intent, design_insight)
    VALUES (new.rowid, new.user_intent, new.design_insight);
END;
CREATE TRIGGER IF NOT EXISTS design_episodes_fts_ad AFTER DELETE ON design_episodes BEGIN
    INSERT INTO design_episodes_fts(design_episodes_fts, rowid, user_intent, design_insight)
    VALUES ('delete', old.rowid, old.user_intent, old.design_insight);
END;
CREATE TRIGGER IF NOT EXISTS design_episodes_fts_au AFTER UPDATE ON design_episodes BEGIN
    INSERT INTO design_episodes_fts(design_episodes_fts, rowid, user_intent, design_insight)
    VALUES ('delete', old.rowid, old.user_intent, old.design_insight);
    INSERT INTO design_episodes_fts(rowid, user_intent, design_insight)
    VALUES (new.rowid, new.user_intent, new.design_insight);
END;

-- AtomicPreference — 原子偏好 (Stage 4 主力)
CREATE TABLE IF NOT EXISTS atomic_preferences (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    preference_type TEXT DEFAULT 'value',
    trigger TEXT DEFAULT '',
    preference TEXT NOT NULL,
    rationale TEXT DEFAULT '',
    scope TEXT DEFAULT 'global',
    scope_value TEXT DEFAULT '',
    source_job_id TEXT DEFAULT '',
    source_session_id TEXT DEFAULT '',
    evidence_episode_ids TEXT DEFAULT '[]',
    confidence REAL DEFAULT 0.5,
    verified_count INTEGER DEFAULT 0,
    contradiction_count INTEGER DEFAULT 0,
    conflict_group TEXT DEFAULT '',
    status TEXT DEFAULT 'active',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_atomic_preferences_user_id ON atomic_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_atomic_preferences_status ON atomic_preferences(status);
CREATE INDEX IF NOT EXISTS idx_atomic_preferences_scope ON atomic_preferences(scope, scope_value);
CREATE INDEX IF NOT EXISTS idx_atomic_preferences_conflict_group ON atomic_preferences(conflict_group);
CREATE INDEX IF NOT EXISTS idx_atomic_preferences_source_session ON atomic_preferences(source_session_id);

CREATE VIRTUAL TABLE IF NOT EXISTS atomic_preferences_fts USING fts5(
    trigger,
    preference,
    rationale,
    content='atomic_preferences',
    content_rowid='rowid',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS atomic_preferences_fts_ai AFTER INSERT ON atomic_preferences BEGIN
    INSERT INTO atomic_preferences_fts(rowid, trigger, preference, rationale)
    VALUES (new.rowid, new."trigger", new.preference, new.rationale);
END;

CREATE TRIGGER IF NOT EXISTS atomic_preferences_fts_ad AFTER DELETE ON atomic_preferences BEGIN
    INSERT INTO atomic_preferences_fts(atomic_preferences_fts, rowid, trigger, preference, rationale)
    VALUES('delete', old.rowid, old."trigger", old.preference, old.rationale);
END;

CREATE TRIGGER IF NOT EXISTS atomic_preferences_fts_au AFTER UPDATE ON atomic_preferences BEGIN
    INSERT INTO atomic_preferences_fts(atomic_preferences_fts, rowid, trigger, preference, rationale)
    VALUES('delete', old.rowid, old."trigger", old.preference, old.rationale);
    INSERT INTO atomic_preferences_fts(rowid, trigger, preference, rationale)
    VALUES (new.rowid, new."trigger", new.preference, new.rationale);
END;

-- ═══════════════════════════════════════════════════════════════
-- 模板档案 (Template Profiles, Stage 5)
-- 表名 design_skills 为历史遗留，存储 TemplateProfile 数据
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS design_skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    category TEXT DEFAULT 'layout',
    skill_type TEXT DEFAULT 'template_layout',
    triggers TEXT DEFAULT '[]',
    keywords TEXT DEFAULT '[]',
    template_source TEXT DEFAULT '',
    slide_count INTEGER DEFAULT 0,
    aspect_ratio TEXT DEFAULT '16:9',
    slide_induction TEXT DEFAULT '{}',
    semantic_model TEXT DEFAULT '{}',
    template_dir TEXT DEFAULT '',
    image_stats TEXT DEFAULT '{}',
    design_constraints TEXT DEFAULT '{}',
    content_patterns TEXT DEFAULT '{}',
    confidence REAL DEFAULT 0.8,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    user_id TEXT NOT NULL,
    status TEXT DEFAULT 'active'
);

CREATE INDEX IF NOT EXISTS idx_skills_template_source ON design_skills(template_source);
CREATE INDEX IF NOT EXISTS idx_skills_user_id ON design_skills(user_id);
CREATE INDEX IF NOT EXISTS idx_skills_skill_type ON design_skills(skill_type);
CREATE INDEX IF NOT EXISTS idx_skills_aspect_ratio ON design_skills(aspect_ratio);

CREATE VIRTUAL TABLE IF NOT EXISTS design_skills_fts USING fts5(
    id UNINDEXED,
    name,
    description,
    keywords,
    template_source,
    content='design_skills',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS design_skills_ai AFTER INSERT ON design_skills BEGIN
    INSERT OR IGNORE INTO design_skills_fts(rowid, id, name, description, keywords, template_source)
    VALUES (NEW.rowid, NEW.id, NEW.name, COALESCE(NEW.description,''), COALESCE(NEW.keywords,''), COALESCE(NEW.template_source,''));
END;

CREATE TRIGGER IF NOT EXISTS design_skills_ad AFTER DELETE ON design_skills BEGIN
    INSERT INTO design_skills_fts(design_skills_fts, rowid, id, name, description, keywords, template_source)
    VALUES ('delete', OLD.rowid, OLD.id, OLD.name, COALESCE(OLD.description,''), COALESCE(OLD.keywords,''), COALESCE(OLD.template_source,''));
END;

CREATE TRIGGER IF NOT EXISTS design_skills_au AFTER UPDATE ON design_skills BEGIN
    INSERT INTO design_skills_fts(design_skills_fts, rowid, id, name, description, keywords, template_source)
    VALUES ('delete', OLD.rowid, OLD.id, OLD.name, COALESCE(OLD.description,''), COALESCE(OLD.keywords,''), COALESCE(OLD.template_source,''));
    INSERT OR IGNORE INTO design_skills_fts(rowid, id, name, description, keywords, template_source)
    VALUES (NEW.rowid, NEW.id, NEW.name, COALESCE(NEW.description,''), COALESCE(NEW.keywords,''), COALESCE(NEW.template_source,''));
END;


-- ═══════════════════════════════════════════════════════════════
-- 工具链存储 (Tool Chain Store, 工具记忆重构)
-- chain_raw_data: 原始链数据（完整 cycle 数据）
-- chain_experiences: 提炼经验（按链签名分桶，subkey 区分多条变体）
-- ═══════════════════════════════════════════════════════════════

-- 原始链数据表
CREATE TABLE IF NOT EXISTS chain_raw_data (
    id TEXT PRIMARY KEY,
    chain_signature TEXT NOT NULL,
    user_id TEXT NOT NULL,
    tool_chain_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    job_id TEXT DEFAULT ''
);

-- 提炼经验表（同一签名可存多条经验，通过 subkey 区分）
CREATE TABLE IF NOT EXISTS chain_experiences (
    id TEXT PRIMARY KEY,
    chain_signature TEXT NOT NULL,
    user_id TEXT NOT NULL,
    experience_json TEXT NOT NULL,
    raw_chain_count INTEGER DEFAULT 0,
    last_updated TEXT NOT NULL,
    subkey TEXT DEFAULT '',
    keyword_embedding_json TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_chain_raw_signature ON chain_raw_data(chain_signature);
CREATE INDEX IF NOT EXISTS idx_chain_raw_user ON chain_raw_data(user_id);
CREATE INDEX IF NOT EXISTS idx_chain_exp_signature ON chain_experiences(chain_signature);
CREATE INDEX IF NOT EXISTS idx_chain_exp_user ON chain_experiences(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chain_exp_unique ON chain_experiences(chain_signature, subkey);
