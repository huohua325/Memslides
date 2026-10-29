-- Stage 12: 为 atomic_preferences 添加来源追踪字段
-- 支持按 Job/Session 追溯偏好来源

-- 添加新字段
ALTER TABLE atomic_preferences ADD COLUMN source_job_id TEXT DEFAULT '';
ALTER TABLE atomic_preferences ADD COLUMN source_session_id TEXT DEFAULT '';

-- 添加索引
CREATE INDEX IF NOT EXISTS idx_atomic_preferences_source_session ON atomic_preferences(source_session_id);
