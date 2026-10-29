-- Stage 12: 为 experience_traces 添加链级扩展字段
-- 支持 ChainExperience → ExperienceTrace 的零损失转换

-- 添加新字段
ALTER TABLE experience_traces ADD COLUMN experience_type TEXT DEFAULT 'generic';
ALTER TABLE experience_traces ADD COLUMN chain_name TEXT DEFAULT '';
ALTER TABLE experience_traces ADD COLUMN tool_sequence TEXT DEFAULT '[]';
ALTER TABLE experience_traces ADD COLUMN anti_pattern TEXT DEFAULT '';
ALTER TABLE experience_traces ADD COLUMN applicable_when TEXT DEFAULT '';
ALTER TABLE experience_traces ADD COLUMN source_chain_ids TEXT DEFAULT '[]';

-- 添加索引
CREATE INDEX IF NOT EXISTS idx_experience_type ON experience_traces(experience_type);
CREATE INDEX IF NOT EXISTS idx_experience_chain_name ON experience_traces(chain_name);

-- 重建 FTS5 虚拟表（包含新字段）
DROP TRIGGER IF EXISTS experience_traces_fts_ai;
DROP TABLE IF EXISTS experience_traces_fts;

CREATE VIRTUAL TABLE experience_traces_fts USING fts5(
    task_description, lessons_learned, applicable_scenarios, anti_pattern, applicable_when,
    content='experience_traces', content_rowid='rowid',
    tokenize='trigram'
);

-- 重建触发器
CREATE TRIGGER experience_traces_fts_ai AFTER INSERT ON experience_traces BEGIN
    INSERT INTO experience_traces_fts(rowid, task_description, lessons_learned, applicable_scenarios, anti_pattern, applicable_when)
    VALUES (new.rowid, new.task_description, new.lessons_learned, new.applicable_scenarios, new.anti_pattern, new.applicable_when);
END;

-- 重新索引现有数据到 FTS5
INSERT INTO experience_traces_fts(rowid, task_description, lessons_learned, applicable_scenarios, anti_pattern, applicable_when)
SELECT rowid, task_description, lessons_learned, applicable_scenarios,
       COALESCE(anti_pattern, ''), COALESCE(applicable_when, '')
FROM experience_traces;
