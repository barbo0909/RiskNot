create table if not exists risk_assessments (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  default_probability double precision not null,
  risk_score integer not null,
  risk_segment text not null,
  predicted_default integer not null,
  selected_threshold double precision,
  final_model_name text,
  input_payload jsonb not null,
  model_response jsonb not null
);

create index if not exists risk_assessments_created_at_idx
  on risk_assessments (created_at desc);

create index if not exists risk_assessments_risk_segment_idx
  on risk_assessments (risk_segment);

grant usage on schema public to service_role;
grant select, insert, update, delete on table public.risk_assessments to service_role;
