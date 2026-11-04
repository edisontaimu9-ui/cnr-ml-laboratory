-- Run once in the Supabase SQL editor after the API has created its tables.
-- The API connects directly as a privileged DB role, so it is unaffected.
-- Enabling RLS with no policies blocks anonymous/PostgREST access to these tables.
do $$
declare t text;
begin
  for t in select tablename from pg_tables where schemaname='public' and tablename like 'mlab\_%'
  loop
    execute format('alter table public.%I enable row level security', t);
  end loop;
end $$;
