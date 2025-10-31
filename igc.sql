--
-- PostgreSQL database dump
--

\restrict hV4298B05hISqB1vbycaXFo1wYxe8cQGIAmwcw0Ne0FWEcW7JwggAYpuUqFBr7E

-- Dumped from database version 17.6 (Ubuntu 17.6-2.pgdg24.04+1)
-- Dumped by pg_dump version 17.6 (Ubuntu 17.6-2.pgdg24.04+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: igc
--

-- *not* creating schema, since initdb creates it


ALTER SCHEMA public OWNER TO igc;

--
-- Name: pgcrypto; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public;


--
-- Name: EXTENSION pgcrypto; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION pgcrypto IS 'cryptographic functions';


--
-- Name: ig_hash_spec(integer, integer, integer, integer, integer, integer, integer, integer, integer, jsonb); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.ig_hash_spec(p_simid integer, p_frame integer, p_metricid integer, p_stepid integer, p_phase integer, p_gridx integer, p_gridy integer, p_gridz integer, p_components integer, p_params jsonb) RETURNS text
    LANGUAGE sql IMMUTABLE PARALLEL SAFE
    AS $$
SELECT encode(
  digest(
    jsonb_build_object(
      'sim',  p_simid,
      'frame',p_frame,
      'metric',p_metricid,
      'step', p_stepid,
      'phase',p_phase,
      'grid', jsonb_build_array(p_gridx,p_gridy,p_gridz),
      'components', p_components,
      'params', p_params
    )::text,
    'sha256'
  ),
  'hex'
);
$$;


ALTER FUNCTION public.ig_hash_spec(p_simid integer, p_frame integer, p_metricid integer, p_stepid integer, p_phase integer, p_gridx integer, p_gridy integer, p_gridz integer, p_components integer, p_params jsonb) OWNER TO postgres;

--
-- Name: ig_hash_spec(bigint, bigint, bigint, integer, bigint, integer, integer, integer, integer, jsonb); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.ig_hash_spec(p_simid bigint, p_frame bigint, p_metricid bigint, p_stepid integer, p_phase bigint, p_gridx integer, p_gridy integer, p_gridz integer, p_components integer, p_params jsonb) RETURNS text
    LANGUAGE sql IMMUTABLE PARALLEL SAFE
    AS $$
SELECT encode(
  digest(
    jsonb_build_object(
      'sim',        p_simid,
      'frame',      p_frame,
      'metric',     p_metricid,
      'step',       p_stepid,
      'phase',      p_phase,
      'grid',       jsonb_build_array(p_gridx,p_gridy,p_gridz),
      'components', p_components,
      'params',     COALESCE(p_params, '{}'::jsonb)
    )::text,
    'sha256'
  ),
  'hex'
);
$$;


ALTER FUNCTION public.ig_hash_spec(p_simid bigint, p_frame bigint, p_metricid bigint, p_stepid integer, p_phase bigint, p_gridx integer, p_gridy integer, p_gridz integer, p_components integer, p_params jsonb) OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: errorlog; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.errorlog (
    id integer NOT NULL,
    jobid integer NOT NULL,
    simid integer NOT NULL,
    metricid integer,
    stepid integer,
    groupid integer,
    fieldid integer,
    jobtype text,
    jobsubtype text,
    phase integer,
    frame integer,
    priority integer,
    output_path text,
    message text NOT NULL,
    "timestamp" timestamp without time zone DEFAULT now()
);


ALTER TABLE public.errorlog OWNER TO postgres;

--
-- Name: ErrorLog_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public."ErrorLog_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public."ErrorLog_id_seq" OWNER TO postgres;

--
-- Name: ErrorLog_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public."ErrorLog_id_seq" OWNED BY public.errorlog.id;


--
-- Name: jobexecutionlog; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.jobexecutionlog (
    logid integer NOT NULL,
    jobid integer NOT NULL,
    simid integer NOT NULL,
    metricid integer,
    groupid integer,
    stepid integer,
    phase integer,
    frame integer,
    "precision" integer,
    status character varying,
    errormessage character varying,
    createdate timestamp without time zone,
    jobtype character varying,
    jobsubtype character varying,
    priority integer,
    startdate timestamp without time zone,
    finishdate timestamp without time zone,
    output_path character varying,
    runtime_ms integer,
    queue_wait_ms integer,
    recorded_at timestamp without time zone DEFAULT now() NOT NULL,
    filename text,
    was_aliased boolean,
    reused_step_id integer,
    reuse_metric_id integer,
    learning_note text
);


ALTER TABLE public.jobexecutionlog OWNER TO postgres;

--
-- Name: JobExecutionLog_logid_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public."JobExecutionLog_logid_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public."JobExecutionLog_logid_seq" OWNER TO postgres;

--
-- Name: JobExecutionLog_logid_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public."JobExecutionLog_logid_seq" OWNED BY public.jobexecutionlog.logid;


--
-- Name: fields; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.fields (
    id bigint NOT NULL,
    fieldid bigint,
    name text,
    description text,
    componentcount bigint,
    type text,
    origin text,
    usedinmetrics text
);


ALTER TABLE public.fields OWNER TO postgres;

--
-- Name: metgroup; Type: TABLE; Schema: public; Owner: igcuser
--

CREATE TABLE public.metgroup (
    id integer NOT NULL,
    name text NOT NULL,
    description text
);


ALTER TABLE public.metgroup OWNER TO igcuser;

--
-- Name: metricfieldmatcher; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.metricfieldmatcher (
    id bigint NOT NULL,
    metricid bigint,
    fieldid bigint,
    metricname text,
    metricdescription text,
    fieldname text,
    fielddescription text,
    fieldtype text,
    fieldorigin text
);


ALTER TABLE public.metricfieldmatcher OWNER TO postgres;

--
-- Name: metricinputmatcher; Type: TABLE; Schema: public; Owner: igc
--

CREATE TABLE public.metricinputmatcher (
    id bigint NOT NULL,
    metric_id bigint,
    metric_name text NOT NULL,
    step smallint NOT NULL,
    role text NOT NULL,
    lib_id integer,
    lib_name text NOT NULL,
    kf_id integer,
    kf_name text NOT NULL,
    op_id integer,
    op_name text,
    logical_name text,
    inputs_from text,
    artifact_ext text,
    artifact_file text,
    fanout_index integer,
    disabled boolean DEFAULT false,
    CONSTRAINT metricinputmatcher_role_check CHECK ((role = ANY (ARRAY['compute'::text, 'flatten'::text, 'final'::text]))),
    CONSTRAINT metricinputmatcher_step_check CHECK ((step = ANY (ARRAY[1, 2, 3])))
);


ALTER TABLE public.metricinputmatcher OWNER TO igc;

--
-- Name: metrics; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.metrics (
    id bigint NOT NULL,
    name text,
    description text,
    requiredfields text,
    outputtypes text,
    isstandard boolean DEFAULT true,
    postprocessing bigint DEFAULT '0'::bigint,
    filename text,
    group_id integer
);


ALTER TABLE public.metrics OWNER TO postgres;

--
-- Name: metrics_steps; Type: VIEW; Schema: public; Owner: igcuser
--

CREATE VIEW public.metrics_steps AS
 SELECT m.id AS met_id,
    m.name AS met_name,
    m.description AS met_description,
    m.requiredfields AS met_requiredfields,
    m.outputtypes AS met_outputtypes,
    m.isstandard AS met_isstandard,
    m.postprocessing AS met_postprocessing,
    m.filename AS met_filename,
    m.group_id AS met_group_id,
    mg.name AS met_group_name,
    mim.id AS mim_id,
    mim.metric_id AS mim_metric_id,
    mim.metric_name AS mim_metric_name,
    mim.step AS mim_step,
    mim.role AS mim_role,
    mim.lib_id AS mim_lib_id,
    mim.lib_name AS mim_lib_name,
    mim.kf_id AS mim_kf_id,
    mim.kf_name AS mim_kf_name,
    mim.op_id AS mim_op_id,
    mim.op_name AS mim_op_name,
    mim.logical_name AS mim_logical_name,
    mim.inputs_from AS mim_inputs_from,
    mim.artifact_ext AS mim_artifact_ext,
    mim.artifact_file AS mim_artifact_file,
    mim.fanout_index AS mim_fanout_index
   FROM ((public.metrics m
     LEFT JOIN public.metgroup mg ON ((mg.id = m.group_id)))
     LEFT JOIN public.metricinputmatcher mim ON ((mim.metric_id = m.id)))
  WHERE ((COALESCE(mim.disabled, false) = false) OR (mim.id IS NULL));


ALTER VIEW public.metrics_steps OWNER TO igcuser;

--
-- Name: simmetjobs; Type: TABLE; Schema: public; Owner: igcuser
--

CREATE TABLE public.simmetjobs (
    jobid integer NOT NULL,
    simid integer NOT NULL,
    metricid integer NOT NULL,
    frame integer NOT NULL,
    phase integer DEFAULT 0,
    status text DEFAULT 'queued'::text NOT NULL,
    priority integer DEFAULT 0,
    createdate timestamp without time zone DEFAULT now() NOT NULL,
    startdate timestamp without time zone,
    finishdate timestamp without time zone,
    spec_hash text,
    output_path text,
    output_extension text,
    output_type text,
    mime_type text,
    written_at timestamp without time zone,
    write_duration_ms integer,
    output_size_bytes bigint,
    artifact_hash text,
    attempts integer DEFAULT 0,
    error_code text,
    error_message text,
    cpu_ms integer,
    mem_peak_mb integer,
    io_mb_written integer,
    mem_grid_mb double precision,
    mem_pipeline_mb double precision,
    mem_total_mb double precision
);


ALTER TABLE public.simmetjobs OWNER TO igcuser;

--
-- Name: simulations; Type: TABLE; Schema: public; Owner: igc
--

CREATE TABLE public.simulations (
    id integer NOT NULL,
    name text NOT NULL,
    label text NOT NULL,
    description text,
    gridx integer,
    gridy integer,
    gridz integer,
    psi0_center double precision,
    psi0_elsewhere double precision,
    phi0 double precision,
    eta0 double precision,
    phi_threshold double precision,
    alpha double precision,
    t_max integer,
    stride integer,
    cleanup boolean,
    n_components integer,
    noise_mode text,
    collapse_rule text,
    profile_type text,
    status text,
    createdate timestamp without time zone,
    substeps_per_at integer,
    dt_per_at double precision,
    dx double precision,
    d_psi double precision,
    d_eta double precision,
    d_phi double precision,
    c_pi_to_eta double precision,
    c_eta_to_phi double precision,
    lambda_eta double precision,
    lambda_phi double precision,
    gate_name text,
    seed_type text,
    seed_field text,
    seed_strength double precision,
    seed_sigma double precision,
    seed_center text,
    seed_phase_a double precision,
    seed_phase_b double precision,
    seed_repeat_at integer,
    c_psi_to_phi double precision,
    c_phi_to_psi double precision,
    c_psi_to_eta double precision,
    c_eta_to_psi double precision,
    lambda_psi double precision,
    rng_seed bigint,
    pi0 double precision,
    pi_init_mode text,
    gamma_pi double precision,
    k_psi_restore double precision,
    save_pi boolean,
    integrator text,
    save_policy text,
    every_n_frames integer,
    checkpoint_interval integer,
    default_gridx integer,
    default_gridy integer,
    default_gridz integer,
    default_psi0_center double precision,
    default_psi0_elsewhere double precision,
    default_phi0 double precision,
    default_eta0 double precision,
    default_phi_threshold double precision,
    default_alpha double precision,
    default_t_max integer,
    default_stride integer,
    default_cleanup boolean,
    default_n_components integer,
    default_noise_mode text,
    default_collapse_rule text,
    default_profile_type text,
    default_substeps_per_at integer,
    default_dt_per_at double precision,
    default_dx double precision,
    default_d_psi double precision,
    default_d_eta double precision,
    default_d_phi double precision,
    default_c_pi_to_eta double precision,
    default_c_eta_to_phi double precision,
    default_lambda_eta double precision,
    default_lambda_phi double precision,
    default_gate_name text,
    default_seed_type text,
    default_seed_field text,
    default_seed_strength double precision,
    default_seed_sigma double precision,
    default_seed_center text,
    default_seed_phase_a double precision,
    default_seed_phase_b double precision,
    default_seed_repeat_at integer,
    default_c_psi_to_phi double precision,
    default_c_phi_to_psi double precision,
    default_c_psi_to_eta double precision,
    default_c_eta_to_psi double precision,
    default_lambda_psi double precision,
    default_rng_seed bigint,
    default_pi0 double precision,
    default_pi_init_mode text,
    default_gamma_pi double precision,
    default_k_psi_restore double precision,
    default_save_pi boolean,
    default_integrator text,
    default_save_policy text,
    default_every_n_frames integer,
    default_checkpoint_interval integer
);


ALTER TABLE public.simulations OWNER TO igc;

--
-- Name: big_view; Type: VIEW; Schema: public; Owner: igcuser
--

CREATE VIEW public.big_view AS
 WITH fl AS (
         SELECT mfm.metricid AS fl_metric_id,
            array_agg(mfm.fieldid ORDER BY mfm.fieldid) AS fl_field_ids,
            array_agg(f2.name ORDER BY mfm.fieldid) AS fl_field_names,
            array_agg(f2.type ORDER BY mfm.fieldid) AS fl_field_types,
            array_agg(f2.origin ORDER BY mfm.fieldid) AS fl_field_origins,
            array_agg(f2.componentcount ORDER BY mfm.fieldid) AS fl_field_components
           FROM (public.metricfieldmatcher mfm
             JOIN public.fields f2 ON ((f2.id = mfm.fieldid)))
          GROUP BY mfm.metricid
        )
 SELECT smj.jobid AS smj_jobid,
    smj.simid AS smj_simid,
    smj.metricid AS smj_metricid,
    smj.frame AS smj_frame,
    smj.phase AS smj_phase,
    smj.status AS smj_status,
    smj.priority AS smj_priority,
    smj.createdate AS smj_createdate,
    smj.startdate AS smj_startdate,
    smj.finishdate AS smj_finishdate,
    smj.spec_hash AS smj_spec_hash,
    smj.output_path AS smj_output_path,
    smj.output_extension AS smj_output_extension,
    smj.output_type AS smj_output_type,
    smj.mime_type AS smj_mime_type,
    smj.written_at AS smj_written_at,
    smj.write_duration_ms AS smj_write_duration_ms,
    smj.output_size_bytes AS smj_output_size_bytes,
    smj.artifact_hash AS smj_artifact_hash,
    smj.attempts AS smj_attempts,
    smj.error_code AS smj_error_code,
    smj.error_message AS smj_error_message,
    smj.cpu_ms AS smj_cpu_ms,
    smj.mem_peak_mb AS smj_mem_peak_mb,
    smj.io_mb_written AS smj_io_mb_written,
    smj.mem_grid_mb AS smj_mem_grid_mb,
    smj.mem_pipeline_mb AS smj_mem_pipeline_mb,
    smj.mem_total_mb AS smj_mem_total_mb,
    sim.id AS sim_id,
    sim.name AS sim_name,
    sim.label AS sim_label,
    sim.description AS sim_description,
    sim.gridx AS sim_gridx,
    sim.gridy AS sim_gridy,
    sim.gridz AS sim_gridz,
    sim.psi0_center AS sim_psi0_center,
    sim.psi0_elsewhere AS sim_psi0_elsewhere,
    sim.phi0 AS sim_phi0,
    sim.eta0 AS sim_eta0,
    sim.phi_threshold AS sim_phi_threshold,
    sim.alpha AS sim_alpha,
    sim.t_max AS sim_t_max,
    sim.stride AS sim_stride,
    sim.cleanup AS sim_cleanup,
    sim.n_components AS sim_n_components,
    sim.noise_mode AS sim_noise_mode,
    sim.collapse_rule AS sim_collapse_rule,
    sim.profile_type AS sim_profile_type,
    sim.status AS sim_status,
    sim.createdate AS sim_createdate,
    sim.substeps_per_at AS sim_substeps_per_at,
    sim.dt_per_at AS sim_dt_per_at,
    sim.dx AS sim_dx,
    sim.d_psi AS sim_d_psi,
    sim.d_eta AS sim_d_eta,
    sim.d_phi AS sim_d_phi,
    sim.c_pi_to_eta AS sim_c_pi_to_eta,
    sim.c_eta_to_phi AS sim_c_eta_to_phi,
    sim.c_psi_to_phi AS sim_c_psi_to_phi,
    sim.c_phi_to_psi AS sim_c_phi_to_psi,
    sim.c_psi_to_eta AS sim_c_psi_to_eta,
    sim.c_eta_to_psi AS sim_c_eta_to_psi,
    sim.lambda_eta AS sim_lambda_eta,
    sim.lambda_phi AS sim_lambda_phi,
    sim.lambda_psi AS sim_lambda_psi,
    sim.gate_name AS sim_gate_name,
    sim.seed_type AS sim_seed_type,
    sim.seed_field AS sim_seed_field,
    sim.seed_strength AS sim_seed_strength,
    sim.seed_sigma AS sim_seed_sigma,
    sim.seed_center AS sim_seed_center,
    sim.seed_phase_a AS sim_seed_phase_a,
    sim.seed_phase_b AS sim_seed_phase_b,
    sim.seed_repeat_at AS sim_seed_repeat_at,
    sim.rng_seed AS sim_rng_seed,
    sim.pi0 AS sim_pi0,
    sim.pi_init_mode AS sim_pi_init_mode,
    sim.gamma_pi AS sim_gamma_pi,
    sim.k_psi_restore AS sim_k_psi_restore,
    sim.save_pi AS sim_save_pi,
    sim.integrator AS sim_integrator,
    sim.save_policy AS sim_save_policy,
    sim.every_n_frames AS sim_every_n_frames,
    sim.checkpoint_interval AS sim_checkpoint_interval,
    met.id AS met_id,
    met.name AS met_name,
    met.description AS met_description,
    met.requiredfields AS met_requiredfields,
    met.outputtypes AS met_outputtypes,
    met.isstandard AS met_isstandard,
    met.postprocessing AS met_postprocessing,
    met.filename AS met_filename,
    met.group_id AS met_group_id,
    ms.mim_id AS ms_mim_id,
    ms.mim_metric_id AS ms_mim_metric_id,
    ms.mim_metric_name AS ms_mim_metric_name,
    ms.mim_step AS ms_mim_step,
    ms.mim_role AS ms_mim_role,
    ms.mim_lib_id AS ms_mim_lib_id,
    ms.mim_lib_name AS ms_mim_lib_name,
    ms.mim_kf_id AS ms_mim_kf_id,
    ms.mim_kf_name AS ms_mim_kf_name,
    ms.mim_op_id AS ms_mim_op_id,
    ms.mim_op_name AS ms_mim_op_name,
    ms.mim_logical_name AS ms_mim_logical_name,
    ms.mim_inputs_from AS ms_mim_inputs_from,
    ms.mim_artifact_ext AS ms_mim_artifact_ext,
    ms.mim_artifact_file AS ms_mim_artifact_file,
    ms.mim_fanout_index AS ms_mim_fanout_index,
    fl.fl_field_ids,
    fl.fl_field_names,
    fl.fl_field_types,
    fl.fl_field_origins,
    fl.fl_field_components
   FROM ((((public.simmetjobs smj
     JOIN public.simulations sim ON ((sim.id = smj.simid)))
     LEFT JOIN public.metrics met ON ((met.id = smj.metricid)))
     LEFT JOIN public.metrics_steps ms ON ((ms.met_id = smj.metricid)))
     LEFT JOIN fl ON ((fl.fl_metric_id = smj.metricid)))
  ORDER BY smj.jobid;


ALTER VIEW public.big_view OWNER TO igcuser;

--
-- Name: fields_pk_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.fields_pk_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.fields_pk_id_seq OWNER TO postgres;

--
-- Name: fields_pk_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.fields_pk_id_seq OWNED BY public.fields.id;


--
-- Name: kernelfamilies; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.kernelfamilies (
    id smallint NOT NULL,
    code text NOT NULL,
    n_tmp smallint,
    n_out smallint,
    note text
);


ALTER TABLE public.kernelfamilies OWNER TO postgres;

--
-- Name: libraries; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.libraries (
    id smallint NOT NULL,
    name text NOT NULL,
    version text NOT NULL,
    abi_tag text,
    active boolean DEFAULT true NOT NULL
);


ALTER TABLE public.libraries OWNER TO postgres;

--
-- Name: metgroup_counts; Type: VIEW; Schema: public; Owner: igcuser
--

CREATE VIEW public.metgroup_counts AS
 SELECT mg.id AS group_id,
    mg.name AS group_name,
    count(m.id) AS metric_count
   FROM (public.metgroup mg
     LEFT JOIN public.metrics m ON ((m.group_id = mg.id)))
  GROUP BY mg.id, mg.name
  ORDER BY mg.name;


ALTER VIEW public.metgroup_counts OWNER TO igcuser;

--
-- Name: metgroup_id_seq; Type: SEQUENCE; Schema: public; Owner: igcuser
--

CREATE SEQUENCE public.metgroup_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.metgroup_id_seq OWNER TO igcuser;

--
-- Name: metgroup_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: igcuser
--

ALTER SEQUENCE public.metgroup_id_seq OWNED BY public.metgroup.id;


--
-- Name: metric_fields; Type: VIEW; Schema: public; Owner: igcuser
--

CREATE VIEW public.metric_fields AS
 SELECT m.id AS metric_id,
    m.name AS metric_name,
    m.description AS metric_description,
    m.group_id AS met_group_id,
    mg.name AS met_group_name,
    mfm.id AS fanout_id,
    mfm.metricid AS fanout_metric_id,
    f.id AS out_field_id,
    f.name AS out_field_name,
    f.type AS out_field_type,
    f.origin AS out_field_origin,
    f.componentcount AS out_field_components
   FROM (((public.metricfieldmatcher mfm
     JOIN public.metrics m ON ((m.id = mfm.metricid)))
     LEFT JOIN public.metgroup mg ON ((mg.id = m.group_id)))
     JOIN public.fields f ON ((f.id = mfm.fieldid)))
  ORDER BY m.id, mfm.id;


ALTER VIEW public.metric_fields OWNER TO igcuser;

--
-- Name: metricfieldmatcher_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.metricfieldmatcher_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.metricfieldmatcher_id_seq OWNER TO postgres;

--
-- Name: metricfieldmatcher_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.metricfieldmatcher_id_seq OWNED BY public.metricfieldmatcher.id;


--
-- Name: metricinputmatcher_id_seq; Type: SEQUENCE; Schema: public; Owner: igc
--

CREATE SEQUENCE public.metricinputmatcher_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.metricinputmatcher_id_seq OWNER TO igc;

--
-- Name: metricinputmatcher_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: igc
--

ALTER SEQUENCE public.metricinputmatcher_id_seq OWNED BY public.metricinputmatcher.id;


--
-- Name: metrics_group; Type: VIEW; Schema: public; Owner: igcuser
--

CREATE VIEW public.metrics_group AS
 SELECT mg.id AS group_id,
    mg.name AS group_name,
    m.id AS metric_id,
    m.name AS metric_name,
    m.description AS metric_description
   FROM (public.metrics m
     JOIN public.metgroup mg ON ((mg.id = m.group_id)))
  ORDER BY mg.name, m.name;


ALTER VIEW public.metrics_group OWNER TO igcuser;

--
-- Name: metrics_pk_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.metrics_pk_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.metrics_pk_id_seq OWNER TO postgres;

--
-- Name: metrics_pk_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.metrics_pk_id_seq OWNED BY public.metrics.id;


--
-- Name: opkinds; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.opkinds (
    id smallint NOT NULL,
    family_id smallint NOT NULL,
    code text NOT NULL
);


ALTER TABLE public.opkinds OWNER TO postgres;

--
-- Name: pathregistry; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.pathregistry (
    id integer NOT NULL,
    timestamptemplate text,
    frametemplate text,
    simnametemplate text,
    groupnametemplate text,
    metricidtemplate text,
    steptemplate text,
    filenametemplate text,
    intermediate text DEFAULT '/intermediate.npy'::text,
    fieldtemplate text
);


ALTER TABLE public.pathregistry OWNER TO postgres;

--
-- Name: pathregistry_new_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.pathregistry_new_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.pathregistry_new_id_seq OWNER TO postgres;

--
-- Name: pathregistry_new_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.pathregistry_new_id_seq OWNED BY public.pathregistry.id;


--
-- Name: simmetricmatcher; Type: TABLE; Schema: public; Owner: igcuser
--

CREATE TABLE public.simmetricmatcher (
    id integer NOT NULL,
    sim_id integer NOT NULL,
    metric_id integer NOT NULL,
    enabled boolean DEFAULT true NOT NULL,
    updated_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.simmetricmatcher OWNER TO igcuser;

--
-- Name: sim_metrics; Type: VIEW; Schema: public; Owner: igcuser
--

CREATE VIEW public.sim_metrics AS
 SELECT s.id AS sim_id,
    s.label AS sim_label,
    mg.name AS group_name,
    m.id AS metric_id,
    m.name AS metric_name,
    m.description AS metric_description,
    sm.updated_at AS last_change
   FROM (((public.simmetricmatcher sm
     JOIN public.simulations s ON ((s.id = sm.sim_id)))
     JOIN public.metrics m ON ((m.id = sm.metric_id)))
     JOIN public.metgroup mg ON ((mg.id = m.group_id)))
  WHERE (sm.enabled = true)
  ORDER BY s.id, mg.name, m.name;


ALTER VIEW public.sim_metrics OWNER TO igcuser;

--
-- Name: simmetjobs_jobid_seq; Type: SEQUENCE; Schema: public; Owner: igcuser
--

CREATE SEQUENCE public.simmetjobs_jobid_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.simmetjobs_jobid_seq OWNER TO igcuser;

--
-- Name: simmetjobs_jobid_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: igcuser
--

ALTER SEQUENCE public.simmetjobs_jobid_seq OWNED BY public.simmetjobs.jobid;


--
-- Name: simmetricmatcher_id_seq; Type: SEQUENCE; Schema: public; Owner: igcuser
--

CREATE SEQUENCE public.simmetricmatcher_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.simmetricmatcher_id_seq OWNER TO igcuser;

--
-- Name: simmetricmatcher_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: igcuser
--

ALTER SEQUENCE public.simmetricmatcher_id_seq OWNED BY public.simmetricmatcher.id;


--
-- Name: simulations_new_id_seq; Type: SEQUENCE; Schema: public; Owner: igc
--

CREATE SEQUENCE public.simulations_new_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.simulations_new_id_seq OWNER TO igc;

--
-- Name: simulations_new_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: igc
--

ALTER SEQUENCE public.simulations_new_id_seq OWNED BY public.simulations.id;


--
-- Name: trackeddomains; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.trackeddomains (
    domainid bigint,
    frame bigint,
    phase bigint,
    centroidx real,
    centroidy real,
    centroidz real,
    psix real,
    psiy real,
    psiz real,
    phisum real,
    phimin real,
    etasum real,
    voxelcount bigint
);


ALTER TABLE public.trackeddomains OWNER TO postgres;

--
-- Name: errorlog id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.errorlog ALTER COLUMN id SET DEFAULT nextval('public."ErrorLog_id_seq"'::regclass);


--
-- Name: fields id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.fields ALTER COLUMN id SET DEFAULT nextval('public.fields_pk_id_seq'::regclass);


--
-- Name: jobexecutionlog logid; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.jobexecutionlog ALTER COLUMN logid SET DEFAULT nextval('public."JobExecutionLog_logid_seq"'::regclass);


--
-- Name: metgroup id; Type: DEFAULT; Schema: public; Owner: igcuser
--

ALTER TABLE ONLY public.metgroup ALTER COLUMN id SET DEFAULT nextval('public.metgroup_id_seq'::regclass);


--
-- Name: metricfieldmatcher id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.metricfieldmatcher ALTER COLUMN id SET DEFAULT nextval('public.metricfieldmatcher_id_seq'::regclass);


--
-- Name: metricinputmatcher id; Type: DEFAULT; Schema: public; Owner: igc
--

ALTER TABLE ONLY public.metricinputmatcher ALTER COLUMN id SET DEFAULT nextval('public.metricinputmatcher_id_seq'::regclass);


--
-- Name: metrics id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.metrics ALTER COLUMN id SET DEFAULT nextval('public.metrics_pk_id_seq'::regclass);


--
-- Name: pathregistry id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.pathregistry ALTER COLUMN id SET DEFAULT nextval('public.pathregistry_new_id_seq'::regclass);


--
-- Name: simmetjobs jobid; Type: DEFAULT; Schema: public; Owner: igcuser
--

ALTER TABLE ONLY public.simmetjobs ALTER COLUMN jobid SET DEFAULT nextval('public.simmetjobs_jobid_seq'::regclass);


--
-- Name: simmetricmatcher id; Type: DEFAULT; Schema: public; Owner: igcuser
--

ALTER TABLE ONLY public.simmetricmatcher ALTER COLUMN id SET DEFAULT nextval('public.simmetricmatcher_id_seq'::regclass);


--
-- Name: simulations id; Type: DEFAULT; Schema: public; Owner: igc
--

ALTER TABLE ONLY public.simulations ALTER COLUMN id SET DEFAULT nextval('public.simulations_new_id_seq'::regclass);


--
-- Data for Name: errorlog; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.errorlog (id, jobid, simid, metricid, stepid, groupid, fieldid, jobtype, jobsubtype, phase, frame, priority, output_path, message, "timestamp") FROM stdin;
1	286	95	25	2	100	\N	step_1	1	0	0	0	data/igc/Time_20251025_0648/Sim_TEST/Frame_0/Group_TESTGroup/Metric_betti_numbers/Step_1/Field_2/intermediate.npy	invalid input syntax for type integer: "Float32"\nCONTEXT:  unnamed portal parameter $8 = '...'	2025-10-25 08:55:36.550191
\.


--
-- Data for Name: fields; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.fields (id, fieldid, name, description, componentcount, type, origin, usedinmetrics) FROM stdin;
1	16	psi_phase_map	Phase angle of psi	1	scalar	computed	20
2	11	psi_phase	Phase angle of ψ	1	derived	cpu	19
3	6	psi_gradient	∇ψ magnitude	1	derived	computed	7
4	8	psi_entropy_map	Entropy proxy from ∇ψ²	1	derived	gpu	5
5	1	psi	Wavefunction components	9	vector	npy	1,2,3,4,5,6,7,8,9
6	12	pointerMask	Pointer mask from φ	1	mask	computed	11,12
7	2	phi	Collapse field	1	scalar	npy	1,2,3,4,5,6,7,8
8	7	eta_curvature	∇²η field	1	derived	computed	10
9	3	eta	Auxiliary shell/curvature field	1	scalar	npy	1,2,3,10
10	10	domain_mask	Domain segmentation mask	1	mask	gpu	13
11	9	domainID	Domain ID label grid	1	mask	gpu	13
13	55	coherence_length	ψ autocorrelation span (coherence length)	1	scalar	computed	40
14	56	echo_radius	Radial distance of detected echo ring	1	scalar	computed	182
15	57	psi_coherence	Mean ψ amplitude at detected echo radius	1	scalar	computed	182
16	58	delta_eta	Mean η change at detected echo radius	1	scalar	computed	182
17	59	echo_time	Simulation time corresponding to detected echo	1	scalar	computed	182
18	60	echo_count	Number of detected echo rings per frame	1	scalar	computed	182
19	\N	collapse_echo_profile	\N	1	npy	intermediate	\N
12	4	collapse_mask	φ-derived collapse regions	1	mask	computed	11,25
\.


--
-- Data for Name: jobexecutionlog; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.jobexecutionlog (logid, jobid, simid, metricid, groupid, stepid, phase, frame, "precision", status, errormessage, createdate, jobtype, jobsubtype, priority, startdate, finishdate, output_path, runtime_ms, queue_wait_ms, recorded_at, filename, was_aliased, reused_step_id, reuse_metric_id, learning_note) FROM stdin;
\.


--
-- Data for Name: kernelfamilies; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.kernelfamilies (id, code, n_tmp, n_out, note) FROM stdin;
10	overlay	2	1	overlay/rasterize buffers
9	serialize	0	0	writer only
8	stats	2	1	reductions/flatten
7	convolve	2	1	generic conv kernels
6	grad	2	1	stencils ∇/∇·/filters
5	reduce	2	1	reductions
4	scan	2	1	radial binning/echo
3	topology	2	1	topology fanouts
2	fft	3	1	3D FFT working + spectrum
1	morph	1	1	threshold/ccs
\.


--
-- Data for Name: libraries; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.libraries (id, name, version, abi_tag, active) FROM stdin;
1	accelerate	vDSP	arm64-macos15	t
2	fftw	3.3.10	arm64-macos15	t
3	gudhi	3.9.0	arm64-macos15	t
4	metal_fft	r1	arm64-macos15	t
5	metal_scan	r1	arm64-macos15	t
99	writer	-	OutputWriter	t
6	graphics	r1	arm64-macos15	t
7	custom_event_pack	r1	arm64-macos15	t
8	openblas	1.0	arm64-macos15	t
9	itk	1.0	arm64-macos15	t
100	gudhi	3.11.0	x86_64-linux-gnu	t
101	numpy	2.3.4	x86_64-linux-gnu	t
102	pyfftw	0.15.1	x86_64-linux-gnu	t
103	scipy.ndimage	1.16.2	x86_64-linux-gnu	t
104	writer	1.0	x86_64-linux-gnu	t
10	ckernels	1.0	arm64-macos15	f
\.


--
-- Data for Name: metgroup; Type: TABLE DATA; Schema: public; Owner: igcuser
--

COPY public.metgroup (id, name, description) FROM stdin;
1	observables	Core summary metrics computed for every frame. Includes averages, extrema, variances, and counts for ψ, φ, η and derived scalars such as σ₈. Use this group to monitor overall simulation health and progression.
2	topology	Topology and geometry of collapse. Contains φ-masking, Betti numbers, shell width and curvature, and gradient analyses that reveal when and where structure forms and how it seals.
3	domains	Per-domain analyses. Generates tables, geometry, fusion and decay tracking for individual coherent domains. Use when studying particle identity, motion, or domain stability.
4	spatial_overlays	Visual and volumetric outputs such as entropy or gradient maps, pointer overlays, and rendered slices. Produces large NPY/PNG/MP4 artifacts for inspection and presentations.
5	spectra	Spectral and correlation analyses. Computes power spectra P(k), correlation functions ξ(r), and coherence-length statistics to quantify order across spatial scales.
6	heavy	Comprehensive or computationally intensive diagnostic suites. Includes entropy volumes, phase and spin analysis, reaction cascades, Standard-Model tables, and other high-cost research metrics. Enable only when you need full scientific detail.
\.


--
-- Data for Name: metricfieldmatcher; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.metricfieldmatcher (id, metricid, fieldid, metricname, metricdescription, fieldname, fielddescription, fieldtype, fieldorigin) FROM stdin;
1	20	1	psi_phase_drift	Phase drift between frames	psi_phase_map	Phase angle of psi	scalar	computed
2	19	2	psi_phase_map	Phase map image (CPU)	psi_phase	Phase angle of ψ	derived	cpu
3	7	3	psi_gradient_magnitude	||∇ψ|| magnitude field	psi_gradient	∇ψ magnitude	derived	computed
4	5	4	psi_entropy_map	GPU entropy map (∇ψ² 3D field)	psi_entropy_map	Entropy proxy from ∇ψ²	derived	gpu
8	4	5	psi_entropy	Mean entropy proxy (∇ψ²)	psi	Wavefunction components	vector	npy
9	5	5	psi_entropy_map	GPU entropy map (∇ψ² 3D field)	psi	Wavefunction components	vector	npy
10	6	5	sigma8	σ₈ smoothed ψ² statistic	psi	Wavefunction components	vector	npy
11	7	5	psi_gradient_magnitude	||∇ψ|| magnitude field	psi	Wavefunction components	vector	npy
12	8	5	phi_shell_width	Width of collapse shell in φ	psi	Wavefunction components	vector	npy
13	9	5	component_density	ψⁱ component-wise densities	psi	Wavefunction components	vector	npy
14	11	6	pointer_class_histogram	Domain pointer class histogram	pointerMask	Pointer mask from φ	mask	computed
15	12	6	domain_tracking	Domain centroid + ψ vector	pointerMask	Pointer mask from φ	mask	computed
19	4	7	psi_entropy	Mean entropy proxy (∇ψ²)	phi	Collapse field	scalar	npy
20	5	7	psi_entropy_map	GPU entropy map (∇ψ² 3D field)	phi	Collapse field	scalar	npy
21	6	7	sigma8	σ₈ smoothed ψ² statistic	phi	Collapse field	scalar	npy
22	7	7	psi_gradient_magnitude	||∇ψ|| magnitude field	phi	Collapse field	scalar	npy
23	8	7	phi_shell_width	Width of collapse shell in φ	phi	Collapse field	scalar	npy
24	10	8	eta_curvature	∇²η curvature	eta_curvature	∇²η field	derived	computed
28	10	9	eta_curvature	∇²η curvature	eta	Auxiliary shell/curvature field	scalar	npy
29	13	10	tracked_domains_gpu	GPU domain tracker	domain_mask	Domain segmentation mask	mask	gpu
30	13	11	tracked_domains_gpu	GPU domain tracker	domainID	Domain ID label grid	mask	gpu
31	11	12	pointer_class_histogram	Domain pointer class histogram	collapseMask	φ-derived collapse regions	mask	computed
40	182	17	collapse_echo_profile	Identifies and logs every detected echo ring. For each ring, outputs timing, radius, ψ coherence, and Δη in a structured JSON.	echo_time	Simulation time corresponding to detected echo	scalar	computed
41	182	18	collapse_echo_profile	Identifies and logs every detected echo ring. For each ring, outputs timing, radius, ψ coherence, and Δη in a structured JSON.	echo_count	Number of detected echo rings per frame	scalar	computed
42	182	15	collapse_echo_profile	Identifies and logs every detected echo ring. For each ring, outputs timing, radius, ψ coherence, and Δη in a structured JSON.	psi_coherence	Mean ψ amplitude at detected echo radius	scalar	computed
43	182	16	collapse_echo_profile	Identifies and logs every detected echo ring. For each ring, outputs timing, radius, ψ coherence, and Δη in a structured JSON.	delta_eta	Mean η change at detected echo radius	scalar	computed
34	251	12	collapse_mask	Binary collapse mask from φ (pp0)	collapse_mask	φ-derived collapse regions	mask	computed
44	182	19	\N	\N	\N	\N	\N	\N
\.


--
-- Data for Name: metricinputmatcher; Type: TABLE DATA; Schema: public; Owner: igc
--

COPY public.metricinputmatcher (id, metric_id, metric_name, step, role, lib_id, lib_name, kf_id, kf_name, op_id, op_name, logical_name, inputs_from, artifact_ext, artifact_file, fanout_index, disabled) FROM stdin;
6	25	betti_numbers	2	flatten	100	gudhi	3	topology	301	betti	betti_partial	labels	npy	intermediate.npy	\N	f
69	33	domain_charge	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	domain_charge.csv	\N	f
15	178	bond_energy_chain	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	bond_energy_chain.csv	\N	f
18	130	bond_energy_estimator	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	bond_energy_estimator.json	\N	f
1	244	atom_overlay	1	compute	6	graphics	9	serialize	\N		s1_compute	domain_table.csv,pointer_map.npy	npy	intermediate.npy	\N	f
2	244	atom_overlay	2	flatten	6	graphics	9	serialize	\N		s2_flatten	s1_compute	npy	intermediate.npy	\N	f
21	179	bond_energy_overlay	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	bond_energy_overlay.csv	\N	f
22	179	bond_energy_overlay	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	bond_energy_overlay.json	\N	f
5	25	betti_numbers	1	compute	9	itk	1	morph	101	connected_components	labels	collapse_mask|mask	npy	intermediate.npy	\N	f
8	177	BField_overlay	1	compute	6	graphics	9	serialize	\N		s1_compute	pointer_map.npy, domain_table.csv, psi	npy	intermediate.npy	\N	f
9	177	BField_overlay	2	flatten	6	graphics	9	serialize	\N		s2_flatten	s1_compute	npy	intermediate.npy	\N	f
25	180	bond_energy_stats	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	bond_energy_stats.csv	\N	f
28	52	bond_lifetime_estimator	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	bond_lifetime_estimator.json	\N	f
29	52	bond_lifetime_estimator	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	bond_lifetime_estimator.csv	\N	f
30	246	branch_point_detector	1	compute	101	numpy	6	grad	601	gradient	s1_compute	domain_table,domain_tracking	npy	intermediate.npy	\N	f
31	246	branch_point_detector	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
19	179	bond_energy_overlay	1	compute	6	graphics	9	serialize	\N		s1_compute	bond_energy_chain.csv, domain_table.csv, pointer_map.npy	npy	intermediate.npy	\N	f
20	179	bond_energy_overlay	2	flatten	6	graphics	9	serialize	\N		s2_flatten	s1_compute	npy	intermediate.npy	\N	f
32	246	branch_point_detector	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	branch_point_detector.csv	\N	f
33	246	branch_point_detector	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	branch_point_detector.json	\N	f
34	65	charge_vector_map	1	compute	101	numpy	6	grad	601	gradient	s1_compute	domain_table.csv	npy	intermediate.npy	\N	f
35	65	charge_vector_map	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
37	65	charge_vector_map	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	charge_vector_map.csv	\N	f
4	244	atom_overlay	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	atom_overlay.csv	2	f
224	207	magnetic_structure_field	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	magnetic_structure_field.json	\N	f
223	207	magnetic_structure_field	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	magnetic_structure_field.csv	\N	f
225	207	magnetic_structure_field	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	magnetic_structure_field.png	\N	f
222	207	magnetic_structure_field	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
41	181	collapse_echo_overlay	1	compute	101	numpy	4	scan	\N		s1_compute	phi, psi, collapse_echo_shellmap.npy	npy	intermediate.npy	\N	f
42	181	collapse_echo_overlay	2	flatten	101	numpy	4	scan	401	shellscan	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
13	178	bond_energy_chain	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	domain_table.csv	npy	intermediate.npy	\N	f
14	178	bond_energy_chain	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
17	130	bond_energy_estimator	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
38	40	coherence_length	1	compute	102	pyfftw	2	fft	202	power_spectrum	spectrum_or_corr	psi	npy	intermediate.npy	\N	f
7	25	betti_numbers	3	final	104	writer	3	serialize	301	betti	betti_table	betti_partial	csv	betti_numbers.csv	\N	f
36	65	charge_vector_map	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	charge_vector_map.npy	\N	f
39	40	coherence_length	2	flatten	101	numpy	2	stats	205	coh_scalar	coh_scalar	spectrum_or_corr	npy	intermediate.npy	\N	f
40	40	coherence_length	3	final	104	writer	2	serialize	204	autocorr_length	coh_length	coh_scalar	csv	coherence_length.csv	\N	f
43	181	collapse_echo_overlay	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	collapse_echo_overlay.png	\N	f
44	182	collapse_echo_profile	1	compute	101	numpy	4	scan	403	mask	mask	phi|psi|eta	npy	intermediate.npy	\N	f
45	182	collapse_echo_profile	2	flatten	101	numpy	4	scan	409	echo_times	echo_times	mask	npy	intermediate.npy	1	f
46	182	collapse_echo_profile	2	flatten	101	numpy	4	scan	407	echo_radii	echo_radii	mask	npy	intermediate.npy	2	f
47	182	collapse_echo_profile	2	flatten	101	numpy	4	scan	406	echo_psi_coh	echo_psi_coh	mask	npy	intermediate.npy	3	f
48	182	collapse_echo_profile	2	flatten	101	numpy	4	scan	405	echo_delta_eta	echo_delta_eta	mask	npy	intermediate.npy	4	f
50	183	collapse_echo_shellmap	2	compute	101	numpy	4	scan	410	shellmap_buffer	shellmap_buffer	psi|phi	npy	intermediate.npy	\N	f
52	184	collapse_echo_summary	2	compute	101	numpy	4	scan	408	echo_stats	echo_stats	phi|psi|eta	npy	intermediate.npy	\N	f
54	251	collapse_mask	1	compute	103	scipy.ndimage	1	morph	101	threshold	mask	phi	npy	intermediate.npy	\N	f
55	251	collapse_mask	2	flatten	103	scipy.ndimage	1	morph	104	voxel_stats	voxel_stats	mask	npy	intermediate.npy	\N	f
58	9	component_density	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	psi	npy	intermediate.npy	\N	f
59	9	component_density	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
60	9	component_density	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	component_density.csv	\N	f
63	32	decay_events	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	decay_events.csv	\N	f
66	185	domain_bundle_tracking	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	domain_bundle_tracking.json	\N	f
67	33	domain_charge	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
68	33	domain_charge	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	psi	npy	intermediate.npy	\N	f
70	41	domain_count	1	compute	103	scipy.ndimage	1	morph	101	threshold	s1_compute	pointer_mask.npy	npy	intermediate.npy	\N	f
71	41	domain_count	2	flatten	103	scipy.ndimage	1	morph	102	connected_components	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
72	41	domain_count	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	domain_count.csv	\N	f
73	30	domain_geometry	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
74	30	domain_geometry	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	psi	npy	intermediate.npy	\N	f
75	30	domain_geometry	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	domain_geometry.csv	\N	f
78	67	domain_hash_map	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	domain_hash_map.json	\N	f
79	67	domain_hash_map	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	domain_hash_map.csv	\N	f
80	186	domain_map_summary	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	domain_hash_map.csv, domain_table.csv, domain_persistence.csv	npy	intermediate.npy	\N	f
81	186	domain_map_summary	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
82	186	domain_map_summary	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	domain_map_summary.csv	\N	f
83	187	domain_mask	1	compute	103	scipy.ndimage	1	morph	101	threshold	s1_compute	domain_table.csv, domain_hash_map.csv	npy	intermediate.npy	\N	f
84	187	domain_mask	2	flatten	103	scipy.ndimage	1	morph	102	connected_components	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
85	187	domain_mask	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	domain_mask.png	\N	f
88	27	domain_persistence	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	domain_persistence.csv	\N	f
89	43	domain_table	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
49	182	collapse_echo_profile	3	final	104	writer	4	serialize	401	shellscan	echo_list	echo_times|echo_radii|echo_psi_coh|echo_delta_eta	json	collapse_echo_profile.json	\N	f
51	183	collapse_echo_shellmap	3	final	104	writer	4	serialize	401	shellscan	shellmap	shellmap_buffer	npy	collapse_echo_shellmap.npy	\N	f
53	184	collapse_echo_summary	3	final	104	writer	4	serialize	401	shellscan	echo_summary	echo_stats	csv	collapse_echo_summary.csv	\N	f
56	251	collapse_mask	3	final	104	writer	1	serialize	103	mask_final	mask_final	mask	npy	collapse_mask.npy	\N	f
57	251	collapse_mask	3	final	104	writer	1	serialize	101	threshold	mask_stats	voxel_stats	csv	collapse_mask.csv	\N	f
90	43	domain_table	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	psi, phi, eta	npy	intermediate.npy	\N	f
91	43	domain_table	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	domain_table.json	\N	f
92	43	domain_table	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	domain_table.csv	\N	f
96	12	domain_tracking	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	domain_tracking.csv	\N	f
97	12	domain_tracking	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	domain_tracking.json	\N	f
98	12	domain_tracking	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	domain_tracking.png	\N	f
99	131	domain_vector_alignment	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	domain_table.csv,psi	npy	intermediate.npy	\N	f
100	131	domain_vector_alignment	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
101	131	domain_vector_alignment	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	domain_vector_alignment.csv	\N	f
102	131	domain_vector_alignment	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	domain_vector_alignment.json	\N	f
103	132	domain_vector_drift	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	domain_table.csv,psi	npy	intermediate.npy	\N	f
104	132	domain_vector_drift	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
105	132	domain_vector_drift	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	domain_vector_drift.csv	\N	f
106	132	domain_vector_drift	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	domain_vector_drift.json	\N	f
108	133	domain_vector_field	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
109	133	domain_vector_field	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	domain_table.csv,psi	npy	intermediate.npy	\N	f
110	133	domain_vector_field	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	domain_vector_field.csv	\N	f
111	72	drift_tracker	1	compute	101	numpy	6	grad	601	gradient	s1_compute	domain_tracking.csv, domain_table.csv	npy	intermediate.npy	\N	f
112	72	drift_tracker	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
113	72	drift_tracker	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	drift_tracker.json	\N	f
114	72	drift_tracker	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	drift_tracker.csv	\N	f
115	188	drift_vector_map	1	compute	101	numpy	6	grad	601	gradient	s1_compute	eta_shear_gradient.npy, domain_tracking.csv	npy	intermediate.npy	\N	f
116	188	drift_vector_map	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
118	189	energy_release	1	compute	101	numpy	6	grad	601	gradient	s1_compute	fusion_events.csv, decay_events.csv, domain_table.csv	npy	intermediate.npy	\N	f
119	189	energy_release	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
120	189	energy_release	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	energy_release.csv	\N	f
121	189	energy_release	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	energy_release.json	\N	f
122	75	energy_release_estimator	1	compute	101	numpy	6	grad	601	gradient	s1_compute	fusion_events.csv, decay_events.csv, domain_table.csv	npy	intermediate.npy	\N	f
123	75	energy_release_estimator	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
124	75	energy_release_estimator	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	energy_release_estimator.csv	\N	f
125	248	entropy_plateau_analyzer	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	psi_entropy_proxy	npy	intermediate.npy	\N	f
126	248	entropy_plateau_analyzer	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
127	248	entropy_plateau_analyzer	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	entropy_plateau_analyzer.csv	\N	f
128	248	entropy_plateau_analyzer	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	entropy_plateau_analyzer.json	\N	f
129	254	eta	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi|phi|eta	npy	intermediate.npy	\N	f
130	254	eta	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
135	10	eta_curvature	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	eta_curvature.csv	\N	f
94	12	domain_tracking	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	psi,phi,eta	npy	intermediate.npy	\N	f
95	12	domain_tracking	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
93	12	domain_tracking	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	domain_tracking.npy	\N	f
107	133	domain_vector_field	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	domain_vector_field.npy	\N	f
139	76	eta_flux_map	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	eta_flux_map.csv	\N	f
140	245	eta_integral	1	compute	101	numpy	6	grad	601	gradient	s1_compute	eta	npy	intermediate.npy	\N	f
141	245	eta_integral	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
137	76	eta_flux_map	1	compute	4	metal_fft	6	grad	601	gradient	s1_compute	eta	npy	intermediate.npy	\N	f
138	76	eta_flux_map	2	flatten	4	metal_fft	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
142	245	eta_integral	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	eta_integral.csv	\N	f
143	190	eta_max	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	eta	npy	intermediate.npy	\N	f
144	190	eta_max	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
145	190	eta_max	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	eta_max.csv	\N	f
146	191	eta_mean	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	eta	npy	intermediate.npy	\N	f
147	191	eta_mean	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
148	191	eta_mean	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	eta_mean.csv	\N	f
149	192	eta_min	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	eta	npy	intermediate.npy	\N	f
150	192	eta_min	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
151	192	eta_min	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	eta_min.csv	\N	f
155	46	eta_shear_gradient	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	eta_shear_gradient.csv	\N	f
156	46	eta_shear_gradient	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	eta_shear_gradient.png	\N	f
157	193	eta_std	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	eta	npy	intermediate.npy	\N	f
158	193	eta_std	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
159	193	eta_std	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	eta_std.csv	\N	f
163	14	flux_map_gpu	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	flux_map_gpu.png	\N	f
166	238	frozen_domains	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	frozen_domains.json	\N	f
169	194	fusion_chain_graph	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	fusion_chain_graph.json	\N	f
161	14	flux_map_gpu	1	compute	4	metal_fft	6	grad	601	gradient	s1_compute	psi,phi	npy	intermediate.npy	\N	f
162	14	flux_map_gpu	2	flatten	4	metal_fft	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
172	195	fusion_decay_matrix	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	fusion_decay_matrix.json	\N	f
175	31	fusion_events	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	fusion_events.csv	\N	f
178	196	fusion_events_summary	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	fusion_events_summary.json	\N	f
179	18	gauge_drift	1	compute	101	numpy	6	grad	601	gradient	s1_compute	phi	npy	intermediate.npy	\N	f
180	18	gauge_drift	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
181	18	gauge_drift	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	gauge_drift.csv	\N	f
182	28	global_phase_drift	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi	npy	intermediate.npy	\N	f
183	28	global_phase_drift	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
184	28	global_phase_drift	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	global_phase_drift.csv	\N	f
164	238	frozen_domains	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	psi,phi,eta	npy	intermediate.npy	\N	f
165	238	frozen_domains	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
154	46	eta_shear_gradient	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	eta_shear_gradient.npy	\N	f
160	14	flux_map_gpu	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	flux_map_gpu.npy	\N	f
187	197	grad_phi	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	grad_phi.npy	\N	f
190	38	grad_phi_max	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	grad_phi_max.csv	\N	f
193	198	grad_phi_mean	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	grad_phi_mean.csv	\N	f
196	199	grad_phi_min	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	grad_phi_min.csv	\N	f
293	216	phi_min	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
205	202	grad_phi_std	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	grad_phi_std.csv	\N	f
206	203	identity_clustering_hierarchical	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	psi, domain_table.csv	npy	intermediate.npy	\N	f
207	203	identity_clustering_hierarchical	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
208	203	identity_clustering_hierarchical	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	identity_clustering_hierarchical.csv	\N	f
209	203	identity_clustering_hierarchical	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	identity_clustering_hierarchical.json	\N	f
210	204	identity_clustering_kmeans	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	psi, domain_table.csv	npy	intermediate.npy	\N	f
211	204	identity_clustering_kmeans	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
212	204	identity_clustering_kmeans	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	identity_clustering_kmeans.csv	\N	f
213	204	identity_clustering_kmeans	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	identity_clustering_kmeans.json	\N	f
217	205	identity_clustering_overlay	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	identity_clustering_overlay.png	\N	f
218	206	lineage_chain_graph	1	compute	101	numpy	6	grad	601	gradient	s1_compute	fusion_events.csv, decay_events.csv, domain_persistence.csv, domain_tracking.csv	npy	intermediate.npy	\N	f
219	206	lineage_chain_graph	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
215	205	identity_clustering_overlay	1	compute	6	graphics	9	serialize	\N		s1_compute	identity_clustering_kmeans.csv,identity_clustering_hierarchical.csv, domain_table.csv	npy	intermediate.npy	\N	f
216	205	identity_clustering_overlay	2	flatten	6	graphics	9	serialize	\N		s2_flatten	s1_compute	npy	intermediate.npy	\N	f
220	206	lineage_chain_graph	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	lineage_chain_graph.json	\N	f
226	85	magnetic_structure_tracker	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi, domain_table.csv	npy	intermediate.npy	\N	f
227	85	magnetic_structure_tracker	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
229	85	magnetic_structure_tracker	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	magnetic_structure_tracker.csv	\N	f
230	85	magnetic_structure_tracker	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	magnetic_structure_tracker.json	\N	f
231	86	mass_ratio_matrix	1	compute	101	numpy	6	grad	601	gradient	s1_compute	domain_table.csv, eta	npy	intermediate.npy	\N	f
232	86	mass_ratio_matrix	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
233	86	mass_ratio_matrix	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	mass_ratio_matrix.json	\N	f
234	86	mass_ratio_matrix	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	mass_ratio_matrix.csv	\N	f
235	239	molecular_bundle_fields	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi,phi,eta	npy	intermediate.npy	\N	f
199	200	grad_phi_shell_overlay	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	grad_phi_shell_overlay.npy	\N	f
202	201	grad_phi_shellmap	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	grad_phi_shellmap.npy	\N	f
214	205	identity_clustering_overlay	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	identity_clustering_overlay.npy	\N	f
228	85	magnetic_structure_tracker	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	magnetic_structure_tracker.npy	\N	f
236	239	molecular_bundle_fields	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
237	239	molecular_bundle_fields	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	molecular_bundle_fields.json	\N	f
238	87	molecular_bundle_tracker	1	compute	101	numpy	6	grad	601	gradient	s1_compute	domain_table.csv, domain_tracking.csv	npy	intermediate.npy	\N	f
239	87	molecular_bundle_tracker	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
240	87	molecular_bundle_tracker	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	molecular_bundle_tracker.json	\N	f
241	208	molecule_chain_report	1	compute	101	numpy	6	grad	601	gradient	s1_compute	molecular_bundle_tracker.json, domain_tracking.csv	npy	intermediate.npy	\N	f
242	208	molecule_chain_report	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
243	208	molecule_chain_report	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	molecule_chain_report.json	\N	f
244	209	molecule_report	1	compute	101	numpy	6	grad	601	gradient	s1_compute	molecular_bundle_tracker.json, molecule_chain_report.json, domain_table.csv	npy	intermediate.npy	\N	f
245	209	molecule_report	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
246	209	molecule_report	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	molecule_report.json	\N	f
247	42	p_k	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi	npy	intermediate.npy	\N	f
248	42	p_k	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
249	42	p_k	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	p_k.csv	\N	f
250	91	particle_table	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi,phi,eta	npy	intermediate.npy	\N	f
251	91	particle_table	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
253	91	particle_table	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	particle_table.json	\N	f
254	91	particle_table	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	particle_table.png	\N	f
255	210	periodic_layout_projection	1	compute	101	numpy	6	grad	601	gradient	s1_compute	domain_table.csv psi	npy	intermediate.npy	\N	f
256	210	periodic_layout_projection	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
257	210	periodic_layout_projection	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	periodic_layout_projection.csv	\N	f
258	210	periodic_layout_projection	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	periodic_layout_projection.json	\N	f
259	210	periodic_layout_projection	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	periodic_layout_projection.png	\N	f
260	211	phase_alignment_map	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi, phase_mask.npy	npy	intermediate.npy	\N	f
261	211	phase_alignment_map	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
263	134	phase_coherence_tensor	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi	npy	intermediate.npy	\N	f
264	134	phase_coherence_tensor	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
265	134	phase_coherence_tensor	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	phase_coherence_tensor.csv	\N	f
266	134	phase_coherence_tensor	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	phase_coherence_tensor.json	\N	f
267	240	phase_mask	1	compute	103	scipy.ndimage	1	morph	101	threshold	s1_compute	domain_tracking.csv,domain_table.csv,molecular_bundle_tracker.json	npy	intermediate.npy	\N	f
268	240	phase_mask	2	flatten	103	scipy.ndimage	1	morph	102	connected_components	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
270	240	phase_mask	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	phase_mask.csv	\N	f
271	253	phi	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi|phi|eta	npy	intermediate.npy	\N	f
272	253	phi	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
274	37	phi_active_percent	1	compute	103	scipy.ndimage	1	morph	101	threshold	s1_compute	phi	npy	intermediate.npy	\N	f
275	37	phi_active_percent	2	flatten	103	scipy.ndimage	1	morph	102	connected_components	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
276	37	phi_active_percent	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	phi_active_percent.csv	\N	f
280	213	phi_collapse_shells	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
281	213	phi_collapse_shells	1	compute	101	numpy	6	grad	601	gradient	s1_compute	grad_phi_shellmap.npy	npy	intermediate.npy	\N	f
283	45	phi_entropy	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	phi	npy	intermediate.npy	\N	f
284	45	phi_entropy	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
278	212	phi_collapse_shell_overlay	1	compute	6	graphics	9	serialize	\N		s1_compute	grad_phi_shellmap.npy	npy	intermediate.npy	\N	f
279	212	phi_collapse_shell_overlay	2	flatten	6	graphics	9	serialize	\N		s2_flatten	s1_compute	npy	intermediate.npy	\N	f
285	45	phi_entropy	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	phi_entropy.csv	\N	f
286	214	phi_max	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	phi	npy	intermediate.npy	\N	f
287	214	phi_max	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
288	214	phi_max	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	phi_max.csv	\N	f
289	215	phi_mean	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	phi	npy	intermediate.npy	\N	f
290	215	phi_mean	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
291	215	phi_mean	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	phi_mean.csv	\N	f
292	216	phi_min	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	phi	npy	intermediate.npy	\N	f
294	216	phi_min	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	phi_min.csv	\N	f
300	8	phi_shell_width	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	phi_shell_width.csv	\N	f
301	233	phi_std	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	phi	npy	intermediate.npy	\N	f
302	233	phi_std	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
303	233	phi_std	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	phi_std.csv	\N	f
304	11	pointer_class_histogram	1	compute	101	numpy	6	grad	601	gradient	s1_compute	phi	npy	intermediate.npy	\N	f
305	11	pointer_class_histogram	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
306	11	pointer_class_histogram	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	pointer_class_histogram.csv	\N	f
307	36	pointer_count	1	compute	103	scipy.ndimage	1	morph	101	threshold	s1_compute	phi	npy	intermediate.npy	\N	f
308	36	pointer_count	2	flatten	103	scipy.ndimage	1	morph	102	connected_components	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
309	36	pointer_count	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	pointer_count.csv	\N	f
310	98	pointer_map	1	compute	101	numpy	6	grad	601	gradient	s1_compute	phi	npy	intermediate.npy	\N	f
311	98	pointer_map	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
313	218	pointer_mask	1	compute	103	scipy.ndimage	1	morph	101	threshold	s1_compute	phi	npy	intermediate.npy	\N	f
314	218	pointer_mask	2	flatten	103	scipy.ndimage	1	morph	102	connected_components	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
319	100	pointer_overlay	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	pointer_overlay.json	\N	f
320	100	pointer_overlay	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	pointer_overlay.csv	\N	f
321	100	pointer_overlay	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	pointer_overlay.png	\N	f
327	220	pointer_overlay_bonds	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	pointer_overlay_bonds.png	\N	f
330	221	pointer_overlay_magnetic	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	pointer_overlay_magnetic.png	\N	f
317	100	pointer_overlay	1	compute	6	graphics	9	serialize	\N		s1_compute	pointer_map.npy, domain_table.csv	npy	intermediate.npy	\N	f
318	100	pointer_overlay	2	flatten	6	graphics	9	serialize	\N		s2_flatten	s1_compute	npy	intermediate.npy	\N	f
333	222	pointer_overlay_molecule	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	pointer_overlay_molecule.png	\N	f
336	223	pointer_overlay_polymer	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	pointer_overlay_polymer.png	\N	f
339	224	pointer_overlay_topology	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	pointer_overlay_topology.png	\N	f
322	219	pointer_overlay_atoms	1	compute	6	graphics	9	serialize	903	overlay_image	overlay_buffer	pointer_map.npy|atom_overlay.csv	npy	intermediate.npy	\N	f
323	219	pointer_overlay_atoms	2	flatten	6	graphics	9	serialize	904	overlay_rgb	overlay_rgb	overlay_buffer	npy	intermediate.npy	\N	f
325	220	pointer_overlay_bonds	1	compute	6	graphics	9	serialize	\N		s1_compute	pointer_map.npy,  bond_energy_chain.csv	npy	intermediate.npy	\N	f
326	220	pointer_overlay_bonds	2	flatten	6	graphics	9	serialize	\N		s2_flatten	s1_compute	npy	intermediate.npy	\N	f
297	217	phi_shell_curvature_map	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	phi_shell_curvature_map.npy	\N	f
312	98	pointer_map	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	pointer_map.npy	\N	f
315	218	pointer_mask	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	pointer_mask.npy	\N	f
316	100	pointer_overlay	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	pointer_overlay.npy	\N	f
340	107	pointer_vector_map	1	compute	101	numpy	6	grad	601	gradient	s1_compute	pointer_map.npy, domain_table.csv	npy	intermediate.npy	\N	f
328	221	pointer_overlay_magnetic	1	compute	6	graphics	9	serialize	\N		s1_compute	pointer_map.npy, magnetic_structure_tracker.csv	npy	intermediate.npy	\N	f
329	221	pointer_overlay_magnetic	2	flatten	6	graphics	9	serialize	\N		s2_flatten	s1_compute	npy	intermediate.npy	\N	f
341	107	pointer_vector_map	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
331	222	pointer_overlay_molecule	1	compute	6	graphics	9	serialize	\N		s1_compute	pointer_map.npy, molecular_bundle_tracker.json	npy	intermediate.npy	\N	f
332	222	pointer_overlay_molecule	2	flatten	6	graphics	9	serialize	\N		s2_flatten	s1_compute	npy	intermediate.npy	\N	f
334	223	pointer_overlay_polymer	1	compute	6	graphics	9	serialize	\N		s1_compute	pointer_map.npy, molecular_bundle_tracker.json	npy	intermediate.npy	\N	f
335	223	pointer_overlay_polymer	2	flatten	6	graphics	9	serialize	\N		s2_flatten	s1_compute	npy	intermediate.npy	\N	f
343	107	pointer_vector_map	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	pointer_vector_map.csv	\N	f
344	247	polymer_persistence_tracker	1	compute	101	numpy	6	grad	601	gradient	s1_compute	domain_tracking,domain_bundle_tracking,bond_energy_estimator	npy	intermediate.npy	\N	f
345	247	polymer_persistence_tracker	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
346	247	polymer_persistence_tracker	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	polymer_persistence_tracker.csv	\N	f
347	247	polymer_persistence_tracker	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	polymer_persistence_tracker.json	\N	f
351	26	precision_monitor	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi,phi,eta	npy	intermediate.npy	\N	f
352	26	precision_monitor	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
353	26	precision_monitor	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	precision_monitor.csv	\N	f
354	252	psi	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi|phi|eta	npy	intermediate.npy	\N	f
355	252	psi	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
357	56	psi_anomaly_scanner	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi, domain_table.csv, standard_model_table.csv	npy	intermediate.npy	\N	f
358	56	psi_anomaly_scanner	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
360	56	psi_anomaly_scanner	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	psi_anomaly_scanner.json	\N	f
361	56	psi_anomaly_scanner	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	psi_anomaly_scanner.csv	\N	f
362	56	psi_anomaly_scanner	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	psi_anomaly_scanner.png	\N	f
363	225	psi_class_map	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi.npy, domain_table.csv,standard_model_table.csv	npy	intermediate.npy	\N	f
364	225	psi_class_map	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
365	225	psi_class_map	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	psi_class_map.json	\N	f
366	226	psi_component_i	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi, psi_class_map.json	npy	intermediate.npy	\N	f
367	226	psi_component_i	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
368	226	psi_component_i	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	psi_component_i.png	\N	f
369	4	psi_entropy	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	psi	npy	intermediate.npy	\N	f
370	4	psi_entropy	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
371	4	psi_entropy	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	psi_entropy.csv	\N	f
373	5	psi_entropy_map	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	psi	npy	intermediate.npy	\N	f
374	5	psi_entropy_map	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
375	5	psi_entropy_map	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	psi_entropy_map.png	\N	f
376	110	psi_entropy_proxy	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	psi	npy	intermediate.npy	\N	f
377	110	psi_entropy_proxy	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
378	110	psi_entropy_proxy	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	psi_entropy_proxy.csv	\N	f
350	22	power_spectrum	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	power_spectrum.csv	\N	f
342	107	pointer_vector_map	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	pointer_vector_map.npy	\N	f
349	22	power_spectrum	2	flatten	101	numpy	2	stats	907	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
337	224	pointer_overlay_topology	1	compute	101	numpy	3	overlay	1002	pointers	s1_compute	pointer_map.npy, topology_overlay.csv	npy	intermediate.npy	\N	f
338	224	pointer_overlay_topology	2	flatten	101	numpy	3	overlay	1002	pointers	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
382	7	psi_gradient_magnitude	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	psi_gradient_magnitude.csv	\N	f
383	227	psi_max	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	psi	npy	intermediate.npy	\N	f
384	227	psi_max	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
385	227	psi_max	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	psi_max.csv	\N	f
386	228	psi_mean	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	psi	npy	intermediate.npy	\N	f
387	228	psi_mean	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
388	228	psi_mean	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	psi_mean.csv	\N	f
389	229	psi_min	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	psi	npy	intermediate.npy	\N	f
390	229	psi_min	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
391	229	psi_min	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	psi_min.csv	\N	f
392	48	psi_periodic_projection	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	psi, domain_table	npy	intermediate.npy	\N	f
393	48	psi_periodic_projection	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
395	48	psi_periodic_projection	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	psi_periodic_projection.json	\N	f
396	48	psi_periodic_projection	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	psi_periodic_projection.csv	\N	f
397	48	psi_periodic_projection	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	psi_periodic_projection.png	\N	f
398	20	psi_phase_drift	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi	npy	intermediate.npy	\N	f
399	20	psi_phase_drift	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
400	20	psi_phase_drift	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	psi_phase_drift.csv	\N	f
401	16	psi_phase_drift_gpu	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi	npy	intermediate.npy	\N	f
402	16	psi_phase_drift_gpu	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
403	16	psi_phase_drift_gpu	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	psi_phase_drift_gpu.csv	\N	f
404	19	psi_phase_map	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi	npy	intermediate.npy	\N	f
405	19	psi_phase_map	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
406	19	psi_phase_map	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	psi_phase_map.png	\N	f
407	17	psi_phase_map_gpu	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi	npy	intermediate.npy	\N	f
408	17	psi_phase_map_gpu	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
409	17	psi_phase_map_gpu	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	psi_phase_map_gpu.png	\N	f
410	111	psi_radiation_flow_map	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi	npy	intermediate.npy	\N	f
411	111	psi_radiation_flow_map	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
412	111	psi_radiation_flow_map	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	psi_radiation_flow_map.png	\N	f
416	230	psi_radiation_overlay	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	psi_radiation_overlay.png	\N	f
417	113	psi_std	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	psi	npy	intermediate.npy	\N	f
418	113	psi_std	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
419	113	psi_std	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	psi_std.csv	\N	f
421	55	psi_update_rate_estimator	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	psi, eta, phi	npy	intermediate.npy	\N	f
414	230	psi_radiation_overlay	1	compute	6	graphics	9	serialize	\N		s1_compute	psi_radiation_flow_map.npy, domain_table.csv	npy	intermediate.npy	\N	f
415	230	psi_radiation_overlay	2	flatten	6	graphics	9	serialize	\N		s2_flatten	s1_compute	npy	intermediate.npy	\N	f
422	55	psi_update_rate_estimator	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
423	55	psi_update_rate_estimator	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	psi_update_rate_estimator.csv	\N	f
424	231	reaction_cascade	1	compute	101	numpy	6	grad	601	gradient	s1_compute	usion_events.csv, decay_events.csv, energy_release.json, domain_tracking.csv	npy	intermediate.npy	\N	f
425	231	reaction_cascade	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
426	231	reaction_cascade	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	reaction_cascade.json	\N	f
381	7	psi_gradient_magnitude	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	psi_gradient_magnitude.npy	\N	f
394	48	psi_periodic_projection	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	psi_periodic_projection.npy	\N	f
429	50	reaction_log	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	reaction_log.json	\N	f
430	50	reaction_log	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	reaction_log.csv	\N	f
431	115	reaction_path_finder	1	compute	101	numpy	6	grad	601	gradient	s1_compute	reaction_cascade.json, fusion_events.csv, decay_events.csv, domain_tracking.csv	npy	intermediate.npy	\N	f
432	115	reaction_path_finder	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
433	115	reaction_path_finder	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	reaction_path_finder.csv	\N	f
434	115	reaction_path_finder	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	reaction_path_finder.json	\N	f
435	116	semantic_id_map	1	compute	101	numpy	6	grad	601	gradient	s1_compute	domain_table.csv, standard_model_table.csv	npy	intermediate.npy	\N	f
436	116	semantic_id_map	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
437	116	semantic_id_map	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	semantic_id_map.csv	\N	f
438	116	semantic_id_map	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	semantic_id_map.json	\N	f
439	117	semantic_id_vector	1	compute	101	numpy	6	grad	601	gradient	s1_compute	domain_table.csv, semantic_id_map.csv	npy	intermediate.npy	\N	f
440	117	semantic_id_vector	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
441	117	semantic_id_vector	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	semantic_id_vector.csv	\N	f
442	117	semantic_id_vector	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	semantic_id_vector.json	\N	f
446	54	spin_phase_instability_map	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi.npy, domain_table.csv, phase_coherence_tensor.csv, spin_signature.csv	npy	intermediate.npy	\N	f
447	54	spin_phase_instability_map	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
449	54	spin_phase_instability_map	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	spin_phase_instability_map.json	\N	f
450	54	spin_phase_instability_map	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	spin_phase_instability_map.csv	\N	f
451	21	spin_signature	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi	npy	intermediate.npy	\N	f
452	21	spin_signature	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
453	21	spin_signature	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	spin_signature.csv	\N	f
454	15	spin_signature_gpu	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi	npy	intermediate.npy	\N	f
455	15	spin_signature_gpu	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
457	15	spin_signature_gpu	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	spin_signature_gpu.png	\N	f
458	118	standard_model_table	1	compute	101	numpy	6	grad	601	gradient	s1_compute	domain_table.csv, psi	npy	intermediate.npy	\N	f
459	118	standard_model_table	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
461	118	standard_model_table	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	standard_model_table.json	\N	f
462	232	stellar_zone_clusters	1	compute	101	numpy	5	reduce	502	minmaxmeanstd	s1_compute	stellar_zone_finder.json	npy	intermediate.npy	\N	f
463	232	stellar_zone_clusters	2	flatten	101	numpy	5	reduce	503	hist	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
464	232	stellar_zone_clusters	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	stellar_zone_clusters.json	\N	f
499	29	tracked_domains	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	tracked_domains.json	\N	f
500	29	tracked_domains	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	tracked_domains.png	\N	f
504	13	tracked_domains_gpu	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	tracked_domains_gpu.csv	\N	f
505	13	tracked_domains_gpu	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	tracked_domains_gpu.json	\N	f
506	13	tracked_domains_gpu	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	tracked_domains_gpu.png	\N	f
507	23	two_point_correlation	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi	npy	intermediate.npy	\N	f
508	23	two_point_correlation	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
448	54	spin_phase_instability_map	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	spin_phase_instability_map.npy	\N	f
445	6	sigma8	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	sigma8.csv	\N	f
443	6	sigma8	1	compute	102	pyfftw	2	fft	\N	power_spectrum	s1_compute	psi	npy	intermediate.npy	\N	f
444	6	sigma8	2	flatten	101	numpy	2	stats	907	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
456	15	spin_signature_gpu	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	spin_signature_gpu.npy	\N	f
483	53	topology_extractor	1	compute	100	gudhi	3	topology	306	betti_domain	s1_compute	psi.npy, phi.npy, domain_table.csv	npy	intermediate.npy	\N	f
484	53	topology_extractor	2	flatten	100	gudhi	3	topology	306	betti_domain	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
488	125	topology_map	1	compute	100	gudhi	3	topology	304	betti_partial	s1_compute	topology_extractor.json, domain_table.csv	npy	intermediate.npy	\N	f
509	23	two_point_correlation	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	two_point_correlation.csv	\N	f
510	24	vortex_line_count	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi	npy	intermediate.npy	\N	f
471	234	structure_map_projector	1	compute	6	graphics	9	serialize	\N		s1_compute	domain_tracking.csv, structure_lineage_tracer.png	npy	intermediate.npy	\N	f
472	234	structure_map_projector	2	flatten	6	graphics	9	serialize	\N		s2_flatten	s1_compute	npy	intermediate.npy	\N	f
511	24	vortex_line_count	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
474	235	structure_overlay	1	compute	6	graphics	9	serialize	\N		s1_compute	structure_map_projector.png	npy	intermediate.npy	\N	f
475	235	structure_overlay	2	flatten	6	graphics	9	serialize	\N		s2_flatten	s1_compute	npy	intermediate.npy	\N	f
132	10	eta_curvature	1	compute	103	scipy.ndimage	7	convolve	906	minmaxmeanstd	s1_compute	eta	npy	intermediate.npy	\N	f
477	124	structure_overlay_map	1	compute	6	graphics	9	serialize	\N		s1_compute	psi, phi,grad_phi_shellmap1.npy,pointer_map0.npy,phase_mask2.npy, domain_mask2.npy	npy	intermediate.npy	\N	f
478	124	structure_overlay_map	2	flatten	6	graphics	9	serialize	\N		s2_flatten	s1_compute	npy	intermediate.npy	\N	f
133	10	eta_curvature	2	flatten	103	scipy.ndimage	7	convolve	701	kernel	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
480	236	structure_overlay_map_fields	1	compute	6	graphics	9	serialize	\N		s1_compute	psi,phi	npy	intermediate.npy	\N	f
481	236	structure_overlay_map_fields	2	flatten	6	graphics	9	serialize	\N		s2_flatten	s1_compute	npy	intermediate.npy	\N	f
152	46	eta_shear_gradient	1	compute	103	scipy.ndimage	7	convolve	906	minmaxmeanstd	s1_compute	eta, psi	npy	intermediate.npy	\N	f
153	46	eta_shear_gradient	2	flatten	103	scipy.ndimage	7	convolve	701	kernel	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
185	197	grad_phi	1	compute	103	scipy.ndimage	7	convolve	906	minmaxmeanstd	s1_compute	phi	npy	intermediate.npy	\N	f
186	197	grad_phi	2	flatten	103	scipy.ndimage	7	convolve	701	kernel	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
489	125	topology_map	2	flatten	9	serialize	3	topology	304	betti_partial	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
188	38	grad_phi_max	1	compute	103	scipy.ndimage	7	convolve	906	minmaxmeanstd	s1_compute	phi	npy	intermediate.npy	\N	f
189	38	grad_phi_max	2	flatten	103	scipy.ndimage	7	convolve	701	kernel	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
191	198	grad_phi_mean	1	compute	103	scipy.ndimage	7	convolve	906	minmaxmeanstd	s1_compute	phi	npy	intermediate.npy	\N	f
192	198	grad_phi_mean	2	flatten	103	scipy.ndimage	7	convolve	701	kernel	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
194	199	grad_phi_min	1	compute	103	scipy.ndimage	7	convolve	906	minmaxmeanstd	s1_compute	phi	npy	intermediate.npy	\N	f
195	199	grad_phi_min	2	flatten	103	scipy.ndimage	7	convolve	701	kernel	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
197	200	grad_phi_shell_overlay	1	compute	103	scipy.ndimage	7	convolve	906	minmaxmeanstd	s1_compute	grad_phi.npy, grad_phi_shellmap.npy	npy	intermediate.npy	\N	f
198	200	grad_phi_shell_overlay	2	flatten	103	scipy.ndimage	7	convolve	701	kernel	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
200	201	grad_phi_shellmap	1	compute	103	scipy.ndimage	7	convolve	906	minmaxmeanstd	s1_compute	grad_phi.npy, phi	npy	intermediate.npy	\N	f
201	201	grad_phi_shellmap	2	flatten	103	scipy.ndimage	7	convolve	701	kernel	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
203	202	grad_phi_std	1	compute	103	scipy.ndimage	7	convolve	906	minmaxmeanstd	s1_compute	phi	npy	intermediate.npy	\N	f
204	202	grad_phi_std	2	flatten	103	scipy.ndimage	7	convolve	701	kernel	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
295	217	phi_shell_curvature_map	1	compute	103	scipy.ndimage	7	convolve	906	minmaxmeanstd	s1_compute	phi_collapse_shells.npy, grad_phi_shellmap.npy	npy	intermediate.npy	\N	f
296	217	phi_shell_curvature_map	2	flatten	103	scipy.ndimage	7	convolve	701	kernel	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
298	8	phi_shell_width	1	compute	103	scipy.ndimage	7	convolve	906	minmaxmeanstd	s1_compute	phi	npy	intermediate.npy	\N	f
299	8	phi_shell_width	2	flatten	103	scipy.ndimage	7	convolve	701	kernel	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
379	7	psi_gradient_magnitude	1	compute	103	scipy.ndimage	7	convolve	906	minmaxmeanstd	s1_compute	psi	npy	intermediate.npy	\N	f
380	7	psi_gradient_magnitude	2	flatten	103	scipy.ndimage	7	convolve	701	kernel	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
492	237	topology_overlay	1	compute	101	numpy	10	overlay	1001	compose	s1_compute	psi|phi|eta	npy	intermediate.npy	\N	f
493	237	topology_overlay	2	flatten	101	numpy	10	overlay	1001	compose	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
16	130	bond_energy_estimator	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	molecular_bundle_tracker.json, domain_table.csv	npy	intermediate.npy	\N	f
23	180	bond_energy_stats	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	bond_energy_chain.csv	npy	intermediate.npy	\N	f
24	180	bond_energy_stats	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
26	52	bond_lifetime_estimator	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	molecular_bundle_tracker.json, domain_table.csv	npy	intermediate.npy	\N	f
27	52	bond_lifetime_estimator	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
512	24	vortex_line_count	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	vortex_line_count.csv	\N	f
513	35	xi_r	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi	npy	intermediate.npy	\N	f
514	35	xi_r	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
61	32	decay_events	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	psi	npy	intermediate.npy	\N	f
62	32	decay_events	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
64	185	domain_bundle_tracking	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	domain_table.csv or json, domain_tracking.csv or json, domain_geometry.csv	npy	intermediate.npy	\N	f
65	185	domain_bundle_tracking	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
76	67	domain_hash_map	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	domain_table.csv, domain_table.json	npy	intermediate.npy	\N	f
77	67	domain_hash_map	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
86	27	domain_persistence	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	phi	npy	intermediate.npy	\N	f
87	27	domain_persistence	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
167	194	fusion_chain_graph	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	fusion_events.csv	npy	intermediate.npy	\N	f
168	194	fusion_chain_graph	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
170	195	fusion_decay_matrix	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	fusion_events.csv, decay_events.csv	npy	intermediate.npy	\N	f
171	195	fusion_decay_matrix	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
173	31	fusion_events	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	psi	npy	intermediate.npy	\N	f
174	31	fusion_events	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
176	196	fusion_events_summary	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	fusion_events.csv	npy	intermediate.npy	\N	f
177	196	fusion_events_summary	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
427	50	reaction_log	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	psi, phi, eta, bond_energy_chain.csv	npy	intermediate.npy	\N	f
428	50	reaction_log	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
496	29	tracked_domains	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	psi,phi,eta	npy	intermediate.npy	\N	f
497	29	tracked_domains	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
502	13	tracked_domains_gpu	1	compute	101	numpy	8	stats	804	overlay_buffer	s1_compute	psi,phi,eta	npy	intermediate.npy	\N	f
503	13	tracked_domains_gpu	2	flatten	101	numpy	8	stats	801	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
348	22	power_spectrum	1	compute	102	pyfftw	2	fft	\N	power_spectrum	s1_compute	psi	npy	intermediate.npy	\N	f
117	188	drift_vector_map	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	drift_vector_map.npy	\N	f
131	254	eta	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	eta.npy	\N	f
134	10	eta_curvature	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	eta_curvature.npy	\N	f
136	76	eta_flux_map	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	eta_flux_map.npy	\N	f
252	91	particle_table	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	particle_table.npy	\N	f
262	211	phase_alignment_map	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	phase_alignment_map.npy	\N	f
515	35	xi_r	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	xi_r.csv	\N	f
221	207	magnetic_structure_field	1	compute	101	numpy	6	grad	601	gradient	s1_compute	psi	npy	intermediate.npy	\N	f
3	244	atom_overlay	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	atom_overlay.png	1	f
10	177	BField_overlay	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	BField_overlay.png	\N	f
11	177	BField_overlay	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	BField_overlay.csv	\N	f
12	177	BField_overlay	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	BField_overlay.json	\N	f
269	240	phase_mask	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	phase_mask.npy	\N	f
273	253	phi	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	phi.npy	\N	f
277	212	phi_collapse_shell_overlay	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	phi_collapse_shell_overlay.npy	\N	f
282	213	phi_collapse_shells	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	phi_collapse_shells.npy	\N	f
324	219	pointer_overlay_atoms	3	final	104	writer	9	serialize	903	overlay_image	overlay_image	overlay_rgb	png	pointer_overlay_atoms.png	\N	f
356	252	psi	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	psi.npy	\N	f
359	56	psi_anomaly_scanner	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	psi_anomaly_scanner.npy	\N	f
372	5	psi_entropy_map	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	psi_entropy_map.npy	\N	f
413	230	psi_radiation_overlay	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	psi_radiation_overlay.npy	\N	f
420	55	psi_update_rate_estimator	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	psi_update_rate_estimator.npy	\N	f
460	118	standard_model_table	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	standard_model_table.npy	\N	f
501	13	tracked_domains_gpu	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	tracked_domains_gpu.npy	\N	f
495	29	tracked_domains	3	final	104	writer	6	serialize	604	npy	final_npy	s2_flatten	npy	tracked_domains.npy	\N	f
465	120	stellar_zone_finder	1	compute	101	numpy	6	grad	601	gradient	s1_compute	domain_table.csv, mass_ratio_matrix.csv, psi,eta	npy	intermediate.npy	\N	f
466	120	stellar_zone_finder	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
467	120	stellar_zone_finder	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	stellar_zone_finder.json	\N	f
468	121	structure_lineage_tracer	1	compute	101	numpy	6	grad	601	gradient	s1_compute	domain_tracking.csv, domain_table.csv	npy	intermediate.npy	\N	f
469	121	structure_lineage_tracer	2	flatten	101	numpy	6	grad	908	minmaxmeanstd	s2_flatten	s1_compute	npy	intermediate.npy	\N	f
470	121	structure_lineage_tracer	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	structure_lineage_tracer.png	\N	f
473	234	structure_map_projector	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	structure_map_projector.png	\N	f
476	235	structure_overlay	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	structure_overlay.png	\N	f
479	124	structure_overlay_map	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	structure_overlay_map.png	\N	f
482	236	structure_overlay_map_fields	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	structure_overlay_map_fields.png	\N	f
485	53	topology_extractor	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	topology_extractor.json	\N	f
486	53	topology_extractor	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	topology_extractor.csv	\N	f
487	53	topology_extractor	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	topology_extractor.png	\N	f
490	125	topology_map	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	topology_map.png	\N	f
491	125	topology_map	3	final	104	writer	9	serialize	902	json	final_json	s2_flatten	json	topology_map.json	\N	f
494	237	topology_overlay	3	final	104	writer	9	serialize	905	png	final_png	s2_flatten	png	topology_overlay.png	\N	f
498	29	tracked_domains	3	final	104	writer	9	serialize	901	csv	final_csv	s2_flatten	csv	tracked_domains.csv	\N	f
\.


--
-- Data for Name: metrics; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.metrics (id, name, description, requiredfields, outputtypes, isstandard, postprocessing, filename, group_id) FROM stdin;
4	psi_entropy	Mean entropy proxy (∇ψ²)	psi	csv	t	0	\N	6
25	betti_numbers	Betti numbers from collapse mask	collapse_mask.npy	csv	t	1	\N	2
111	psi_radiation_flow_map		psi	png	t	0	\N	4
98	pointer_map	Pointer (fixed domain) region map in simulation field.	phi	npy	t	0	\N	4
214	phi_max	Calculates the maximum value of the φ (permission/collapse) field in each simulation frame, identifying the most open or least collapsed region. Used to track regions resisting collapse or to audit field bounds.	phi	csv	t	0	\N	1
215	phi_mean	Computes the mean (average) φ value over the entire simulation grid per frame, giving a global measure of the gating state and collapse progress	phi	csv	t	0	\N	1
132	domain_vector_drift	Measures drift (velocity or centroid change) of domain vectors over time, used for tracking migration, flow, or causal domain dynamics.	domain_table.csv,psi	csv,json	t	1	\N	3
85	magnetic_structure_tracker	Tracks field or domain magnetic-like structures and rotational features.	psi, domain_table.csv	csv,json,npy	t	1	\N	6
195	fusion_decay_matrix	Constructs a matrix or graph showing the relationships, transitions, and possible pathways among all detected fusion and decay events. This JSON output encodes, for every pair of domains, the likelihood or count of transitions (fusion or decay), enabling the study of structural evolution and decay chains in the simulation.	fusion_events.csv, decay_events.csv	json	t	1	\N	6
210	periodic_layout_projection	Generates a PNG visualization that projects or unwraps the simulation grid into a periodic or toroidal layout, allowing inspection of how domains, particles, or field structures wrap around boundaries. This metric highlights spatial continuity, periodic connectivity, and topological features, supporting both qualitative audit and visual publication. It processes outputs from prior segmentation, mask, or field summary metrics, mapping them into a periodic tiling or 2D/3D projection.	domain_table.csv psi	csv,json,png	t	1	\N	4
5	psi_entropy_map	GPU entropy map (∇ψ² 3D field)	psi	npy,png	t	0	\N	4
6	sigma8	σ₈ smoothed ψ² statistic	psi	csv	t	0	\N	1
7	psi_gradient_magnitude	||∇ψ|| magnitude field	psi	csv,npy	t	0	\N	4
8	phi_shell_width	Width of collapse shell in φ	phi	csv	t	0	\N	2
9	component_density	ψⁱ component-wise densities	psi	csv	t	0	\N	6
10	eta_curvature	∇²η curvature	eta	csv,npy	t	0	\N	2
11	pointer_class_histogram	Domain pointer class histogram	phi	csv	t	0	\N	6
12	domain_tracking	Domain centroid + ψ vector	psi,phi,eta	csv,json,npy,png	t	0	\N	3
13	tracked_domains_gpu	GPU domain tracker	psi,phi,eta	csv,json,npy,png	t	0	\N	6
14	flux_map_gpu	GPU ψ flux field	psi,phi	npy,png	t	0	\N	6
41	domain_count	Number of segmented pointer domains	pointer_mask.npy	csv	t	1	\N	1
16	psi_phase_drift_gpu	GPU global phase drift	psi	csv	t	0	\N	6
17	psi_phase_map_gpu	GPU ψ phase map image	psi	png	t	0	\N	6
18	gauge_drift	Gauge drift from φ across time	phi	csv	t	0	\N	6
19	psi_phase_map	Phase map image (CPU)	psi	png	t	0	\N	4
20	psi_phase_drift	Phase drift between frames	psi	csv	t	0	\N	6
22	power_spectrum	P(k) power spectrum	psi	csv	t	0	\N	5
23	two_point_correlation	ξ(r) 2-point correlation	psi	csv	t	0	\N	5
24	vortex_line_count	Topological line count	psi	csv	t	0	\N	6
26	precision_monitor	NaN / overflow checker	psi,phi,eta	csv	t	0	\N	6
27	domain_persistence	Domain presence over time	phi	csv	t	0	\N	3
28	global_phase_drift	Global ψ phase drift over time	psi	csv	t	0	\N	6
245	eta_integral	Sums all η values across the simulation grid per frame, representing total accumulated memory or entropy.	eta	csv	t	0	\N	1
31	fusion_events	Detects and logs domain fusion events between frames	psi	csv	f	0	\N	3
32	decay_events	Detects and logs vanished ψ-domains between frames	psi	csv	f	0	\N	3
33	domain_charge	Assigns discrete charge to ψ-domains by component dominance	psi	csv	f	0	\N	3
35	xi_r	Computes radial correlation function ξ(r) from ψ	psi	csv	f	0	\N	5
251	collapse_mask	Threshold φ to binary mask and run connected-components to produce labeled domains (pp0).	phi	npy,csv	t	0	collapse_mask	2
30	domain_geometry	Centroid, radius, and anisotropy of each ψ-domain	psi	csv	f	0	\N	3
43	domain_table	Extract ψⁱ, ηⁱ, φⁱ, COM, and volume per domain	psi, phi, eta	json,csv	f	0	\N	3
21	spin_signature	Calculates the spin signature for each domain or region in the simulation by analyzing the ψ field’s symmetry properties. For every domain, it determines spin class, parity, and related quantum numbers, outputting a CSV table suitable for audit, clustering, or comparison to theoretical predictions.	psi	csv	t	0	\N	6
40	coherence_length	ψ autocorrelation span (coherence length)	psi	csv	t	0	\N	1
37	phi_active_percent	Percentage of voxels with φ above collapse threshold	phi	csv	t	0	\N	6
48	psi_periodic_projection	Project ψⁱ domain vectors to semantic identity grid	psi, domain_table	json,csv,png,npy	f	1	\N	6
52	bond_lifetime_estimator	TraCalculates the lifespan of each detected bond across the simulation time series, tracking the creation and dissolution of bonds between domains or particles. For each bond, this metric records its start and end frames, total lifetime, and any significant state transitions. The output allows analysis of bond stability, transient structure, and dynamics of molecular or domain interactions.ck lifetime, strength, and stability of ψⁱ bonds	molecular_bundle_tracker.json, domain_table.csv	json,csv	f	2	\N	6
65	charge_vector_map	Computes the spatial map of charge vectors across the simulation grid or per domain by analyzing ψ, η, and other relevant fields. The resulting vector map is used for further analysis of charge distribution and orientation, especially in domains or structural regions.	domain_table.csv	csv,npy	t	1	\N	6
230	psi_radiation_overlay		psi_radiation_flow_map.npy, domain_table.csv	png,npy	t	1	\N	4
130	bond_energy_estimator	This metric produces a structured audit log and metadata record for each bond energy estimation run, capturing the method, parameters, summary errors, run statistics, and key configuration details. It serves as a forensic record of the entire bond energy calculation, supporting reproducibility and debugging. It does not compute new bond energies but assembles all relevant metadata from the primary bond energy calculation step.	molecular_bundle_tracker.json, domain_table.csv	json	t	2	\N	6
67	domain_hash_map	Analyzes output domain tables and field data, assigning each detected domain a unique, reproducible hash (identity fingerprint) based on local field values or geometry.	domain_table.csv, domain_table.json	json,csv	t	1	\N	3
75	energy_release_estimator	Calculates quantitative estimates of field energy released for each event (fusion, decay, collapse, etc.) by processing relevant domain and event files, and outputs a detailed CSV report. This includes per-event energy, participating domain IDs, and relevant state parameters. The output supports statistical analysis, plotting, or further scientific review.	fusion_events.csv, decay_events.csv, domain_table.csv	csv	t	1	\N	6
50	reaction_log	Causal trace of bond swaps and rebindings	psi, phi, eta, bond_energy_chain.csv	json,csv	f	2	\N	6
246	branch_point_detector	Detects ψⁱ domains bonded to more than two other domains, indicating chain branching violations.	domain_table,domain_tracking	csv,json	f	1	\N	6
247	polymer_persistence_tracker	Tracks uninterrupted ψⁱ polymer chain persistence, length, and domain count across time.	domain_tracking,domain_bundle_tracking,bond_energy_estimator	csv,json	f	3	\N	6
248	entropy_plateau_analyzer	Verifies ψ entropy does not decline throughout chain growth; plateaus or increases are valid.	psi_entropy_proxy	csv,json	f	1	\N	6
76	eta_flux_map	Map of η (memory field) fluxes or gradients across simulation grid.	eta	csv,npy	t	1	\N	4
87	molecular_bundle_tracker	Analyzes output from primary domain segmentation and tracking (e.g., domain_table.csv, domain_tracking.csv) to identify and track composite bundles over time, grouping domains into persistent aggregates (molecules) and recording their temporal evolution.	domain_table.csv, domain_tracking.csv	json	t	1	\N	6
45	phi_entropy	Calculates the Shannon entropy of the φ (permission/gating) field across the entire simulation grid for each frame. This metric quantifies the degree of disorder, unpredictability, or inhomogeneity in the φ field, providing a direct statistical measure of how spread out or mixed the collapse permission is in space. High entropy indicates a highly variable or fragmented collapse state, while low entropy reflects uniform gating or well-defined collapse fronts. The output CSV records the entropy value per frame and can be used to audit collapse progression, identify regions of collapse irregularity, and compare simulation runs or parameter regimes.	phi	csv	t	0	\N	6
72	drift_tracker	Analyzes outputs from primary domain metrics to track the motion (drift) of each domain’s centroid and identity over time. It reconstructs trajectories, velocities, and lineage for all domains or pointers, producing both a detailed time series (csv) and a structured, audit-ready summary (json). This metric enables studies of migration, flow, and causal domain dynamics.	domain_tracking.csv, domain_table.csv	json, csv	t	1	\N	6
131	domain_vector_alignment	Calculates alignment (e.g., mean cosine/dot product) between domain orientation vectors to analyze collective order, flow, or polarization.	domain_table.csv,psi	csv,json	t	1	\N	3
46	eta_shear_gradient	Compute η field shear, curl, and drift alignment	eta, psi	npy,csv,png	f	0	\N	6
42	p_k	Power spectrum of ψ slice	psi	csv	t	0	\N	5
240	phase_mask	Produces a binary or labeled mask (phase_mask.npy) for a specified region, chain, bundle, or domain set, based on domain_tracking, domain_table, or bundle_tracker. Used as input to phase alignment and region-specific analysis metrics.	domain_tracking.csv,domain_table.csv,molecular_bundle_tracker.json	npy,csv	t	2	\N	4
100	pointer_overlay	Produces a multi-format overlay visualization of pointer domains/regions, combining pointer masks, domain segmentation, field maps, and possibly molecular or topological annotations. Each overlay highlights a specific analytic perspective.	pointer_map.npy, domain_table.csv	json,csv,png,npy	t	1	\N	4
115	reaction_path_finder	Analyzes the causal paths within reaction or domain transformation chains, mapping all possible routes from initial to final states in domain evolution. This metric reconstructs the network of events—such as fusion, decay, and energy release—using prior event logs and tracking data. It outputs a comprehensive mapping of all possible reaction paths, reporting detailed chains, branchings, and cyclic or convergent transformations. Results include a CSV listing all path sequences and transitions, and a structured JSON with full path metadata and properties for audit, further analysis, or visualization.	reaction_cascade.json, fusion_events.csv, decay_events.csv, domain_tracking.csv	csv,json	t	3	\N	6
116	semantic_id_map	Analyzes outputs of prior segmentation and identity metrics (domain_table.csv, standard_model_table.csv, clustering_assignment.csv) to generate a semantic ID/classification map for all detected domains, including additional trait or type information as available.	domain_table.csv, standard_model_table.csv	csv,json	t	2	\N	6
117	semantic_id_vector	Generates per-domain or per-region trait vectors (numeric, categorical, or vector embeddings) representing semantic types, field structure, or class assignments. Outputs a CSV and JSON with vectorized trait data for downstream clustering, visualization, or audit. Relies on prior segmentation and ID mapping outputs.	domain_table.csv, semantic_id_map.csv	csv,json	t	3	\N	6
120	stellar_zone_finder	Identifies and records all high-mass or high-activity clusters (stellar zones) within the simulation, corresponding to analogs of stars or dense/high-energy regions in field or domain space. The metric scans outputs from domain detection and mass/energy analysis, applying thresholding or clustering to find candidate stellar regions. It produces a structured JSON report detailing each detected zone’s properties (location, total mass/energy, activity, constituent domains), supporting both audit and further quantitative or visual analysis.	domain_table.csv, mass_ratio_matrix.csv, psi,eta	json	t	2	\N	6
124	structure_overlay_map	Map combining field structure overlProjects the spatial and/or ancestry map of all tracked structures into a 2D or 3D visualization, typically as a PNG. This metric aggregates spatial or temporal lineage data from tracking outputs, enabling visual inspection of ancestry, structural relationships, or migration paths.ays.	psi, phi,grad_phi_shellmap1.npy,pointer_map0.npy,phase_mask2.npy, domain_mask2.npy	png	t	3	\N	4
252	psi	raw ψ field from simulation		npy	t	0	\N	4
253	phi	raw φ field from simulation		npy	t	0	\N	4
254	eta	raw η field from simulation		npy	t	0	\N	4
91	particle_table	Detects, classifies, and tabulates all particle or domain types directly from the simulation fields (typically ψ, η, φ), assigning quantum/classical identities, structure, and labels per domain. This metric creates a full per-particle/domain table, mapping field-derived properties without relying on any post-processed segmentation.	psi,phi,eta	json, png, npy	t	0	\N	6
107	pointer_vector_map	Computes and outputs a vector field representing the local orientation, direction, or drift of pointer domains/regions, using spatial analysis of the pointer mask	pointer_map.npy, domain_table.csv	csv,npy	t	1	\N	6
113	psi_std	Standard deviation of psi field for each frame.	psi	csv	t	0	\N	1
118	standard_model_table	Maps each detected domain or region in the simulation to its closest Standard Model (SM) analog, based on ψ, η, φ, and possibly additional classification outputs. For every domain, the metric assigns a best-fit SM identity (e.g., electron, quark, neutrino), quantum numbers, and class properties, producing a structured JSON table for audit and a NPY array for per-domain label masks or embeddings. This mapping is crucial for connecting simulation results to physical particle analogs, performing identity audit, and supporting automated downstream analysis.	domain_table.csv, psi	json,npy	t	1	\N	6
121	structure_lineage_tracer	Tracks the ancestry and structural lineage of each domain or object across time, building a time-resolved PNG (or series of PNGs) visualizing how structures split, merge, persist, or disappear throughout the simulation. Each output highlights ancestry chains, lineage persistence, and structural evolution for forensic review and visualization	domain_tracking.csv, domain_table.csv	png	t	1	\N	6
110	psi_entropy_proxy	Proxy for entropy (disorder) of the psi (coherence) field.	psi	csv	t	0	\N	1
179	bond_energy_overlay	Generates a composite visualization overlay that displays bond energies and spatial relationships between domains or molecules, mapped onto the simulation grid or a projection. This overlay helps with the inspection and publication of bond patterns, energy hotspots, or network connectivity, using data from previous bond analysis metrics and domain segmentation.	bond_energy_chain.csv, domain_table.csv, pointer_map.npy	csv,json	t	2	\N	4
177	BField_overlay	Calculates and visualizes the spatial magnetic (B-field) signature across the simulation grid, derived from the curl of the ψ field (∇×ψ). This metric overlays the B-field direction and magnitude on a background field image (such as ψ or φ), producing a composite visualization of local and global magnetic structure, suitable for inspection, publication, or further quantification.	pointer_map.npy, domain_table.csv, psi	png, csv, json	t	1	\N	4
178	bond_energy_chain	Calculates and records every detected bond between domains or molecular subunits for the current simulation frame. For each bond, this metric stores identifiers for the bonded domains, their bond energy, and other properties such as bond type or spatial relation. The output is a comprehensive bond-by-bond record suitable for structural audit or downstream analysis.	domain_table.csv	csv	t	1	\N	6
180	bond_energy_stats	Aggregates and summarizes the energies of all bonds computed by bond_energy_chain, providing statistics such as mean, standard deviation, minimum, maximum, and possibly histograms or grouped energy classes. This metric produces summary statistics for the bond energy distribution, used for reporting, visualization, and theory comparison.	bond_energy_chain.csv	csv	t	2	\N	6
181	collapse_echo_overlay	Generates PNG overlays that visualize detected echo rings on top of the ψ or φ field, highlighting spatial structure and propagation.	phi, psi, collapse_echo_shellmap.npy	png	t	2	\N	4
182	collapse_echo_profile	Identifies and logs every detected echo ring. For each ring, outputs timing, radius, ψⁱ coherence, and Δη in a structured JSON.	phi, psi, eta	json	t	0	\N	2
183	collapse_echo_shellmap	Writes a 3D NPY array that maps the shell/ring regions for each frame, assigning each voxel a shell/echo ID or label.	psi,phi	npy	t	0	\N	4
184	collapse_echo_summary	Produces a summary CSV with frame number, ring radius, entropy, ψⁱ decay, echo ID, amplitude, and other statistics for each echo across the time series.	phi, psi, eta	csv	t	0	\N	2
185	domain_bundle_tracking	domain_bundle_tracking is a post-processing (pp=1) metric that analyzes the relationships and connectivity among multiple domains, identifying bundles (e.g., molecular clusters or physically connected aggregates) from the outputs of regular domain metrics. It generates a structured JSON file recording bundle membership, bundle statistics (number of domains, spatial extent, persistence, etc.), and potentially temporal properties (formation/dissolution events) based on the results of metrics like domain_table, domain_tracking, and domain_geometry.	domain_table.csv or json, domain_tracking.csv or json, domain_geometry.csv	json	t	1	\N	3
186	domain_map_summary	Aggregates and summarizes properties from prior domain metrics (including hash_map, domain_table, and potentially domain_persistence), outputting high-level statistics, frequency counts, and summary labels for all domains across time or space.	domain_hash_map.csv, domain_table.csv, domain_persistence.csv	csv	t	2	\N	3
187	domain_mask	Creates a final visualization mask or PNG overlay for all domains, using the output of earlier metrics (domain_table, domain_hash_map, or pointer_map).	domain_table.csv, domain_hash_map.csv	png	t	2	\N	3
190	eta_max	Calculates the maximum value of η across the entire simulation grid per frame, identifying peak memory or accumulation points.	eta	csv	t	0	\N	1
191	eta_mean	Computes the mean (average) η value across all voxels, tracking overall memory or entropy accumulation per frame.	eta	csv	t	0	\N	1
192	eta_min	Calculates the minimum η value, providing a floor for memory depletion or erasure in each frame.	eta	csv	t	0	\N	1
193	eta_std	Computes the standard deviation of η, quantifying the spread or variability of the memory field.	eta	csv	t	0	\N	1
188	drift_vector_map	Generates a 3D NPY field of per-voxel drift vectors, showing the spatial map of local domain motion over the simulation. This post-post-processing metric computes voxelwise or regional drift direction and magnitude using the outputs of drift_tracker (json/csv) and domain_tracking, enabling visualization and quantitative flow analysis.	eta_shear_gradient.npy, domain_tracking.csv	npy	t	1	\N	6
189	energy_release	Compiles a structured audit log (JSON) of all energy-releasing events in the simulation, such as domain formation, fusion, decay, or collapse, based on the output of domain-level primary metrics. For each event, the metric logs the type, timing, domains involved, and total energy released, enabling a full energy budget audit and event timeline for forensic and physical review.	fusion_events.csv, decay_events.csv, domain_table.csv	csv,json	t	1	\N	6
194	fusion_chain_graph	Analyzes the time series of fusion events to reconstruct and output a JSON graph that captures the entire chain (ancestry, lineage) of domain fusions over the simulation. Each node represents a domain, and each edge represents a fusion event (with time, involved IDs, and properties). The result is a fully linked fusion tree/graph for scientific and audit review.	fusion_events.csv	json	t	1	\N	6
196	fusion_events_summary	Produces a structured JSON file summarizing all fusion events detected by the regular (pp=0) metric fusion_events. The JSON includes detailed metadata, configuration, event aggregation, and may be used for downstream audit or pipeline handoff.	fusion_events.csv	json	t	1	\N	6
198	grad_phi_mean	Computes the mean (average) gradient magnitude of φ per frame, giving a measure of the typical collapse front strength or shell boundary in the simulation. Used for smoothness and completeness analysis.	phi	csv	t	0	\N	1
199	grad_phi_min	Computes the minimum gradient magnitude of φ in each frame, detecting regions of maximal smoothness or uncollapsed volume.	phi	csv	t	0	\N	1
202	grad_phi_std	Calculates the standard deviation of φ gradient magnitudes, quantifying the spatial variability and roughness of collapse fronts or shell boundaries. Essential for statistical shell analysis.	phi	csv	t	0	\N	1
203	identity_clustering_hierarchical	Performs hierarchical agglomerative clustering on ψⁱ domain identity vectors, using psi.npy and domain_table.csv. Builds a dendrogram or tree structure of domains based on identity vector similarity, records cut thresholds, assigns domains to hierarchical clusters at specified levels, and outputs full tree/assignment details. Enables comparison with k-means, exploration of nested families or gradated identity structures.	psi, domain_table.csv	csv,json	t	1	\N	6
204	identity_clustering_kmeans	Performs k-means clustering on ψⁱ domain identity vectors extracted from the simulation, using the current frame’s psi.npy and the list of domains in domain_table.csv. Assigns each domain to a cluster based on proximity in identity vector space, determines centroids, and outputs both the cluster assignments and full clustering metadata. Used for population analysis, family detection, and quantitative identity audit	psi, domain_table.csv	csv,json	t	1	\N	6
207	magnetic_structure_field 	Analyzes the global and local magnetic structure of the simulation by computing the curl (∇×ψ) and related rotational features directly from the ψ field. This metric produces a spatial map of magnetic-like activity, such as vorticity and rotational coherence, across the entire simulation grid, without reference to segmented domains. The output enables audit and visualization of emergent rotational order, detection of vortices or magnetic cores, and comparison of field-wide magnetic signatures over time. Used for whole-system diagnostics, visualization, and foundational studies of field-theoretic or condensed-matter analogues.	psi	csv,json,png	t	0	\N	6
208	molecule_chain_report	Compiles a high-level, structured JSON report on all molecule (bundle) chains, using the output of molecular_bundle_tracker and/or domain tracking. Summarizes bundle lifespans, fusion/split events, lineage, and chain persistence over time for audit and review.	molecular_bundle_tracker.json, domain_tracking.csv	json	t	2	\N	6
197	grad_phi	Computes the full spatial gradient vector field ∇φ (dφ/dx, dφ/dy, dφ/dz) at every grid point, from the original φ simulation data. Used for all further analysis and statistics on collapse front sharpness and shell structure.	phi	npy	t	0	\N	2
200	grad_phi_shell_overlay	Generates an overlay (NPY) that visually or numerically aligns detected shell regions with the computed φ gradient field. Used for visualization and quantitative spatial comparison of shell and gradient features.	grad_phi.npy, grad_phi_shellmap.npy	npy	t	2	\N	4
201	grad_phi_shellmap	Produces a 3D NPY array assigning each voxel a shell or ring label, determined by analysis of the φ gradient field (and optionally φ or mask overlays). Enables precise tracking and segmentation of shell structures.	grad_phi.npy, phi	npy	t	1	\N	4
205	identity_clustering_overlay	Generates a spatial or vector field overlay (as PNG or NPY) visualizing the results of a selected clustering assignment (from either k-means or hierarchical metrics). Each domain is rendered with color or pattern according to its cluster, projected on the simulation grid or a 2D/3D slice, for publication, inspection, or spatial analysis.	identity_clustering_kmeans.csv,identity_clustering_hierarchical.csv, domain_table.csv	png,npy	t	2	\N	4
206	lineage_chain_graph	Constructs a detailed JSON graph representing the full lineage (ancestry and descent) of all domains or pointers over time, based on fusion, split, drift, and persistence data. Each node in the graph corresponds to a domain instance, with edges recording fusion events, splits, or time continuity. The output enables forensic audit of domain ancestry, reconstructs the entire chain of emergence, fusion, and decay, and is foundational for causal history and identity inheritance studies.	fusion_events.csv, decay_events.csv, domain_persistence.csv, domain_tracking.csv	json	t	1	\N	6
209	molecule_report	Generates a detailed JSON report on all detected molecules/bundles in the simulation, integrating all available bundle metrics and temporal/structural info for forensic review or downstream analysis.	molecular_bundle_tracker.json, molecule_chain_report.json, domain_table.csv	json	t	3	\N	6
211	phase_alignment_map	Calculates and outputs a 3D NPY field where each voxel records the local phase alignment, i.e., the degree to which ψ (or other field) vectors in the surrounding neighborhood are aligned in phase or direction. Used to visualize and quantify regions of high/low phase regularity, detect phase defects or boundaries, and study phase coherence structures across the grid. Operates on simulation field outputs and/or post-processed summary statistics.	psi, phase_mask.npy	npy	t	3	\N	4
86	mass_ratio_matrix	Computes the full matrix of mass ratios between all detected domains, using the integrated η (or ψ) field values as the definition of mass for each domain. The metric requires the output of a domain segmentation metric (domain_table.csv) to identify all domains and their extents. It outputs both a CSV matrix of pairwise ratios and a structured JSON summary for audit and pipeline use. This analysis is fundamental for studying mass hierarchy, domain clustering, and emergent structure in the simulation.	domain_table.csv, eta	json,csv	t	1	\N	6
212	phi_collapse_shell_overlay	Generates a composite overlay in NPY format that visualizes and aligns the detected shell regions (from grad_phi_shellmap.npy) with the original φ field (phi.npy) and/or an external shell mask (shell_mask.npy). This overlay highlights the spatial correspondence and accuracy of shell segmentation, enabling visual audit and publication-quality figures that display collapse shells in the context of raw field data. It is particularly useful for validating shell detection methods, comparing different thresholding or gradient schemes, and communicating results.	grad_phi_shellmap.npy	npy	t	2	\N	4
216	phi_min	Calculates the minimum φ value in each frame, detecting regions of maximal collapse or complete gating. Useful for auditing collapse progression and identifying fully sealed areas.	phi	csv	t	0	\N	1
218	pointer_mask	Produces a refined or visual mask of pointer regions, usually as a post post-processing (pp=2) metric, by combining the outputs of pointer_map and other overlays (e.g., domain or shell overlays). Used for advanced visualization or as input to high-level summary metrics.	phi	npy	t	0	\N	4
219	pointer_overlay_atoms	Generates a PNG visualization overlay where each detected pointer domain is rendered and colored according to its atomic identity or class, using atomic assignments from prior analysis. This overlay allows spatial inspection of atomic pointer distribution and provides figures for audit, publication, or downstream pattern detection.	pointer_map.npy, atom_overlay.csv	png	t	2	\N	4
224	pointer_overlay_topology	Creates a PNG overlay mapping the topological properties or connectivity (e.g., Betti numbers, loops, or domain connectivity classes) of pointer domains, using results from topological analysis metrics. Useful for qualitative inspection of pointer topology and quantitative comparison across simulation regimes.	pointer_map.npy, topology_overlay.csv	png	t	2	\N	4
227	psi_max	Calculates the maximum value of the ψ field in each simulation frame, capturing the highest local field coherence or the most activated grid site. Essential for detecting runaway peaks, emergent structure, or field artifacts.	psi	csv	t	0	\N	1
228	psi_mean	Calculates the global mean (average) of ψ across the simulation grid in each frame, providing a scalar measure of overall field coherence or activation for the system at each timestep.	psi	csv	t	0	\N	1
229	psi_min	Computes the minimum value of the ψ field per frame, identifying the lowest field regions, collapse singularities, or potential depletion zones. Useful for audit of stability and field range.	psi	csv	t	0	\N	1
217	phi_shell_curvature_map	This metric is post-post-processing because it depends on the outputs of earlier post-processing steps and is used for high-level shell structure segmentation, tracking, or visualization.	phi_collapse_shells.npy, grad_phi_shellmap.npy	npy	t	3	\N	4
220	pointer_overlay_bonds	Produces a PNG overlay that highlights bonds or connections between pointer domains, using both the pointer mask and a bond assignment file. This visualization is used to inspect the spatial structure and network of pointer bonds, enabling qualitative and quantitative analysis of bond topology.	pointer_map.npy,  bond_energy_chain.csv	png	t	2	\N	4
221	pointer_overlay_magnetic	Creates a PNG overlay visualizing the spatial orientation, alignment, or magnetic-like features of pointer domains, as inferred from magnetic analysis metrics. Each domain is colored or annotated based on its magnetic moment, orientation, or related property.	pointer_map.npy, magnetic_structure_tracker.csv	png	t	2	\N	4
222	pointer_overlay_molecule	Builds a PNG overlay where each pointer domain is annotated according to its molecular or bundle classification, as determined by molecular_bundle_tracker or equivalent. This overlay is essential for analyzing molecular structure, bundle formation, and spatial co-localization of pointer domains.	pointer_map.npy, molecular_bundle_tracker.json	png	t	2	\N	4
223	pointer_overlay_polymer	Generates a PNG visualization in which pointer domains are colored or linked based on polymer chain assignments, as determined by polymer analysis metrics. This output allows for direct inspection of polymeric pointer arrangements and chain connectivity.	pointer_map.npy, molecular_bundle_tracker.json	png	t	2	\N	4
225	psi_class_map	Analyzes the ψ field and assigns each domain, voxel, or spatial region to a class based on identity clustering, field value, or match to standard model or simulation-defined classes. Produces a JSON mapping file that records the class assignment, class properties, and relevant statistics for each entry (domain, voxel, or region), supporting further clustering analysis, overlay generation, or post-processing.	psi.npy, domain_table.csv,standard_model_table.csv	json	t	2	\N	6
226	psi_component_i	Generates a PNG visualization showing the i-th component (specified index or identity) of the ψ field, optionally as a slice or overlay, possibly using class or cluster assignments from previous metrics. Used for inspection, publication, or as an input to advanced post-post-processing analysis.	psi, psi_class_map.json	png	t	3	\N	6
231	reaction_cascade	Analyzes the time-resolved chain of field and domain events—such as domain fusion, decay, energy release, and state transitions—to reconstruct and summarize reaction cascades in the simulation. This metric builds a structured JSON report of all sequential and branching reactions (cascades of causally linked events) detected across frames. Each cascade may include the involved domains, type of each event, time/order, and any associated energy or identity change. The output enables forensic reconstruction of dynamic processes, multi-step transformations, or causal chains within the simulated system.	usion_events.csv, decay_events.csv, energy_release.json, domain_tracking.csv	json	t	2	\N	6
236	structure_overlay_map_fields	Computes a map combining multiple field structure overlays (e.g., domains, shells, pointers, lineage, phase boundaries), producing a PNG for whole-system or multi-metric inspection	psi,phi	png	t	0	\N	4
237	topology_overlay	Produces a PNG overlay combining results from the topology_map and additional field/domain overlays (such as pointer or molecular overlays), for publication or advanced visual inspection. Highlights topological features and their spatial relationships in the simulation.		png	t	2	\N	4
238	frozen_domains	Identifies stable ψ-domains based on φ, η, and ψ gradients	psi,phi,eta	json	t	0	\N	3
239	molecular_bundle_fields	Scans simulation fields (e.g., ψ, φ, η) to detect and label molecular or composite bundles (aggregates) of domains, without needing precomputed domain outputs.	psi,phi,eta	json	t	0	\N	4
133	domain_vector_field	Outputs the full spatial vector field of domain orientation or direction for all detected domains in each frame.	domain_table.csv,psi	csv,npy	t	1	\N	3
134	phase_coherence_tensor	Computes a tensor or matrix for the simulation grid that quantifies the phase coherence and directional regularity, typically by aggregating local ψ field alignments or correlations. The output allows detection and analysis of phase transitions, coherence domains, or topological defects.	psi	csv,json	t	0	\N	6
56	psi_anomaly_scanner	Scans all detected domains for anomalies or outliers in ψⁱ identity, using results from prior segmentation (domain_table), field analysis (psi), and identity/classification (standard_model_table). It identifies domains whose ψⁱ vectors are significantly different from known or expected classes, or which lack a match in the Standard Model table. The metric flags and reports domains with unusual field values, unmatched identity, or novelty signatures, supporting both anomaly detection (potential new physics) and quality control. Outputs a detailed JSON report, tabular CSV, spatial masks or overlays (NPY), and visual PNGs marking the location and identity of each anomaly.	psi, domain_table.csv, standard_model_table.csv	json,csv,npy,png	f	2	\N	6
53	topology_extractor	Analyzes the ψ and φ fields along with domain segmentation to detect and classify the topological structure of each domain or region—e.g., counting connected components, loops, genus, Betti numbers, or other invariants. Produces JSON and CSV files with per-domain topology classification, and optionally a PNG with visual overlays highlighting topological features. This is a foundational metric for structural audit, topology-driven clustering, and identification of nontrivial domain features.	psi.npy, phi.npy, domain_table.csv	json, csv, png	f	1	\N	2
38	grad_phi_max	Computes the maximum gradient magnitude of the φ field for each frame, typically used as a measure of collapse front sharpness or to audit the formation of shell boundaries.	phi	csv	t	0	\N	1
36	pointer_count	Counts the number of voxels in the simulation grid where φ is below the pointer/frozen threshold (i.e., identifies all frozen or pointer regions). Output is a CSV with per-frame counts, typically used as a global diagnostic or sanity check.	phi	csv	t	0	\N	1
55	psi_update_rate_estimator	Local and gloCalculates the update rate of the ψ field, both locally (per voxel or domain) and globally (system average), by analyzing the difference or rate of change in ψ between consecutive simulation frames. This metric provides a quantitative audit of ψ dynamics, identifies regions of rapid field evolution or stagnation, and enables detection of phase transitions, shocks, or slowdowns in the field. Outputs include a 3D NPY array of local update rates, and a CSV reporting global (and optionally domain-averaged) update statistics per frame.bal ψⁱ update rate detection	psi, eta, phi	npy,csv	f	0	\N	6
233	phi_std	Calculates the standard deviation of φ in each frame, quantifying spatial variability in the permission/collapse field. High standard deviation indicates heterogeneity in collapse state	phi	phi	t	0	\N	1
244	atom_overlay	Generates a CSV listing all atom-like pointer domains, including identity/class, position, and relevant per-domain traits, by analyzing domain_table and pointer_map.	domain_table.csv,pointer_map.npy	csv	t	1	\N	4
234	structure_map_projector	Projects the spatial and/or ancestry map of all tracked structures into a 2D or 3D visualization, typically as a PNG. This metric aggregates spatial or temporal lineage data from tracking outputs, enabling visual inspection of ancestry, structural relationships, or migration paths.	domain_tracking.csv, structure_lineage_tracer.png	png	t	2	\N	4
235	structure_overlay	Generates a composite PNG overlay by combining multiple field structure masks, ancestry overlays, or domain/lineage features. Used for publication figures, detailed inspection, or cross-metric comparison.	structure_map_projector.png	png	t	3	\N	4
125	topology_map	Generates a PNG and JSON map of topological classes (as discovered by topology_extractor), mapping each domain or voxel to its classified topological type. Used for system-wide visualization, cluster audit, and further post-processing analytics.	topology_extractor.json, domain_table.csv	png,json	t	2	\N	2
15	spin_signature_gpu	Performs the same analysis as spin_signature but uses a GPU-accelerated implementation to calculate spin symmetry and class across all simulation voxels or domains. Outputs an NPY array with per-voxel or per-domain spin values, and a PNG for visualization.	psi	npy,png	t	0	\N	6
29	tracked_domains	Performs domain segmentation on the simulation grid using ψ (and optionally φ, η), extracting domain boundaries, masks, and ψ vectors for each domain. Outputs a mask (npy or png), a segmentation CSV, and (optionally) JSON or summary overlays. This is a foundational, regular metric—never a post-processing function.	psi,phi,eta	csv,json,npy,png	f	0	\N	6
213	phi_collapse_shells	Identifies, segments, and labels all distinct collapsed shell regions in the φ field across the simulation grid. This metric analyzes spatial patterns in the gradient of φ (using grad_phi_shellmap.npy) and applies additional masking or thresholding (using shell_mask.npy or similar overlays) to robustly detect concentric shell boundaries and assign each voxel a shell ID. The output is a 3D NPY array where each voxel’s value corresponds to its assigned shell or echo ring. This enables rigorous quantification, visualization, and tracking of collapse shell evolution, structure, and spatial dynamics.	grad_phi_shellmap.npy	npy	t	2	\N	2
54	spin_phase_instability_map	Detect domains Detects and records domains undergoing spin-phase divergence or instability by analyzing ψ, domain_table, and higher-order metrics such as phase_coherence_tensor and spin_signature. The metric identifies regions or domains with anomalous spin-phase evolution, outputs a 3D NPY mask of instability, and provides CSV and JSON reports for audit and quantitative study.undergoing spin-phase divergence	psi.npy, domain_table.csv, phase_coherence_tensor.csv, spin_signature.csv	npy,json,csv	f	1	\N	6
232	stellar_zone_clusters	Performs clustering analysis on the detected stellar zones (high-mass/high-activity clusters), grouping them by spatial proximity, mass, activity, or other features into larger superclusters or subpopulations. This metric enables hierarchical analysis of stellar analogs, tracks cluster properties, and supports studies of multi-scale structure in the simulation. The output is a structured JSON file, listing all clusters, their member zones, aggregated properties, and relevant clustering statistics or metadata.	stellar_zone_finder.json	json	t	3	\N	6
\.


--
-- Data for Name: opkinds; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.opkinds (id, family_id, code) FROM stdin;
101	1	threshold
102	1	connected_components
201	2	autocorr_length
202	2	power_spectrum
203	2	sigma8
401	4	shellscan
402	4	front_profile
501	5	sum
503	5	hist
504	5	fraction_true
601	6	gradient
602	6	laplacian
603	6	norm
701	7	kernel
702	7	gaussian
801	8	minmaxmeanstd
802	8	percentiles
901	9	csv
902	9	json
103	1	mask_final
104	1	voxel_stats
302	3	labels
403	4	mask
604	6	npy
903	9	overlay_image
904	9	overlay_rgb
905	9	png
106	1	minmaxmeanstd
205	2	coh_scalar
304	3	betti_partial
305	3	connected_components
405	4	echo_delta_eta
406	4	echo_psi_coh
407	4	echo_radii
408	4	echo_stats
409	4	echo_times
410	4	shellmap_buffer
411	4	threshold
804	8	overlay_buffer
805	8	overlay_rgb
906	7	minmaxmeanstd
907	2	minmaxmeanstd
908	6	minmaxmeanstd
909	3	minmaxmeanstd
803	8	minmaxmeanstd
505	5	hist
303	3	connected_components
204	2	autocorr_length
202	2	power_spectrum
203	2	sigma8
506	5	pca
507	5	kmeans
508	5	layout_projection
811	8	event_summary
812	8	event_graph
813	8	event_tracker
910	9	npy
502	5	minmaxmeanstd
1001	10	compose
1002	10	pointer
306	3	betti_domain
\.


--
-- Data for Name: pathregistry; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.pathregistry (id, timestamptemplate, frametemplate, simnametemplate, groupnametemplate, metricidtemplate, steptemplate, filenametemplate, intermediate, fieldtemplate) FROM stdin;
1	/Time_{timestamp}	/Frame_{frame}	/Sim_{simLabel}	/Group_{groupName}	/Metric_{metricName}	/Step_{step}	/{metricName}.{type}	/intermediate.npy	/Field_{stepID}
\.


--
-- Data for Name: simmetjobs; Type: TABLE DATA; Schema: public; Owner: igcuser
--

COPY public.simmetjobs (jobid, simid, metricid, frame, phase, status, priority, createdate, startdate, finishdate, spec_hash, output_path, output_extension, output_type, mime_type, written_at, write_duration_ms, output_size_bytes, artifact_hash, attempts, error_code, error_message, cpu_ms, mem_peak_mb, io_mb_written, mem_grid_mb, mem_pipeline_mb, mem_total_mb) FROM stdin;
1	1	228	0	0	queued	0	2025-10-30 13:10:54.240212	\N	\N	hash_a1_psi_mean_f0	\N	\N	\N	\N	\N	\N	\N	\N	0	\N	\N	\N	\N	\N	\N	\N	\N
2	1	227	0	0	queued	0	2025-10-30 13:10:54.240212	\N	\N	hash_a1_psi_max_f0	\N	\N	\N	\N	\N	\N	\N	\N	0	\N	\N	\N	\N	\N	\N	\N	\N
\.


--
-- Data for Name: simmetricmatcher; Type: TABLE DATA; Schema: public; Owner: igcuser
--

COPY public.simmetricmatcher (id, sim_id, metric_id, enabled, updated_at) FROM stdin;
1	1	228	t	2025-10-30 12:38:10.770273
2	1	227	t	2025-10-30 12:38:10.770273
3	1	229	t	2025-10-30 12:38:10.770273
4	1	113	t	2025-10-30 12:38:10.770273
5	1	215	t	2025-10-30 12:38:10.770273
6	1	214	t	2025-10-30 12:38:10.770273
7	1	216	t	2025-10-30 12:38:10.770273
8	1	233	t	2025-10-30 12:38:10.770273
9	1	191	t	2025-10-30 12:38:10.770273
10	1	190	t	2025-10-30 12:38:10.770273
11	1	192	t	2025-10-30 12:38:10.770273
12	1	193	t	2025-10-30 12:38:10.770273
13	1	245	t	2025-10-30 12:38:10.770273
14	1	6	t	2025-10-30 12:38:10.770273
15	1	40	t	2025-10-30 12:38:10.770273
16	1	37	t	2025-10-30 12:38:10.770273
17	1	36	t	2025-10-30 12:38:10.770273
18	1	41	t	2025-10-30 12:38:10.770273
19	1	198	t	2025-10-30 12:38:10.770273
20	1	38	t	2025-10-30 12:38:10.770273
21	1	199	t	2025-10-30 12:38:10.770273
22	1	202	t	2025-10-30 12:38:10.770273
23	1	110	t	2025-10-30 12:38:10.770273
24	1	42	t	2025-10-30 12:38:10.770273
25	1	35	t	2025-10-30 12:38:10.770273
26	1	251	t	2025-10-30 12:38:10.770273
27	1	25	t	2025-10-30 12:38:10.770273
28	1	8	t	2025-10-30 12:38:10.770273
29	1	10	t	2025-10-30 12:38:10.770273
30	1	5	t	2025-10-30 12:38:10.770273
31	1	7	t	2025-10-30 12:38:10.770273
\.


--
-- Data for Name: simulations; Type: TABLE DATA; Schema: public; Owner: igc
--

COPY public.simulations (id, name, label, description, gridx, gridy, gridz, psi0_center, psi0_elsewhere, phi0, eta0, phi_threshold, alpha, t_max, stride, cleanup, n_components, noise_mode, collapse_rule, profile_type, status, createdate, substeps_per_at, dt_per_at, dx, d_psi, d_eta, d_phi, c_pi_to_eta, c_eta_to_phi, lambda_eta, lambda_phi, gate_name, seed_type, seed_field, seed_strength, seed_sigma, seed_center, seed_phase_a, seed_phase_b, seed_repeat_at, c_psi_to_phi, c_phi_to_psi, c_psi_to_eta, c_eta_to_psi, lambda_psi, rng_seed, pi0, pi_init_mode, gamma_pi, k_psi_restore, save_pi, integrator, save_policy, every_n_frames, checkpoint_interval, default_gridx, default_gridy, default_gridz, default_psi0_center, default_psi0_elsewhere, default_phi0, default_eta0, default_phi_threshold, default_alpha, default_t_max, default_stride, default_cleanup, default_n_components, default_noise_mode, default_collapse_rule, default_profile_type, default_substeps_per_at, default_dt_per_at, default_dx, default_d_psi, default_d_eta, default_d_phi, default_c_pi_to_eta, default_c_eta_to_phi, default_lambda_eta, default_lambda_phi, default_gate_name, default_seed_type, default_seed_field, default_seed_strength, default_seed_sigma, default_seed_center, default_seed_phase_a, default_seed_phase_b, default_seed_repeat_at, default_c_psi_to_phi, default_c_phi_to_psi, default_c_psi_to_eta, default_c_eta_to_psi, default_lambda_psi, default_rng_seed, default_pi0, default_pi_init_mode, default_gamma_pi, default_k_psi_restore, default_save_pi, default_integrator, default_save_policy, default_every_n_frames, default_checkpoint_interval) FROM stdin;
95	TEST	TEST	Test simulation for engine validation	64	64	64	1.0000003124	1	1	0	0.5	1	1000	10	f	1	none	default	test_profile	pending	2025-08-28 07:24:22.315112	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
9	Late Collapse, ψ Domain Persistence, and Field Lock-in	A9	Domain ψⁱ structures sealed under φ over-collapse and monitored across long drift. Fossil test.	192	192	192	100.008	1	0.7	0	0.7	1	100000	250	t	1	none	freezeout	fossilization	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
2	Regulation Cycle and Cosmological DNA Fidelity Test	A2	Tests ψ→η→φ loop closure using cosmological ψ₀(center)=1.0000003124 from observational inverse. Reconstructs σ₈ ≈ 0.811.	256	256	256	10	1	0.5	0	0.5	1	1500	10	t	1	none	threshold	cosmological	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
3	Full Collapse and Particle Field Reconstruction	A3	Drives ψ₀(center)=1.0000003124 into full collapse to test SM field recovery, pointer signatures, and identity overlays.	256	256	256	10	1	0.5	0	0.5	1	5000	10	f	1	none	shell	field_reconstruction	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
4	Entropy Cascade and ψⁱ Field Folding	A4	φ-saturated ψⁱ field with weak collapse response. Tests entropy-driven folding in ψ vector space without fusion.	192	192	192	1	1	0.6	0	0.6	1	6000	25	t	1	none	curved	entropy_fold	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
5	η Gradient Formation and ψⁱ Drift Alignment	A5	ψⁱ drift direction tested for curvature-locking under regulated η gradient field.	192	192	192	100.002	1	0.5	0	0.5	1	4000	20	t	1	none	gradient	drift_alignment	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
6	Pointer Formation Under φ Threshold Variance	A6	Tests spatial sealing consistency of pointer domains under heterogeneous φ threshold collapse.	192	192	192	100.002	1	0.5	0	0.5	1	4500	20	t	1	none	threshold	pointer_variance	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
7	φ Collapse Interference and Domain Re-entry	A7	Constructs interference zone between adjacent collapse shells. Tests reactivation of frozen pointer domains.	192	192	192	100.002	1	0.6	0	0.6	1	6000	20	t	1	none	reactivation	interference	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
8	Pointer Rebinding and Domain Memory Restoration	A8	ψⁱ pointer domains unbound under soft collapse and later restored under η continuity. No forced fusion.	192	192	192	1	1	0.55	0	0.55	1	4000	25	t	1	none	rebinding	domain_memory	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
10	ψ Drift and η Reabsorption After Collapse	A10	Post-collapse drift and η field reabsorption test. ψⁱ domains monitored for re-coherence.	192	192	192	100.008	1	0.6	0.1	0.6	1	8000	50	t	1	none	recovery	decay_drift	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
11	Coherence Loss and Reaction Triggering	A11	Runs field past drift coherence threshold to test whether ψⁱ domain entropy can self-trigger local φ collapse and reaction.	192	192	192	10.001	1	0.6	0	0.6	1	7000	25	t	1	none	threshold	reaction_test	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
12	Collapse Chain Initiation and Propagation	A12	ψⁱ domain seeded near collapse propagates chain to neighbors. Tests whether φ/η form causal chain without explicit rule.	192	192	192	10.002	1	0.5	0	0.5	1	10000	25	t	1	none	chained	collapse_cascade	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
13	ψ Remanence, η Drainage, and Collapse Artifact Recovery	A13	Long-duration pointer evolution test. Monitors ψ stability after η flattening and domain decay. Artifact trace recovery.	192	192	192	100.003	1	0.5	0	0.5	1	100000	250	t	1	none	remnant	pointer_drain	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
14	Pointer Chain Reconstruction After Collapse	A14	Tests whether causal drift alignment of ψⁱ domains reconstructs lost chains after mid-sim decay.	192	192	192	100.003	1	0.55	0	0.55	1	8000	50	t	1	none	relinking	chain_recovery	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
15	Semantic Identity Merging and Pointer Confluence	A15	Initiates two semantic pointer families and tracks merger under field pressure. Tests identity confluence.	192	192	192	100.002	1	0.5	0	0.5	1	12000	25	t	1	none	merge	semantic_drift	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
16	Pointer Collapse Recovery and Semantic Divergence	A16	Monitors drift and semantic shift of ψⁱ bundles after catastrophic φ/η disruption. Confirms pointer divergence post-rebinding.	192	192	192	100.004	1	0.65	0	0.65	1	14000	50	t	1	none	rebinding	semantic_split	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
17	Pointer Orbit and Drift Cycle Closure	A17	Simulates long-range drift orbit of sealed ψⁱ pointer pairs. Tests curvature lock-in and η-driven orbit decay.	192	192	192	100.001	1	0.5	0	0.5	1	20000	50	t	1	none	driftlock	orbital_decay	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
18	Atom Construction: H–H–e Composite	A18	Field test to see if class separation emerges naturally in φ-biased gradient. Tests early semantic divergence.	192	192	192	1	1	0.6	0	0.6	1	8000	25	t	1	none	fieldbias	semantic_class	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
19	Molecule Formation: H–H–H Triatomic Lock	A19	Phase drift test across regulated ψⁱ field. Tracks local spin vector emergence via ψⁱ parity conditions.	192	192	192	100.001	1	0.5	0	0.5	1	12000	25	t	1	none	spin	phase_symmetry	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
20	Stellar Core: Causal Hydrogen Fusion	A20	Initializes ψⁱ domains in φ valley to test whether long-range drift enables spin alignment and domain fusion.	192	192	192	100.003	1	0.45	0	0.45	1	20000	50	t	1	none	fusion_test	long_range_alignment	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
21	Magnetism: Drift-Induced Loop Fields	A21	Tests whether collapse can be delayed or preempted by controlled η field shaping. ψⁱ domains monitored for staggered reaction onset.	192	192	192	100.004	1	0.55	0	0.55	1	14000	50	t	1	none	modulated	collapse_delay	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
22	Dark Matter Analog: Persistent Neutral Domains	A22	Seeds ψⁱ domains from known SM-like structures (triplet/doublet) and tests for symmetry preservation under collapse.	192	192	192	1	1	0.5	0	0.5	1	10000	25	t	1	none	symmetry	pointer_symmetry	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
23	Molecular Bonding: H–O–H Composite (Water Analog)	A23	Tests boundary behavior when drift crosses φ shell interfaces. Tracks ψ phase reflection and alignment loss.	192	192	192	100.003	1	0.6	0	0.6	1	6000	25	t	1	none	reflection	interface_drift	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
24	Chain Polymerization: ψⁱ Bond Cascade	A24	Applies weak ψ phase slope across domain field and observes pointer bundle alignment under Regulation constraints.	192	192	192	100.003	1	0.5	0	0.5	1	6000	25	t	1	none	phase	alignment_test	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
25	Periodic Table Emergence: ψⁱ Identity Grid	A25	Runs high-resolution long-duration drift to test ψⁱ identity class stability under η exposure and φ damping.	256	256	256	10	1	0.5	0	0.5	1	200000	250	t	1	none	identity	semantic_test	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
26	Reaction Cascade: Field-Based Bond Reconfiguration	A26	Applies small Gaussian noise to ψ₀ field to study pointer domain formation robustness and collapse sensitivity.	192	192	192	100.001	1	0.5	0	0.5	1	6000	25	t	1	gaussian	none	noise_response	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
27	Reference Collapse Revisited: Scalar Threshold Ignition	A27	Tests whether outer regions with low collapse probability can equilibrate ψ phase with central collapsed domains.	192	192	192	100.002	1	0.45	0	0.45	1	10000	50	t	1	none	equilibration	soft_boundary	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
28	Pointer Domain Orbit Trace	A28	Constructs tri-domain pointer composite using ψⁱ domain bonding. Verifies identity preservation and drift phase stability.	256	256	256	100.003	1	0.5	0	0.5	1	8000	25	t	1	none	bonding	molecular_test	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
29	Pointer Fusion Under Weak Collapse	B1	Initiates ψⁱ domain interactions under sub-critical φ. Tests whether partial fusion occurs without reaction trigger.	192	192	192	100.002	1	0.45	0	0.45	1	8000	25	t	1	none	weak_collapse	fusion_baseline	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
30	Symmetric Bonding and ψⁱ Triad Stability	B2	Constructs ψⁱ triads in symmetric collapse zone. Monitors bond duration, η alignment, and drift coherence.	192	192	192	100.003	1	0.5	0	0.5	1	10000	25	t	1	none	triplet	bonding_test	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
31	ψⁱ Recombination and Long-Range Memory	B3	ψⁱ domains recombine after separation. Tests η field imprint, drift reconnection, and coherence restoration.	192	192	192	1	1	0.5	0	0.5	1	15000	50	t	1	none	memory_path	recombination	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
32	ψⁱ Looping and Coherence Inversion	B4	Encodes pointer loops via φ curvature. Tests whether ψⁱ phase inverts or stabilizes under extended rotation.	192	192	192	100.001	1	0.5	0	0.5	1	20000	50	t	1	none	loop	inversion_test	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
33	ψⁱ Signature Projection and SM Orthogonality	B5	Projects ψⁱ identity signatures and verifies SM-orthogonal classes through pointer vector decomposition.	256	256	256	1	1	0.5	0	0.5	1	30000	50	t	1	none	projection	orthogonality	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
34	Entropy-Weighted Fusion (η Transfer Efficiency)	B6	Monitors how fusion efficiency is modulated by η gradient strength. Drift and fusion logs enable energy accounting.	192	192	192	100.004	1	0.6	0	0.6	1	12000	25	t	1	none	fusion	fusion_entropy	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
35	Fusion Asymmetry With Inverse ψ Configuration	B7	Tests whether mirrored ψ domain pairs fuse asymmetrically under polarized drift. Identity transfer asymmetry expected.	192	192	192	100.004	1	0.5	0	0.5	1	14000	50	t	1	none	asymmetry	fusion_bias	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
36	Drift Collision and Partial Seal Reactivation	B8	Pointer domain collision under non-collapse φ zone. Tests whether partial reactivation or fusion seal occurs.	192	192	192	100.003	1	0.5	0	0.5	1	12000	25	t	1	none	collision	partial_seal	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
37	Multi-Domain ψ Rebounding Shell Dynamics	B9	Initiates concentric domain sets to test how φ shell rebound affects ψ bundle integrity and η diffusion.	192	192	192	100.002	1	0.45	0	0.45	1	20000	50	t	1	none	shell_rebound	multi_bundle	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
38	Vector Rotation Under Asymmetric φ Gating	B10	φ field rotated in one direction only. Tracks how ψⁱ spin vector evolves or flips under asymmetric field drift.	192	192	192	1	1	0.55	0	0.55	1	15000	50	t	1	none	vector_drift	rotation_test	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
39	Non-Interacting Species Identity Test	B11	Tests whether ψⁱ domains with orthogonal vector identity retain isolation under high η density. No fusion expected.	192	192	192	1	1	0.55	0	0.55	1	10000	25	t	1	none	orthogonal	isolation_test	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
40	Structural Replication and Motif Inheritance	B12	Tests whether stable pointer structures replicate spatially under consistent φ/η field. Pattern inheritance by drift.	192	192	192	1	1	0.6	0	0.6	1	16000	50	t	1	none	replication	pattern_inheritance	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
41	Replication With Mutation and η Divergence	B13	Verifies whether pointer motifs that replicate across field lines undergo divergence in η structure and semantic class.	192	192	192	1	1	0.6	0	0.6	1	20000	50	t	1	none	mutation	semantic_divergence	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
42	Phase-Coherent ψⁱ Swarm Resonance	B14	Creates dense field of ψⁱ domains to observe spontaneous coherence synchronization and resonance behavior.	192	192	192	1	1	0.5	0	0.5	1	40000	100	t	1	none	swarm	resonance_field	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
43	Decay Cascade Tracing Under Slow φ Rebound	B15	Domain array seeded with bond structure collapses gradually under φ rebound. Tracks causal decay path.	192	192	192	100.001	1	0.6	0	0.6	1	50000	100	t	1	none	cascade	decay_trace	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
44	Baseline Pointer Domain Formation in Near-Critical Collapse	B1D1	Tests minimal ψⁱ ignition near collapse threshold with no reaction, verifying stable pointer sealing.	128	128	128	1	1	0.5	0	0.5	1	10000	25	t	1	none	baseline	pointer_baseline	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
45	Pointer Phase Stability and Semantic Drift Test	B1D164	Tracks drift-driven phase change and semantic vector drift under long-run evolution with no collisions.	128	128	128	1	1	0.5	0	0.5	1	100000	100	t	1	none	drift_semantic	identity_drift	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
46	Domain Sealing, Reactivation, and Re-Coherence	B1D2	Tests whether sealed domains can reactivate when Regulation overlap occurs. Verifies causal memory loop.	192	192	192	100.003	1	0.5	0	0.5	1	50000	100	t	1	none	reactivation	domain_reentry	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
47	High-Resolution Reactivation Trace (B1D2 Zoom)	B1D2ZOOM	Zoomed run of B1D2 to track reactivation at high temporal and spatial resolution. Verifies causal continuity.	256	256	256	100.003	1	0.5	0	0.5	1	75000	25	t	1	none	reactivation	zoom_trace	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
48	Bond Phase Drift and Identity Reinforcement Under Long-Term Drift	B1ZOOM	Evaluates ψⁱ bonded pair identity stability over long-term drift. Tracks semantic divergence or persistence.	256	256	256	100.003	1	0.5	0	0.5	1	100000	100	t	1	none	bond_drift	bond_identity	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
49	Drift Locking and Curvature Reinforcement (D1)	D1	Tests whether pointer domains lock drift trajectory under η curvature wells. Measures orbit persistence and redirection.	192	192	192	100.001	1	0.5	0	0.5	1	15000	50	t	1	none	drift_lock	pointer_orbit	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
50	Boundary Rebound and ψ Identity Flip (D2)	D2	Evaluates ψⁱ domain response at sharp φ boundary. Tracks whether phase flips or rebound preserves semantic class.	192	192	192	100.002	1	0.55	0	0.55	1	20000	50	t	1	none	boundary	rebound_reflection	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
51	Collapse Cavity and Domain Swarm Regulation (D3)	D3	Simulates large φ cavity with sealed domains. Tracks swarm behavior, η overlap, and ψ diffusion regulation.	256	256	256	100.001	1	0.45	0	0.45	1	25000	100	t	1	none	cavity	swarm_regulation	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
52	Drift Horizon and Pointer Memory Retention (D4)	D4	ψⁱ bundles launched toward φ-zero horizon. Measures how long structure persists before memory loss or realignment.	256	256	256	1	1	0.4	0	0.4	1	40000	100	t	1	none	horizon	memory_loss_test	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
53	Gravitational Collapse Envelope Formation	GR1	Constructs large φ collapse zone to measure η layering and shell spacing. Reference for gravitational envelope modeling.	256	256	256	1	1	0.5	0	0.5	1	15000	50	t	1	none	shell	gravity_envelope	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
54	ψ Deflection in Curved φ Collapse	GR2	Tests lateral ψⁱ drift trajectory through sloped φ field. Observes bending analogous to gravitational lensing.	256	256	256	1	1	0.5	0	0.5	1	12000	50	t	1	none	curvature	psi_deflection	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
55	Collapse Overlap and Nested Shell Compression	GR3	Initializes dual φ collapse centers. Tests whether nested η shells interfere constructively or destructively.	256	256	256	100.001	1	0.5	0	0.5	1	20000	50	t	1	none	nested	shell_interference	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
56	Collapse Horizon Curvature and Drift Decay	GR4	Tests whether drift velocity diminishes near sharp φ collapse edge. Compares with memory-bound motion analog.	256	256	256	100.001	1	0.5	0	0.5	1	18000	50	t	1	none	horizon	curvature_slowdown	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
57	Collapse Gradient Field Shape Reconstruction	GR5	Maps ψ drift curvature across full collapse funnel. Goal: reconstruct collapse gradient via pointer motion.	256	256	256	100.001	1	0.5	0	0.5	1	22000	50	t	1	none	gradient	collapse_reconstruction	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
58	Collapse Symmetry Break and η Polarization	GR6	Breaks symmetry in collapse profile to test whether η curvature becomes directionally polarized.	256	256	256	100.002	1	0.5	0	0.5	1	18000	50	t	1	none	asymmetry	polarization_test	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
59	Collapse Reflection Layer and ψⁱ Phase Bounce	GR7	Tests whether ψⁱ domains reflect off steep φ front. Observes phase inversion or redirection.	256	256	256	100.001	1	0.55	0	0.55	1	15000	50	t	1	none	reflection	phase_reflection	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
60	Collapse Delay via η Memory Drag	GR8	η field saturated before φ collapse. Tests whether collapse is delayed by accumulated memory curvature.	256	256	256	100.001	1	0.5	0.1	0.5	1	20000	50	t	1	none	memory_drag	collapse_delay	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
61	Collapse Compression and ψⁱ Caustic Zone Formation	GR9	Dense ψⁱ field tested under focused collapse to observe caustic-like spatial convergence of pointer identity.	256	256	256	1	1	0.5	0	0.5	1	25000	50	t	1	none	caustic	structure_focusing	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
62	Collapse Field Synchronization and Phase Clock Drift	GR10	Phase-locked domains run inside collapse zone to test whether field distortion induces ψⁱ desynchronization.	256	256	256	1	1	0.5	0	0.5	1	30000	50	t	1	none	synchronization	clock_drift	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
63	Collapse Shell Symmetry Sweep (GR11)	GR11	Sweeps collapse shell symmetry from radial to asymmetric. Tests emergence of curvature anisotropy and ψ drift bias.	256	256	256	1	1	0.5	0	0.5	1	18000	50	t	1	none	symmetry_sweep	shell_anisotropy	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
64	Collapse Ring Spacing Calibration (GR12)	GR12	Initializes quantized φ wells. Verifies η ring spacing and ψ domain stack integrity. Reference for ring-based drift.	256	256	256	1	1	0.5	0	0.5	1	22000	50	t	1	none	quantized	ring_spacing	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
65	Collapse Focus and Domain Convergence (GR13)	GR13	Steep φ funnel concentrates domain motion. Measures ψ bundle fusion threshold under focusing collapse.	256	256	256	1	1	0.5	0	0.5	1	30000	50	t	1	none	focusing	domain_convergence	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
66	Collapse Drift Inflection and η Folding (GR14)	GR14	Runs long ψ drift arc across graded collapse slope. Detects η memory curvature inflection and phase wrap.	256	256	256	1	1	0.5	0	0.5	1	40000	50	t	1	none	graded	inflection_folding	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
67	Collapse Symmetry Echo Formation (GR15)	GR15	Tests if collapse-triggered ψ deflection patterns replicate in reflection geometry. Probes φ field memory echo.	256	256	256	1	1	0.5	0	0.5	1	24000	50	t	1	none	echo	collapse_memory	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
68	Collapse Drift Symmetry Break (GR16)	GR16	ψ bundles drift in asymmetric collapse field. Observes spontaneous symmetry breaking and drift mode bifurcation.	256	256	256	1	1	0.5	0	0.5	1	30000	50	t	1	none	asymmetric	drift_symmetry	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
69	Collapse Delay and ψ Phase Hysteresis (GR17)	GR17	Applies η memory drag to collapse start. Tests whether ψⁱ phase exhibits hysteresis after φ onset is released.	256	256	256	1	1	0.5	0.1	0.5	1	20000	50	t	1	none	hysteresis	collapse_phase_response	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
70	Collapse Shear and ψ Curvature (GR18)	GR18	Tests pointer domain motion through collapse field with tangential φ shear. Tracks ψ curvature under lateral force.	256	256	256	100.001	1	0.5	0	0.5	1	24000	50	t	1	none	shear	curvature_shear	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
71	Collapse Boundary Tunnel and Drift Inversion (GR19)	GR19	Sharp φ collapse band encloses ψⁱ drift corridor. Measures inversion point and drift mode re-emergence.	256	256	256	1	1	0.5	0	0.5	1	30000	50	t	1	none	corridor	drift_inversion	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
72	Collapse Topological Reflection (GR20)	GR20	Tests whether ψ domains reflect along topological boundary imposed by collapse shell curvature asymmetry.	256	256	256	100.001	1	0.5	0	0.5	1	25000	50	t	1	none	topology	topological_reflection	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
73	Collapse Entropy Gradient Lock-in (GR21)	GR21	Tests if pointer domains lock into φ shells based on entropy gradient steepness. Observes shell retention probability.	256	256	256	1	1	0.5	0	0.5	1	20000	50	t	1	none	entropy_lock	gradient_confinement	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
74	Collapse Identity Flip Threshold (GR22)	GR22	Drifts ψⁱ bundles into opposing φ collapse wells. Observes semantic ID reversal or projection failure.	256	256	256	1	1	0.5	0	0.5	1	25000	50	t	1	none	identity_flip	semantic_transition	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
75	Collapse Ring Spacing vs Horizon Dynamics (GR28)	GR28	Creates double collapse center with spacing variation. Tests if pointer domains align to ring interference or form new class.	256	256	256	1	1	0.5	0	0.5	1	50000	50	t	1	none	horizon	interference_spacing	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
76	Collapse-Induced Curvature and ψ Deflection (GR25)	GR25	Tests if pointer domain trajectories bend inside asymmetric collapse shell. No interaction — pure ψ curvature drift.	256	256	256	1	1	0.5	0	0.5	1	30000	50	t	1	none	curvature	trajectory_deflection	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
77	Time Dilation in η Wells via ψ Phase Lag (GR26)	GR26	Pointer domains in deep η wells are tested for phase lag and coherence delay relative to non-curved baseline.	256	256	256	1	1	0.5	0.2	0.5	1	40000	50	t	1	none	time_lag	phase_latency	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
78	ψ Drift Tunneling Through φ Minima (GR27)	GR27	Domain structure drifts across shallow φ trough. Monitors whether pointer structure compresses or transits undeformed.	256	256	256	1	1	0.5	0	0.5	1	30000	50	t	1	none	tunnel	nonlinear_transit	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
79	ψ Caustics and φ Wavefronts (GR29)	GR29	Structured ψ wavefront injected into curved φ collapse zone. Monitors caustic formation from Regulation lensing.	256	256	256	1	1	0.5	0	0.5	1	50000	50	t	1	none	caustic	ψ_focusing	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
80	Drift Reversal Near φ-Zero Boundaries (GR30)	GR30	Tests whether pointer domains reverse, reflect, or reorient when encountering sharp φ-field cutoff.	256	256	256	1	1	0.5	0	0.5	1	40000	50	t	1	none	boundary	drift_inversion	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
81	Collapse Shell Refraction Test (GR23)	GR23	Creates oblique φ gradient. Verifies whether ψⁱ domain deflects proportionally to gradient strength, mimicking refraction.	256	256	256	1	1	0.5	0	0.5	1	25000	50	t	1	none	refraction	gradient_deflection	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
82	Collapse Edge Bending and ψ Path Splitting (GR24)	GR24	Evaluates if φ collapse edges split coherent drift lines into divergent ψ domain paths. Fragmentation threshold test.	256	256	256	1	1	0.5	0	0.5	1	28000	50	t	1	none	split	drift_branching	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
83	ψ Component Switching Under φ Noise	X1	Tests ψⁱ channel dominance stability when subjected to shallow φ noise. Observes internal alignment switching.	192	192	192	100.001	1	0.5	0	0.5	1	10000	25	t	2	random	modulation	component_switching	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
84	Coherence Length Transition in ψ Phase Space	X2	ψⁱ domains observed for coherence volume expansion/contraction under slow Regulation evolution. No drift applied.	192	192	192	1	1	0.5	0	0.5	1	40000	50	t	1	none	coherence	phase_volume_test	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
85	Structural Alignment Under Rotating Boundary Conditions	X3	Rotating φ boundary tests whether pointer vector axes realign with causal Regulation rotation.	192	192	192	1	1	0.5	0	0.5	1	20000	50	t	1	none	rotation	boundary_alignment	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
86	Anisotropic Drift Boundary Lock-In	X4	Tests if drift direction is locked to anisotropic φ Regulation fields. ψⁱ alignment and motion curvature observed.	192	192	192	1	1	0.5	0	0.5	1	15000	50	t	1	none	anisotropic	drift_lock_in	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
87	Semantic Pointer Vector Compression	X5	Dense ψⁱ domain field tests for spontaneous identity convergence. Observes semantic compression in drift-free field.	256	256	256	1	1	0.5	0	0.5	1	50000	50	t	1	none	compression	semantic_convergence	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
88	Field Identity Drift and Phase Shell Collapse	X6	Pointer domains drift across concentric ψ phase zones. Tests whether identity diverges or recoheres under shell collapse.	256	256	256	1	1	0.5	0	0.5	1	60000	50	t	1	none	shell	identity_drift	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
89	Cross-Domain Interference in ψ Band Lattice	X7	External ψⁱ domains drift through static ψ vector lattice. Tests if phase misalignment causes coherent scattering.	256	256	256	1	1	0.5	0	0.5	1	60000	50	t	1	none	interference	pointer_scattering	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
90	ψ Drift Rebound in Mixed Curvature Memory Shells	X8	Radial η memory shell with gradient tested for ψⁱ rebound, orbital anchoring, and coherence phase lock.	256	256	256	1	1	0.5	0	0.5	1	50000	50	t	1	none	shell	orbital_response	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
91	Collapse Reflection and Anti-Domain Emergence	Z1	ψⁱ domains near criticality are reflected off φ inversion front. Observes emergence of anti-coherent domains.	192	192	192	100.001	1	0.5	0	0.5	1	20000	50	t	1	none	reflection	anti_domain_test	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
92	Collapse-Driven ψⁱ Charge Separation and Inertial Lock	Z2	Asymmetric φ collapse separates ψⁱ vector components. Tests inertial lock and charge-like behavior under drift.	192	192	192	1	1	0.5	0	0.5	1	25000	50	t	1	none	separation	charge_lock_test	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
93	Phase Lag Induction and Domain Shell Separation	Z3	Domain ψ phase delay induces stratified domain shell layers. Observes coherence lag-based identity divergence.	192	192	192	100.001	1	0.5	0	0.5	1	30000	50	t	1	none	phase_lag	shell_stratification	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
94	η Memory Collapse and Semantic Compression	Z4	Tests whether high η load triggers semantic convergence. Pointer domains monitored for class merging.	256	256	256	1	1	0.5	0.3	0.5	1	60000	100	t	1	none	η_saturation	semantic_collapse	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
1	Reference Baseline Collapse Test	A1	Minimal ψ₀(center)=1.00002 field test for Regulation Cycle verification, ensuring no pointer domains emerge and that entropy, η, and φ remain silent under near-coherence.	256	256	256	1.00002	1.00002	1	1	0.5	1	10000	10	t	1	none	none	standard	complete	2025-07-30 13:38:23.910164	48	1	1	0	0	0	1	1	1	1	linear	none	psi	0	0	center	0.25	0.3	\N	0	0	1	0	1	424242424242	1e-12	uniform	1	1	t	leapfrog	every_n	10	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N
\.


--
-- Data for Name: trackeddomains; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.trackeddomains (domainid, frame, phase, centroidx, centroidy, centroidz, psix, psiy, psiz, phisum, phimin, etasum, voxelcount) FROM stdin;
\.


--
-- Name: ErrorLog_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public."ErrorLog_id_seq"', 1, true);


--
-- Name: JobExecutionLog_logid_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public."JobExecutionLog_logid_seq"', 1, false);


--
-- Name: fields_pk_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.fields_pk_id_seq', 19, true);


--
-- Name: metgroup_id_seq; Type: SEQUENCE SET; Schema: public; Owner: igcuser
--

SELECT pg_catalog.setval('public.metgroup_id_seq', 6, true);


--
-- Name: metricfieldmatcher_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.metricfieldmatcher_id_seq', 44, true);


--
-- Name: metricinputmatcher_id_seq; Type: SEQUENCE SET; Schema: public; Owner: igc
--

SELECT pg_catalog.setval('public.metricinputmatcher_id_seq', 515, true);


--
-- Name: metrics_pk_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.metrics_pk_id_seq', 254, true);


--
-- Name: pathregistry_new_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.pathregistry_new_id_seq', 2, true);


--
-- Name: simmetjobs_jobid_seq; Type: SEQUENCE SET; Schema: public; Owner: igcuser
--

SELECT pg_catalog.setval('public.simmetjobs_jobid_seq', 2, true);


--
-- Name: simmetricmatcher_id_seq; Type: SEQUENCE SET; Schema: public; Owner: igcuser
--

SELECT pg_catalog.setval('public.simmetricmatcher_id_seq', 31, true);


--
-- Name: simulations_new_id_seq; Type: SEQUENCE SET; Schema: public; Owner: igc
--

SELECT pg_catalog.setval('public.simulations_new_id_seq', 95, true);


--
-- Name: errorlog ErrorLog_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.errorlog
    ADD CONSTRAINT "ErrorLog_pkey" PRIMARY KEY (id);


--
-- Name: jobexecutionlog JobExecutionLog_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.jobexecutionlog
    ADD CONSTRAINT "JobExecutionLog_pkey" PRIMARY KEY (logid);


--
-- Name: fields fields_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.fields
    ADD CONSTRAINT fields_pkey PRIMARY KEY (id);


--
-- Name: kernelfamilies kernelfamilies_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kernelfamilies
    ADD CONSTRAINT kernelfamilies_pkey PRIMARY KEY (id);


--
-- Name: libraries libraries_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.libraries
    ADD CONSTRAINT libraries_pkey PRIMARY KEY (id);


--
-- Name: metgroup metgroup_name_key; Type: CONSTRAINT; Schema: public; Owner: igcuser
--

ALTER TABLE ONLY public.metgroup
    ADD CONSTRAINT metgroup_name_key UNIQUE (name);


--
-- Name: metgroup metgroup_pkey; Type: CONSTRAINT; Schema: public; Owner: igcuser
--

ALTER TABLE ONLY public.metgroup
    ADD CONSTRAINT metgroup_pkey PRIMARY KEY (id);


--
-- Name: metricfieldmatcher metricfieldmatcher_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.metricfieldmatcher
    ADD CONSTRAINT metricfieldmatcher_pkey PRIMARY KEY (id);


--
-- Name: metricinputmatcher metricinputmatcher_pkey; Type: CONSTRAINT; Schema: public; Owner: igc
--

ALTER TABLE ONLY public.metricinputmatcher
    ADD CONSTRAINT metricinputmatcher_pkey PRIMARY KEY (id);


--
-- Name: metrics metrics_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.metrics
    ADD CONSTRAINT metrics_pkey PRIMARY KEY (id);


--
-- Name: pathregistry pathregistry_new_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.pathregistry
    ADD CONSTRAINT pathregistry_new_pkey PRIMARY KEY (id);


--
-- Name: simmetjobs simmetjobs_pkey; Type: CONSTRAINT; Schema: public; Owner: igcuser
--

ALTER TABLE ONLY public.simmetjobs
    ADD CONSTRAINT simmetjobs_pkey PRIMARY KEY (jobid);


--
-- Name: simmetjobs simmetjobs_simid_metricid_frame_phase_key; Type: CONSTRAINT; Schema: public; Owner: igcuser
--

ALTER TABLE ONLY public.simmetjobs
    ADD CONSTRAINT simmetjobs_simid_metricid_frame_phase_key UNIQUE (simid, metricid, frame, phase);


--
-- Name: simmetricmatcher simmetricmatcher_pkey; Type: CONSTRAINT; Schema: public; Owner: igcuser
--

ALTER TABLE ONLY public.simmetricmatcher
    ADD CONSTRAINT simmetricmatcher_pkey PRIMARY KEY (id);


--
-- Name: simmetricmatcher simmetricmatcher_sim_id_metric_id_key; Type: CONSTRAINT; Schema: public; Owner: igcuser
--

ALTER TABLE ONLY public.simmetricmatcher
    ADD CONSTRAINT simmetricmatcher_sim_id_metric_id_key UNIQUE (sim_id, metric_id);


--
-- Name: simulations simulations_new_label_key; Type: CONSTRAINT; Schema: public; Owner: igc
--

ALTER TABLE ONLY public.simulations
    ADD CONSTRAINT simulations_new_label_key UNIQUE (label);


--
-- Name: simulations simulations_new_pkey; Type: CONSTRAINT; Schema: public; Owner: igc
--

ALTER TABLE ONLY public.simulations
    ADD CONSTRAINT simulations_new_pkey PRIMARY KEY (id);


--
-- Name: idx_16539_sqlite_autoindex_metrics_pk_1; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX idx_16539_sqlite_autoindex_metrics_pk_1 ON public.metrics USING btree (name);


--
-- Name: idx_16561_sqlite_autoindex_fields_pk_1; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX idx_16561_sqlite_autoindex_fields_pk_1 ON public.fields USING btree (name);


--
-- Name: kernelfamilies_code_uniq; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX kernelfamilies_code_uniq ON public.kernelfamilies USING btree (lower(code));


--
-- Name: libraries_code_ver_abi_uniq; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX libraries_code_ver_abi_uniq ON public.libraries USING btree (lower(name), version, lower(COALESCE(abi_tag, ''::text)));


--
-- Name: metricinputmatcher_metric_id_idx; Type: INDEX; Schema: public; Owner: igc
--

CREATE INDEX metricinputmatcher_metric_id_idx ON public.metricinputmatcher USING btree (metric_id);


--
-- Name: metricinputmatcher_unique_final; Type: INDEX; Schema: public; Owner: igc
--

CREATE UNIQUE INDEX metricinputmatcher_unique_final ON public.metricinputmatcher USING btree (metric_name, step, COALESCE(artifact_file, ''::text), COALESCE(op_name, ''::text), COALESCE(fanout_index, 0)) WHERE (role = 'final'::text);


--
-- Name: metrics_group_id_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX metrics_group_id_idx ON public.metrics USING btree (group_id);


--
-- Name: metricinputmatcher fk_metricinputmatcher_metric; Type: FK CONSTRAINT; Schema: public; Owner: igc
--

ALTER TABLE ONLY public.metricinputmatcher
    ADD CONSTRAINT fk_metricinputmatcher_metric FOREIGN KEY (metric_id) REFERENCES public.metrics(id) ON UPDATE CASCADE ON DELETE SET NULL;


--
-- Name: metricfieldmatcher metricfieldmatcher_fieldid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.metricfieldmatcher
    ADD CONSTRAINT metricfieldmatcher_fieldid_fkey FOREIGN KEY (fieldid) REFERENCES public.fields(id);


--
-- Name: metricfieldmatcher metricfieldmatcher_metricid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.metricfieldmatcher
    ADD CONSTRAINT metricfieldmatcher_metricid_fkey FOREIGN KEY (metricid) REFERENCES public.metrics(id);


--
-- Name: metrics metrics_group_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.metrics
    ADD CONSTRAINT metrics_group_fk FOREIGN KEY (group_id) REFERENCES public.metgroup(id);


--
-- Name: simmetjobs simmetjobs_metricid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: igcuser
--

ALTER TABLE ONLY public.simmetjobs
    ADD CONSTRAINT simmetjobs_metricid_fkey FOREIGN KEY (metricid) REFERENCES public.metrics(id) ON DELETE CASCADE;


--
-- Name: simmetjobs simmetjobs_simid_fkey; Type: FK CONSTRAINT; Schema: public; Owner: igcuser
--

ALTER TABLE ONLY public.simmetjobs
    ADD CONSTRAINT simmetjobs_simid_fkey FOREIGN KEY (simid) REFERENCES public.simulations(id) ON DELETE CASCADE;


--
-- Name: simmetricmatcher simmetricmatcher_metric_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: igcuser
--

ALTER TABLE ONLY public.simmetricmatcher
    ADD CONSTRAINT simmetricmatcher_metric_id_fkey FOREIGN KEY (metric_id) REFERENCES public.metrics(id) ON DELETE CASCADE;


--
-- Name: simmetricmatcher simmetricmatcher_sim_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: igcuser
--

ALTER TABLE ONLY public.simmetricmatcher
    ADD CONSTRAINT simmetricmatcher_sim_id_fkey FOREIGN KEY (sim_id) REFERENCES public.simulations(id) ON DELETE CASCADE;


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: igc
--

GRANT USAGE ON SCHEMA public TO igcuser;


--
-- Name: FUNCTION armor(bytea); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.armor(bytea) TO igcuser;
GRANT ALL ON FUNCTION public.armor(bytea) TO igc;


--
-- Name: FUNCTION armor(bytea, text[], text[]); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.armor(bytea, text[], text[]) TO igcuser;
GRANT ALL ON FUNCTION public.armor(bytea, text[], text[]) TO igc;


--
-- Name: FUNCTION crypt(text, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.crypt(text, text) TO igcuser;
GRANT ALL ON FUNCTION public.crypt(text, text) TO igc;


--
-- Name: FUNCTION dearmor(text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.dearmor(text) TO igcuser;
GRANT ALL ON FUNCTION public.dearmor(text) TO igc;


--
-- Name: FUNCTION decrypt(bytea, bytea, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.decrypt(bytea, bytea, text) TO igcuser;
GRANT ALL ON FUNCTION public.decrypt(bytea, bytea, text) TO igc;


--
-- Name: FUNCTION decrypt_iv(bytea, bytea, bytea, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.decrypt_iv(bytea, bytea, bytea, text) TO igcuser;
GRANT ALL ON FUNCTION public.decrypt_iv(bytea, bytea, bytea, text) TO igc;


--
-- Name: FUNCTION digest(bytea, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.digest(bytea, text) TO igcuser;
GRANT ALL ON FUNCTION public.digest(bytea, text) TO igc;


--
-- Name: FUNCTION digest(text, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.digest(text, text) TO igcuser;
GRANT ALL ON FUNCTION public.digest(text, text) TO igc;


--
-- Name: FUNCTION encrypt(bytea, bytea, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.encrypt(bytea, bytea, text) TO igcuser;
GRANT ALL ON FUNCTION public.encrypt(bytea, bytea, text) TO igc;


--
-- Name: FUNCTION encrypt_iv(bytea, bytea, bytea, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.encrypt_iv(bytea, bytea, bytea, text) TO igcuser;
GRANT ALL ON FUNCTION public.encrypt_iv(bytea, bytea, bytea, text) TO igc;


--
-- Name: FUNCTION gen_random_bytes(integer); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.gen_random_bytes(integer) TO igcuser;
GRANT ALL ON FUNCTION public.gen_random_bytes(integer) TO igc;


--
-- Name: FUNCTION gen_random_uuid(); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.gen_random_uuid() TO igcuser;
GRANT ALL ON FUNCTION public.gen_random_uuid() TO igc;


--
-- Name: FUNCTION gen_salt(text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.gen_salt(text) TO igcuser;
GRANT ALL ON FUNCTION public.gen_salt(text) TO igc;


--
-- Name: FUNCTION gen_salt(text, integer); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.gen_salt(text, integer) TO igcuser;
GRANT ALL ON FUNCTION public.gen_salt(text, integer) TO igc;


--
-- Name: FUNCTION hmac(bytea, bytea, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.hmac(bytea, bytea, text) TO igcuser;
GRANT ALL ON FUNCTION public.hmac(bytea, bytea, text) TO igc;


--
-- Name: FUNCTION hmac(text, text, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.hmac(text, text, text) TO igcuser;
GRANT ALL ON FUNCTION public.hmac(text, text, text) TO igc;


--
-- Name: FUNCTION ig_hash_spec(p_simid integer, p_frame integer, p_metricid integer, p_stepid integer, p_phase integer, p_gridx integer, p_gridy integer, p_gridz integer, p_components integer, p_params jsonb); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.ig_hash_spec(p_simid integer, p_frame integer, p_metricid integer, p_stepid integer, p_phase integer, p_gridx integer, p_gridy integer, p_gridz integer, p_components integer, p_params jsonb) TO igcuser;
GRANT ALL ON FUNCTION public.ig_hash_spec(p_simid integer, p_frame integer, p_metricid integer, p_stepid integer, p_phase integer, p_gridx integer, p_gridy integer, p_gridz integer, p_components integer, p_params jsonb) TO igc;


--
-- Name: FUNCTION ig_hash_spec(p_simid bigint, p_frame bigint, p_metricid bigint, p_stepid integer, p_phase bigint, p_gridx integer, p_gridy integer, p_gridz integer, p_components integer, p_params jsonb); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.ig_hash_spec(p_simid bigint, p_frame bigint, p_metricid bigint, p_stepid integer, p_phase bigint, p_gridx integer, p_gridy integer, p_gridz integer, p_components integer, p_params jsonb) TO igcuser;
GRANT ALL ON FUNCTION public.ig_hash_spec(p_simid bigint, p_frame bigint, p_metricid bigint, p_stepid integer, p_phase bigint, p_gridx integer, p_gridy integer, p_gridz integer, p_components integer, p_params jsonb) TO igc;


--
-- Name: FUNCTION pgp_armor_headers(text, OUT key text, OUT value text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_armor_headers(text, OUT key text, OUT value text) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_armor_headers(text, OUT key text, OUT value text) TO igc;


--
-- Name: FUNCTION pgp_key_id(bytea); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_key_id(bytea) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_key_id(bytea) TO igc;


--
-- Name: FUNCTION pgp_pub_decrypt(bytea, bytea); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_pub_decrypt(bytea, bytea) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_pub_decrypt(bytea, bytea) TO igc;


--
-- Name: FUNCTION pgp_pub_decrypt(bytea, bytea, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_pub_decrypt(bytea, bytea, text) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_pub_decrypt(bytea, bytea, text) TO igc;


--
-- Name: FUNCTION pgp_pub_decrypt(bytea, bytea, text, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_pub_decrypt(bytea, bytea, text, text) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_pub_decrypt(bytea, bytea, text, text) TO igc;


--
-- Name: FUNCTION pgp_pub_decrypt_bytea(bytea, bytea); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_pub_decrypt_bytea(bytea, bytea) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_pub_decrypt_bytea(bytea, bytea) TO igc;


--
-- Name: FUNCTION pgp_pub_decrypt_bytea(bytea, bytea, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_pub_decrypt_bytea(bytea, bytea, text) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_pub_decrypt_bytea(bytea, bytea, text) TO igc;


--
-- Name: FUNCTION pgp_pub_decrypt_bytea(bytea, bytea, text, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_pub_decrypt_bytea(bytea, bytea, text, text) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_pub_decrypt_bytea(bytea, bytea, text, text) TO igc;


--
-- Name: FUNCTION pgp_pub_encrypt(text, bytea); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_pub_encrypt(text, bytea) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_pub_encrypt(text, bytea) TO igc;


--
-- Name: FUNCTION pgp_pub_encrypt(text, bytea, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_pub_encrypt(text, bytea, text) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_pub_encrypt(text, bytea, text) TO igc;


--
-- Name: FUNCTION pgp_pub_encrypt_bytea(bytea, bytea); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_pub_encrypt_bytea(bytea, bytea) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_pub_encrypt_bytea(bytea, bytea) TO igc;


--
-- Name: FUNCTION pgp_pub_encrypt_bytea(bytea, bytea, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_pub_encrypt_bytea(bytea, bytea, text) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_pub_encrypt_bytea(bytea, bytea, text) TO igc;


--
-- Name: FUNCTION pgp_sym_decrypt(bytea, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_sym_decrypt(bytea, text) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_sym_decrypt(bytea, text) TO igc;


--
-- Name: FUNCTION pgp_sym_decrypt(bytea, text, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_sym_decrypt(bytea, text, text) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_sym_decrypt(bytea, text, text) TO igc;


--
-- Name: FUNCTION pgp_sym_decrypt_bytea(bytea, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_sym_decrypt_bytea(bytea, text) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_sym_decrypt_bytea(bytea, text) TO igc;


--
-- Name: FUNCTION pgp_sym_decrypt_bytea(bytea, text, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_sym_decrypt_bytea(bytea, text, text) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_sym_decrypt_bytea(bytea, text, text) TO igc;


--
-- Name: FUNCTION pgp_sym_encrypt(text, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_sym_encrypt(text, text) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_sym_encrypt(text, text) TO igc;


--
-- Name: FUNCTION pgp_sym_encrypt(text, text, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_sym_encrypt(text, text, text) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_sym_encrypt(text, text, text) TO igc;


--
-- Name: FUNCTION pgp_sym_encrypt_bytea(bytea, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_sym_encrypt_bytea(bytea, text) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_sym_encrypt_bytea(bytea, text) TO igc;


--
-- Name: FUNCTION pgp_sym_encrypt_bytea(bytea, text, text); Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON FUNCTION public.pgp_sym_encrypt_bytea(bytea, text, text) TO igcuser;
GRANT ALL ON FUNCTION public.pgp_sym_encrypt_bytea(bytea, text, text) TO igc;


--
-- Name: TABLE errorlog; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.errorlog TO igcuser;
GRANT ALL ON TABLE public.errorlog TO igc;


--
-- Name: SEQUENCE "ErrorLog_id_seq"; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public."ErrorLog_id_seq" TO igcuser;
GRANT ALL ON SEQUENCE public."ErrorLog_id_seq" TO igc;


--
-- Name: TABLE jobexecutionlog; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.jobexecutionlog TO igcuser;
GRANT ALL ON TABLE public.jobexecutionlog TO igc;


--
-- Name: SEQUENCE "JobExecutionLog_logid_seq"; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public."JobExecutionLog_logid_seq" TO igcuser;
GRANT ALL ON SEQUENCE public."JobExecutionLog_logid_seq" TO igc;


--
-- Name: TABLE fields; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.fields TO igcuser;
GRANT ALL ON TABLE public.fields TO igc;


--
-- Name: TABLE metricfieldmatcher; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.metricfieldmatcher TO igcuser;
GRANT ALL ON TABLE public.metricfieldmatcher TO igc;


--
-- Name: TABLE metrics; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.metrics TO igcuser;
GRANT ALL ON TABLE public.metrics TO igc;


--
-- Name: SEQUENCE fields_pk_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.fields_pk_id_seq TO igcuser;
GRANT ALL ON SEQUENCE public.fields_pk_id_seq TO igc;


--
-- Name: TABLE kernelfamilies; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.kernelfamilies TO igcuser;
GRANT ALL ON TABLE public.kernelfamilies TO igc;


--
-- Name: TABLE libraries; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.libraries TO igcuser;
GRANT ALL ON TABLE public.libraries TO igc;


--
-- Name: SEQUENCE metricfieldmatcher_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.metricfieldmatcher_id_seq TO igcuser;
GRANT ALL ON SEQUENCE public.metricfieldmatcher_id_seq TO igc;


--
-- Name: SEQUENCE metrics_pk_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.metrics_pk_id_seq TO igcuser;
GRANT ALL ON SEQUENCE public.metrics_pk_id_seq TO igc;


--
-- Name: TABLE opkinds; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.opkinds TO igcuser;
GRANT ALL ON TABLE public.opkinds TO igc;


--
-- Name: TABLE pathregistry; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.pathregistry TO igcuser;
GRANT ALL ON TABLE public.pathregistry TO igc;


--
-- Name: SEQUENCE pathregistry_new_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT ALL ON SEQUENCE public.pathregistry_new_id_seq TO igcuser;
GRANT ALL ON SEQUENCE public.pathregistry_new_id_seq TO igc;


--
-- Name: TABLE trackeddomains; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.trackeddomains TO igcuser;
GRANT ALL ON TABLE public.trackeddomains TO igc;


--
-- Name: DEFAULT PRIVILEGES FOR SEQUENCES; Type: DEFAULT ACL; Schema: public; Owner: postgres
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON SEQUENCES TO igcuser;


--
-- Name: DEFAULT PRIVILEGES FOR SEQUENCES; Type: DEFAULT ACL; Schema: public; Owner: igc
--

ALTER DEFAULT PRIVILEGES FOR ROLE igc IN SCHEMA public GRANT ALL ON SEQUENCES TO igc;


--
-- Name: DEFAULT PRIVILEGES FOR FUNCTIONS; Type: DEFAULT ACL; Schema: public; Owner: postgres
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON FUNCTIONS TO igcuser;


--
-- Name: DEFAULT PRIVILEGES FOR FUNCTIONS; Type: DEFAULT ACL; Schema: public; Owner: igc
--

ALTER DEFAULT PRIVILEGES FOR ROLE igc IN SCHEMA public GRANT ALL ON FUNCTIONS TO igc;


--
-- Name: DEFAULT PRIVILEGES FOR TABLES; Type: DEFAULT ACL; Schema: public; Owner: postgres
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT SELECT,INSERT,DELETE,UPDATE ON TABLES TO igcuser;


--
-- Name: DEFAULT PRIVILEGES FOR TABLES; Type: DEFAULT ACL; Schema: public; Owner: igc
--

ALTER DEFAULT PRIVILEGES FOR ROLE igc IN SCHEMA public GRANT ALL ON TABLES TO igc;


--
-- PostgreSQL database dump complete
--

\unrestrict hV4298B05hISqB1vbycaXFo1wYxe8cQGIAmwcw0Ne0FWEcW7JwggAYpuUqFBr7E

