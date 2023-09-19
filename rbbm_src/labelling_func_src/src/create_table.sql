--
-- PostgreSQL database dump
--

-- Dumped from database version 14.4
-- Dumped by pg_dump version 14.4

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: adult500; Type: TABLE; Schema: public; Owner: holocleanuser
--

CREATE TABLE public.adult500 (
    _tid_ bigint,
    age text,
    workclass text,
    education text,
    "marital-status" text,
    occupation text,
    relationship text,
    race text,
    sex text,
    "hours-per-week" text,
    "native-country" text,
    income text
);


ALTER TABLE public.adult500 OWNER TO holocleanuser;

--
-- Name: adult500_0; Type: INDEX; Schema: public; Owner: holocleanuser
--

CREATE INDEX adult500_0 ON public.adult500 USING btree (age);


--
-- Name: adult500_1; Type: INDEX; Schema: public; Owner: holocleanuser
--

CREATE INDEX adult500_1 ON public.adult500 USING btree (workclass);


--
-- Name: adult500_10; Type: INDEX; Schema: public; Owner: holocleanuser
--

CREATE INDEX adult500_10 ON public.adult500 USING btree (income);


--
-- Name: adult500_2; Type: INDEX; Schema: public; Owner: holocleanuser
--

CREATE INDEX adult500_2 ON public.adult500 USING btree (education);


--
-- Name: adult500_3; Type: INDEX; Schema: public; Owner: holocleanuser
--

CREATE INDEX adult500_3 ON public.adult500 USING btree ("marital-status");


--
-- Name: adult500_4; Type: INDEX; Schema: public; Owner: holocleanuser
--

CREATE INDEX adult500_4 ON public.adult500 USING btree (occupation);


--
-- Name: adult500_5; Type: INDEX; Schema: public; Owner: holocleanuser
--

CREATE INDEX adult500_5 ON public.adult500 USING btree (relationship);


--
-- Name: adult500_6; Type: INDEX; Schema: public; Owner: holocleanuser
--

CREATE INDEX adult500_6 ON public.adult500 USING btree (race);


--
-- Name: adult500_7; Type: INDEX; Schema: public; Owner: holocleanuser
--

CREATE INDEX adult500_7 ON public.adult500 USING btree (sex);


--
-- Name: adult500_8; Type: INDEX; Schema: public; Owner: holocleanuser
--

CREATE INDEX adult500_8 ON public.adult500 USING btree ("hours-per-week");


--
-- Name: adult500_9; Type: INDEX; Schema: public; Owner: holocleanuser
--

CREATE INDEX adult500_9 ON public.adult500 USING btree ("native-country");


--
-- PostgreSQL database dump complete
--

