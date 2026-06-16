-- Type to hold gap results
DO $$ BEGIN
    CREATE TYPE gap_range AS (gap_start TIMESTAMPTZ, gap_end TIMESTAMPTZ);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE OR REPLACE FUNCTION _bisect_right(
    p_arr       TIMESTAMPTZ[],
    p_target    TIMESTAMPTZ
)
RETURNS INT
LANGUAGE plpgsql IMMUTABLE
AS $$
DECLARE
    lo INT := 1;
    hi INT := array_length(p_arr, 1);
    mid INT;
BEGIN
    IF hi IS NULL THEN RETURN 0; END IF;
    WHILE lo <= hi LOOP
        mid := (lo + hi) / 2;
        IF p_arr[mid] <= p_target THEN
            lo := mid + 1;
        ELSE
            hi := mid - 1;
        END IF;
    END LOOP;
    RETURN hi;  -- last index <= target
END;
$$;

-- Main entry point: finds gaps and merges adjacent ones
CREATE OR REPLACE FUNCTION find_ohlcv_gaps(
    p_start_date    TIMESTAMPTZ,
    p_end_date      TIMESTAMPTZ,
    p_timeframe     TEXT,            -- '1h', '1m', '1s', '1d', etc.
    p_exchange      TEXT,
    p_symbol        TEXT
)
RETURNS SETOF gap_range
LANGUAGE plpgsql STABLE
AS $$
DECLARE
    v_interval      INTERVAL;
    v_raw_gaps      gap_range[];
    v_merged        gap_range[];
    v_prev          gap_range;
    v_curr          gap_range;
    i               INT;
BEGIN
    v_interval := _timeframe_to_interval(p_timeframe);

    -- Collect timestamps present in the table
    -- Pass them into the recursive worker
    SELECT _find_gaps_recursive(
        p_start_date,
        p_end_date,
        v_interval,
        ARRAY(
            SELECT timestamp
            FROM ohlcv
            WHERE exchange  = p_exchange
              AND symbol    = p_symbol
              AND timeframe = p_timeframe
              AND timestamp >= p_start_date
              AND timestamp <= p_end_date
            ORDER BY timestamp
        )
    ) INTO v_raw_gaps;

    -- Merge adjacent/overlapping gaps
    IF array_length(v_raw_gaps, 1) IS NULL THEN
        RETURN;
    END IF;

    v_merged := ARRAY[v_raw_gaps[1]];

    FOR i IN 2 .. array_length(v_raw_gaps, 1) LOOP
        v_curr := v_raw_gaps[i];
        v_prev := v_merged[array_length(v_merged, 1)];
        IF v_curr.gap_start <= v_prev.gap_end + v_interval THEN
            -- Extend previous gap
            v_merged[array_length(v_merged, 1)] :=
                ROW(v_prev.gap_start, v_curr.gap_end)::gap_range;
        ELSE
            v_merged := v_merged || v_curr;
        END IF;
    END LOOP;

    IF v_merged IS NOT NULL THEN
        FOR i IN 1 .. array_length(v_merged, 1) LOOP
            -- Only return gaps that have a duration
            IF v_merged[i].gap_start < v_merged[i].gap_end THEN
                RETURN NEXT v_merged[i];
            END IF;
        END LOOP;
    END IF;
END;
$$;


CREATE OR REPLACE FUNCTION _find_gaps_recursive(
    p_start         TIMESTAMPTZ,
    p_end           TIMESTAMPTZ,
    p_interval      INTERVAL,
    p_timestamps    TIMESTAMPTZ[]
)
RETURNS gap_range[]
LANGUAGE plpgsql STABLE
AS $$
DECLARE
    v_expected_count BIGINT;
    v_present_count  INT;
    v_half_ts        TIMESTAMPTZ;
    v_split_idx      BIGINT;
    v_half_offset    BIGINT;
    v_left           TIMESTAMPTZ[];
    v_right          TIMESTAMPTZ[];
BEGIN
    v_expected_count := _count_intervals(p_start, p_end, p_interval);
    v_present_count  := COALESCE(array_length(p_timestamps, 1), 0);

    -- Базовий випадок 1: всі дані на місці, дір немає
    IF v_present_count = v_expected_count THEN
        RETURN ARRAY[]::gap_range[];
    END IF;

    -- Базовий випадок 2: повністю порожній відрізок (усі дані пропущені)
    IF v_present_count = 0 THEN
        RETURN ARRAY[ROW(p_start, p_end)::gap_range];
    END IF;

    -- ВИПРАВЛЕНО: Ділимо навпіл кількість інтервалів, а не масиви
    v_half_offset := FLOOR(v_expected_count / 2);
    v_half_ts     := p_start + (v_half_offset - 1) * p_interval;

    -- Бінарний пошук індексу розбиття масиву міток часу
    v_split_idx := _bisect_right(p_timestamps, v_half_ts);

    -- Розрізаємо масив на ліву та праву частини
    v_left  := p_timestamps[1 : v_split_idx];
    v_right := p_timestamps[v_split_idx + 1 : array_length(p_timestamps, 1)];

    -- Рекурсивно шукаємо в обох половинках та об'єднуємо результати
    RETURN _find_gaps_recursive(p_start, v_half_ts, p_interval, v_left)
        || _find_gaps_recursive(v_half_ts + p_interval, p_end, p_interval, v_right);
END;
$$;

-- Convert timeframe string like '1h', '5m', '1s', '1d' to INTERVAL
CREATE OR REPLACE FUNCTION _timeframe_to_interval(p_timeframe TEXT)
RETURNS INTERVAL
LANGUAGE plpgsql IMMUTABLE
AS $$
DECLARE
    v_num  INT;
    v_unit TEXT;
BEGIN
    v_num  := (regexp_match(p_timeframe, '^(\d+)'))[1]::INT;
    v_unit := (regexp_match(p_timeframe, '(\D+)$'))[1];

    RETURN CASE v_unit
        WHEN 's'   THEN make_interval(secs  => v_num)
        WHEN 'm'   THEN make_interval(mins  => v_num)
        WHEN 'h'   THEN make_interval(hours => v_num)
        WHEN 'd'   THEN make_interval(days  => v_num)
        WHEN 'w'   THEN make_interval(weeks => v_num)
        WHEN 'M'   THEN make_interval(months => v_num)
        ELSE             p_timeframe::INTERVAL  -- fallback: let PG parse it
    END;
END;
$$;


-- Count how many intervals fit in [start, end] inclusive
CREATE OR REPLACE FUNCTION _count_intervals(
    p_start    TIMESTAMPTZ,
    p_end      TIMESTAMPTZ,
    p_interval INTERVAL
)
RETURNS BIGINT
LANGUAGE SQL IMMUTABLE
AS $$
    SELECT EXTRACT(EPOCH FROM (p_end - p_start))::BIGINT
         / EXTRACT(EPOCH FROM p_interval)::BIGINT
    + 1;
$$;


-- ============================================================
-- Usage examples
-- ============================================================

-- Find all gaps in BTC/USDT 1h data for January 2020:
--
--   SELECT * FROM find_ohlcv_gaps(
--       '2020-01-01'::timestamptz,
--       '2020-02-01'::timestamptz,
--       '1h',
--       'binance',
--       'BTC/USDT'
--   );
--
-- With right-side bias disabled:
--
--   SELECT * FROM find_ohlcv_gaps(
--       '2020-01-01'::timestamptz,
--       '2020-02-01'::timestamptz,
--       '1h',
--       'binance',
--       'BTC/USDT',
--       FALSE
--   );