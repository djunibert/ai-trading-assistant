-- =========================================================
-- SCHÉMA POSTGRESQL
-- AI TRADING SYSTEM
-- =========================================================

-- Table des plateformes de données
CREATE TABLE IF NOT EXISTS platforms (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Table des symboles
CREATE TABLE IF NOT EXISTS symbols (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL UNIQUE,
    description VARCHAR(255),
    asset_class VARCHAR(50),
    exchange VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Table des unités de temps
CREATE TABLE IF NOT EXISTS timeframes (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(20) NOT NULL UNIQUE,
    minutes INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Table principale OHLCV
CREATE TABLE IF NOT EXISTS market_bars (
    id BIGSERIAL PRIMARY KEY,

    platform_id BIGINT NOT NULL,
    symbol_id BIGINT NOT NULL,
    timeframe_id BIGINT NOT NULL,

    timestamp TIMESTAMPTZ NOT NULL,

    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION,

    contract_symbol VARCHAR(50),
    file_name VARCHAR(255),

    quality_status VARCHAR(20) NOT NULL DEFAULT 'raw',

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_market_bars_platform
        FOREIGN KEY (platform_id)
        REFERENCES platforms(id),

    CONSTRAINT fk_market_bars_symbol
        FOREIGN KEY (symbol_id)
        REFERENCES symbols(id),

    CONSTRAINT fk_market_bars_timeframe
        FOREIGN KEY (timeframe_id)
        REFERENCES timeframes(id),

    CONSTRAINT uq_market_bars
        UNIQUE (
            platform_id,
            symbol_id,
            timeframe_id,
            timestamp
        ),

    CONSTRAINT chk_market_bars_high
        CHECK (high >= open AND high >= close AND high >= low),

    CONSTRAINT chk_market_bars_low
        CHECK (low <= open AND low <= close AND low <= high),

    CONSTRAINT chk_market_bars_volume
        CHECK (volume IS NULL OR volume >= 0)
);

-- Journal des imports
CREATE TABLE IF NOT EXISTS data_import_log (
    id BIGSERIAL PRIMARY KEY,

    platform_id BIGINT,
    file_name VARCHAR(255) NOT NULL,
    symbol VARCHAR(50),
    timeframe VARCHAR(20),

    rows_read INTEGER NOT NULL DEFAULT 0,
    rows_inserted INTEGER NOT NULL DEFAULT 0,
    rows_updated INTEGER NOT NULL DEFAULT 0,
    rows_rejected INTEGER NOT NULL DEFAULT 0,

    status VARCHAR(20) NOT NULL DEFAULT 'started',
    error_message TEXT,

    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    CONSTRAINT fk_import_log_platform
        FOREIGN KEY (platform_id)
        REFERENCES platforms(id)
);

-- Prédictions des modèles
CREATE TABLE IF NOT EXISTS model_predictions (
    id BIGSERIAL PRIMARY KEY,

    symbol_id BIGINT NOT NULL,
    timeframe_id BIGINT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,

    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),

    prediction INTEGER NOT NULL,
    confidence DOUBLE PRECISION,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_predictions_symbol
        FOREIGN KEY (symbol_id)
        REFERENCES symbols(id),

    CONSTRAINT fk_predictions_timeframe
        FOREIGN KEY (timeframe_id)
        REFERENCES timeframes(id)
);

-- Trades simulés ou réels
CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,

    symbol_id BIGINT NOT NULL,
    timeframe_id BIGINT NOT NULL,

    signal VARCHAR(20) NOT NULL,

    entry_timestamp TIMESTAMPTZ NOT NULL,
    exit_timestamp TIMESTAMPTZ,

    entry_price DOUBLE PRECISION NOT NULL,
    exit_price DOUBLE PRECISION,

    stop_loss DOUBLE PRECISION,
    take_profit DOUBLE PRECISION,

    position_size DOUBLE PRECISION,
    pnl DOUBLE PRECISION,

    status VARCHAR(20) NOT NULL DEFAULT 'OPEN',
    source VARCHAR(30) NOT NULL DEFAULT 'BACKTEST',

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_trades_symbol
        FOREIGN KEY (symbol_id)
        REFERENCES symbols(id),

    CONSTRAINT fk_trades_timeframe
        FOREIGN KEY (timeframe_id)
        REFERENCES timeframes(id)
);

-- =========================================================
-- INDEX
-- =========================================================

CREATE INDEX IF NOT EXISTS idx_market_bars_lookup
ON market_bars (
    symbol_id,
    timeframe_id,
    timestamp
);

CREATE INDEX IF NOT EXISTS idx_market_bars_platform
ON market_bars (
    platform_id,
    timestamp
);

CREATE INDEX IF NOT EXISTS idx_market_bars_timestamp
ON market_bars (
    timestamp
);

CREATE INDEX IF NOT EXISTS idx_predictions_lookup
ON model_predictions (
    symbol_id,
    timeframe_id,
    timestamp
);

CREATE INDEX IF NOT EXISTS idx_trades_lookup
ON trades (
    symbol_id,
    timeframe_id,
    entry_timestamp
);

-- =========================================================
-- TRIGGER updated_at
-- =========================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_market_bars_updated_at ON market_bars;

CREATE TRIGGER trg_market_bars_updated_at
BEFORE UPDATE ON market_bars
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trg_trades_updated_at ON trades;

CREATE TRIGGER trg_trades_updated_at
BEFORE UPDATE ON trades
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- =========================================================
-- DONNÉES DE RÉFÉRENCE
-- =========================================================

INSERT INTO platforms (name)
VALUES
    ('TradingView'),
    ('Yahoo'),
    ('Databento')
ON CONFLICT (name) DO NOTHING;

INSERT INTO timeframes (name, minutes)
VALUES
    ('1m', 1),
    ('5m', 5),
    ('15m', 15),
    ('30m', 30),
    ('1h', 60),
    ('4h', 240),
    ('1d', 1440)
ON CONFLICT (name) DO NOTHING;