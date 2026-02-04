-- Migration: Add geographic metadata columns to entities table
-- Required for: Feature 1 - Geographic Proximity Queries
-- Primitives: FILTER_CITY, FILTER_REGION, FILTER_COORDINATES

-- Add geographic columns for place entities
ALTER TABLE entities ADD COLUMN IF NOT EXISTS place_city TEXT;
ALTER TABLE entities ADD COLUMN IF NOT EXISTS place_region TEXT;  -- State/Province
ALTER TABLE entities ADD COLUMN IF NOT EXISTS place_country TEXT;
ALTER TABLE entities ADD COLUMN IF NOT EXISTS place_lat DOUBLE PRECISION;
ALTER TABLE entities ADD COLUMN IF NOT EXISTS place_lng DOUBLE PRECISION;

-- Comments for documentation
COMMENT ON COLUMN entities.place_city IS 'City name for place entities (e.g., "Washington", "Los Angeles")';
COMMENT ON COLUMN entities.place_region IS 'State/Province/Region for place entities (e.g., "California", "Tennessee")';
COMMENT ON COLUMN entities.place_country IS 'Country name or code for place entities (e.g., "USA", "USSR", "France")';
COMMENT ON COLUMN entities.place_lat IS 'Latitude coordinate for place entities (-90 to 90)';
COMMENT ON COLUMN entities.place_lng IS 'Longitude coordinate for place entities (-180 to 180)';

-- Create indexes for geographic queries
CREATE INDEX IF NOT EXISTS idx_entities_place_city ON entities (place_city) WHERE entity_type = 'place';
CREATE INDEX IF NOT EXISTS idx_entities_place_region ON entities (place_region) WHERE entity_type = 'place';
CREATE INDEX IF NOT EXISTS idx_entities_place_country ON entities (place_country) WHERE entity_type = 'place';

-- Create composite index for coordinate queries (bounding box)
CREATE INDEX IF NOT EXISTS idx_entities_place_coords ON entities (place_lat, place_lng) 
WHERE entity_type = 'place' AND place_lat IS NOT NULL AND place_lng IS NOT NULL;

-- Optional: If PostGIS is available, create geography column and spatial index
-- Uncomment the following if PostGIS extension is installed:

-- CREATE EXTENSION IF NOT EXISTS postgis;
-- 
-- ALTER TABLE entities ADD COLUMN IF NOT EXISTS place_geog GEOGRAPHY(POINT, 4326);
-- 
-- UPDATE entities SET place_geog = ST_SetSRID(ST_MakePoint(place_lng, place_lat), 4326)::geography
-- WHERE entity_type = 'place' AND place_lat IS NOT NULL AND place_lng IS NOT NULL;
-- 
-- CREATE INDEX IF NOT EXISTS idx_entities_place_geog ON entities USING GIST (place_geog)
-- WHERE entity_type = 'place';
--
-- -- Trigger to auto-update geography column
-- CREATE OR REPLACE FUNCTION update_place_geog() RETURNS TRIGGER AS $$
-- BEGIN
--     IF NEW.entity_type = 'place' AND NEW.place_lat IS NOT NULL AND NEW.place_lng IS NOT NULL THEN
--         NEW.place_geog := ST_SetSRID(ST_MakePoint(NEW.place_lng, NEW.place_lat), 4326)::geography;
--     ELSE
--         NEW.place_geog := NULL;
--     END IF;
--     RETURN NEW;
-- END;
-- $$ LANGUAGE plpgsql;
--
-- DROP TRIGGER IF EXISTS trg_update_place_geog ON entities;
-- CREATE TRIGGER trg_update_place_geog BEFORE INSERT OR UPDATE ON entities
-- FOR EACH ROW EXECUTE FUNCTION update_place_geog();
